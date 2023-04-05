import os
import logging
import asyncio
import traceback
import html
import json
import tempfile
import pydub
from pathlib import Path
from datetime import datetime, timedelta

import telegram
from telegram import (
    Update, 
    User, 
    InlineKeyboardButton, 
    InlineKeyboardMarkup, 
    BotCommand
)
from telegram.ext import (
    Application,
    ApplicationBuilder,
    CallbackContext,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    AIORateLimiter,
    filters
)
from telegram.constants import ParseMode, ChatAction

import config
import database
import openai_utils

from ym import ymquickpay
from ym import checkpay

# setup
db = database.Database()
logger = logging.getLogger(__name__)
user_semaphores = {}


HELP_MESSAGE = """Список команд:
⚪ /retry – Регенерировать предидущий запрос
⚪ /new – Начать новый диалог
⚪ /mode – Выбрать режим работы бота
⚪ /settings – Показать настройки
⚪ /balance – Показать баланс
⚪ /help – Помощь
⚪ /premium - Подписка на премиум
"""


def split_text_into_chunks(text, chunk_size):
    for i in range(0, len(text), chunk_size):
        yield text[i:i + chunk_size]


async def register_user_if_not_exists(update: Update, context: CallbackContext, user: User):
    if not db.check_if_user_exists(user.id):
        db.add_new_user(
            user.id,
            update.message.chat_id,
            username=user.username,
            first_name=user.first_name,
            last_name= user.last_name
        )
        db.start_new_dialog(user.id)

    if db.get_user_attribute(user.id, "current_dialog_id") is None:
        db.start_new_dialog(user.id)

    if user.id not in user_semaphores:
        user_semaphores[user.id] = asyncio.Semaphore(1)

    if db.get_user_attribute(user.id, "current_model") is None:
        db.set_user_attribute(user.id, "current_model", config.models["available_text_models"][0])

    # back compatibility for n_used_tokens field
    n_used_tokens = db.get_user_attribute(user.id, "n_used_tokens")
    if isinstance(n_used_tokens, int):  # old format
        new_n_used_tokens = {
            "gpt-3.5-turbo": {
                "n_input_tokens": 0,
                "n_output_tokens": n_used_tokens
            }
        }
        db.set_user_attribute(user.id, "n_used_tokens", new_n_used_tokens)

    # voice message transcription
    if db.get_user_attribute(user.id, "n_transcribed_seconds") is None:
        db.set_user_attribute(user.id, "n_transcribed_seconds", 0.0)


async def start_handle(update: Update, context: CallbackContext):
    await register_user_if_not_exists(update, context, update.message.from_user)
    user_id = update.message.from_user.id
    
    db.set_user_attribute(user_id, "last_interaction", datetime.now())
    db.start_new_dialog(user_id)
    
    reply_text = "Привет Я <b>ChatGPT Telegram бот</b> я работаю с помощью GPT-3.5 Turbo OpenAI API 🤖, ты можешь использовать команды и обращаться ко мне на Русском или Английском языке\n\n"
    reply_text += HELP_MESSAGE

    reply_text += "\nАсейчас... спроси меня о чем нибудь!"
    
    await update.message.reply_text(reply_text, parse_mode=ParseMode.HTML)


async def help_handle(update: Update, context: CallbackContext):
    await register_user_if_not_exists(update, context, update.message.from_user)
    user_id = update.message.from_user.id
    db.set_user_attribute(user_id, "last_interaction", datetime.now())
    await update.message.reply_text(HELP_MESSAGE, parse_mode=ParseMode.HTML)


async def retry_handle(update: Update, context: CallbackContext):
    await register_user_if_not_exists(update, context, update.message.from_user)
    if await is_previous_message_not_answered_yet(update, context): return
    
    user_id = update.message.from_user.id
    db.set_user_attribute(user_id, "last_interaction", datetime.now())

    dialog_messages = db.get_dialog_messages(user_id, dialog_id=None)
    if len(dialog_messages) == 0:
        await update.message.reply_text("Нет сообщений 🤷‍♂️")
        return

    last_dialog_message = dialog_messages.pop()
    db.set_dialog_messages(user_id, dialog_messages, dialog_id=None)  # last message was removed from the context

    await message_handle(update, context, message=last_dialog_message["user"], use_new_dialog_timeout=False)


async def message_handle(update: Update, context: CallbackContext, message=None, use_new_dialog_timeout=True):
    # check if message is edited
    if update.edited_message is not None:
        await edited_message_handle(update, context)
        return
        
    await register_user_if_not_exists(update, context, update.message.from_user)
    if await is_previous_message_not_answered_yet(update, context): return
    
    
    user_id = update.message.from_user.id
    chat_mode = db.get_user_attribute(user_id, "current_chat_mode")
    await check_user_request(user_id)
    user_request_count = db.get_user_attribute(user_id, "user_request_count")
    
    if user_request_count is None:
        db.add_new_message_count(user_id)
    
    
    if user_request_count <= 0:
         # TODO: проверка подписки пользователя
             await update.message.reply_text("Вы превысили лимит запросов. Остаток запросов " + str(user_request_count))
             return
    
    db.minus_message_count(user_id)
    
    async with user_semaphores[user_id]:
        # new dialog timeout
        if use_new_dialog_timeout:
            if (datetime.now() - db.get_user_attribute(user_id, "last_interaction")).seconds > config.new_dialog_timeout and len(db.get_dialog_messages(user_id)) > 0:
                db.start_new_dialog(user_id)
                await update.message.reply_text(f"Начало нового диалога из-за тайм-аута (<b>{openai_utils.CHAT_MODES[chat_mode]['name']}</b> mode) ✅", parse_mode=ParseMode.HTML)
        db.set_user_attribute(user_id, "last_interaction", datetime.now())

        # send typing action
        await update.message.chat.send_action(action="typing")

        try:
            message = message or update.message.text

            current_model = db.get_user_attribute(user_id, "current_model")
            dialog_messages = db.get_dialog_messages(user_id, dialog_id=None)
            parse_mode = {
                "html": ParseMode.HTML,
                "markdown": ParseMode.MARKDOWN
            }[openai_utils.CHAT_MODES[chat_mode]["parse_mode"]]

            chatgpt_instance = openai_utils.ChatGPT(model=current_model)
            if config.enable_message_streaming:
                gen = chatgpt_instance.send_message_stream(message, dialog_messages=dialog_messages, chat_mode=chat_mode)
            else:
                answer, (n_input_tokens, n_output_tokens), n_first_dialog_messages_removed = await chatgpt_instance.send_message(
                    message,
                    dialog_messages=dialog_messages,
                    chat_mode=chat_mode
                )

                async def fake_gen():
                    yield "finished", answer, (n_input_tokens, n_output_tokens), n_first_dialog_messages_removed

                gen = fake_gen()

            # send message to user
            prev_answer = ""
            i = -1
            async for gen_item in gen:
                i += 1

                status = gen_item[0]
                if status == "not_finished":
                    status, answer = gen_item
                elif status == "finished":
                    status, answer, (n_input_tokens, n_output_tokens), n_first_dialog_messages_removed = gen_item
                else:
                    raise ValueError(f"Статус {status} неизвестен")

                answer = answer[:4096]  # telegram message limit
                if i == 0:  # send first message (then it'll be edited if message streaming is enabled)
                    try:                    
                        sent_message = await update.message.reply_text(answer, parse_mode=parse_mode)
                    except telegram.error.BadRequest as e:
                        if str(e).startswith("Сообщение должно быть непустым"):  # first answer chunk from openai was empty
                            i = -1  # try again to send first message
                            continue
                        else:
                            sent_message = await update.message.reply_text(answer)
                else:  # edit sent message
                    # update only when 100 new symbols are ready
                    if abs(len(answer) - len(prev_answer)) < 100 and status != "finished":
                        continue

                    try:                    
                        await context.bot.edit_message_text(answer, chat_id=sent_message.chat_id, message_id=sent_message.message_id, parse_mode=parse_mode)
                    except telegram.error.BadRequest as e:
                        if str(e).startswith("Сообщение не изменено"):
                            continue
                        else:
                            await context.bot.edit_message_text(answer, chat_id=sent_message.chat_id, message_id=sent_message.message_id)

                    await asyncio.sleep(0.01)  # wait a bit to avoid flooding
                    
                prev_answer = answer

            # update user data
            new_dialog_message = {"user": message, "bot": answer, "date": datetime.now()}
            db.set_dialog_messages(
                user_id,
                db.get_dialog_messages(user_id, dialog_id=None) + [new_dialog_message],
                dialog_id=None
            )

            db.update_n_used_tokens(user_id, current_model, n_input_tokens, n_output_tokens) 
        except Exception as e:
            error_text = f"Что-то пошло не так во время завершения. Причина: {e}"
            logger.error(error_text)
            await update.message.reply_text(error_text)
            return

        # send message if some messages were removed from the context
        if n_first_dialog_messages_removed > 0:
            if n_first_dialog_messages_removed == 1:
                text = "✍️ <i>Note:</i> Ваш текущий диалог слишком длинный, поэтому ваше <b>первое сообщение</b> было удалено из контекста.\n Отправьте команду /new, чтобы начать новый диалог"
            else:
                text = f"✍️ <i>Примечание.</i> Ваш текущий диалог слишком длинный, поэтому <b>{n_first_dialog_messages_removed} первые сообщения</b> были удалены из контекста.\n Отправьте команду /new, чтобы начать новый диалог"
            await update.message.reply_text(text, parse_mode=ParseMode.HTML)


async def is_previous_message_not_answered_yet(update: Update, context: CallbackContext):
    await register_user_if_not_exists(update, context, update.message.from_user)

    user_id = update.message.from_user.id
    if user_semaphores[user_id].locked():
        text = "⏳ <b>Подождите</b> ответа на предыдущее сообщение"
        await update.message.reply_text(text, reply_to_message_id=update.message.id, parse_mode=ParseMode.HTML)
        return True
    else:
        return False


async def voice_message_handle(update: Update, context: CallbackContext):
    await register_user_if_not_exists(update, context, update.message.from_user)
    if await is_previous_message_not_answered_yet(update, context): return

    user_id = update.message.from_user.id
    db.set_user_attribute(user_id, "last_interaction", datetime.now())

    voice = update.message.voice
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_dir = Path(tmp_dir)
        voice_ogg_path = tmp_dir / "voice.ogg"

        # download
        voice_file = await context.bot.get_file(voice.file_id)
        await voice_file.download_to_drive(voice_ogg_path)

        # convert to mp3
        voice_mp3_path = tmp_dir / "voice.mp3"
        pydub.AudioSegment.from_file(voice_ogg_path).export(voice_mp3_path, format="mp3")

        # transcribe
        with open(voice_mp3_path, "rb") as f:
            transcribed_text = await openai_utils.transcribe_audio(f)

    text = f"🎤: <i>{transcribed_text}</i>"
    await update.message.reply_text(text, parse_mode=ParseMode.HTML)

    # update n_transcribed_seconds
    db.set_user_attribute(user_id, "n_transcribed_seconds", voice.duration + db.get_user_attribute(user_id, "n_transcribed_seconds"))

    await message_handle(update, context, message=transcribed_text)


async def new_dialog_handle(update: Update, context: CallbackContext):
    await register_user_if_not_exists(update, context, update.message.from_user)
    if await is_previous_message_not_answered_yet(update, context): return

    user_id = update.message.from_user.id
    db.set_user_attribute(user_id, "last_interaction", datetime.now())

    db.start_new_dialog(user_id)
    await update.message.reply_text("Начитаем новый диалог ✅")

    chat_mode = db.get_user_attribute(user_id, "current_chat_mode")
    await update.message.reply_text(f"{openai_utils.CHAT_MODES[chat_mode]['welcome_message']}", parse_mode=ParseMode.HTML)


async def show_chat_modes_handle(update: Update, context: CallbackContext):
    await register_user_if_not_exists(update, context, update.message.from_user)
    if await is_previous_message_not_answered_yet(update, context): return

    user_id = update.message.from_user.id
    db.set_user_attribute(user_id, "last_interaction", datetime.now())

    keyboard = []
    for chat_mode, chat_mode_dict in openai_utils.CHAT_MODES.items():
        keyboard.append([InlineKeyboardButton(chat_mode_dict["name"], callback_data=f"set_chat_mode|{chat_mode}")])
    reply_markup = InlineKeyboardMarkup(keyboard)

    await update.message.reply_text("Выберите режим бота:", reply_markup=reply_markup)


async def set_chat_mode_handle(update: Update, context: CallbackContext):
    await register_user_if_not_exists(update.callback_query, context, update.callback_query.from_user)
    user_id = update.callback_query.from_user.id

    query = update.callback_query
    await query.answer()

    chat_mode = query.data.split("|")[1]

    db.set_user_attribute(user_id, "current_chat_mode", chat_mode)
    db.start_new_dialog(user_id)

    await query.edit_message_text(f"{openai_utils.CHAT_MODES[chat_mode]['welcome_message']}", parse_mode=ParseMode.HTML)


def get_settings_menu(user_id: int):
    current_model = db.get_user_attribute(user_id, "current_model")
    text = config.models["info"][current_model]["description"]

    text += "\n\n"
    score_dict = config.models["info"][current_model]["scores"]
    for score_key, score_value in score_dict.items():
        text += "🟢" * score_value + "⚪️" * (5 - score_value) + f" – {score_key}\n\n"

    text += "\nSelect <b>model</b>:"

    # buttons to choose models
    buttons = []
    for model_key in config.models["available_text_models"]:
        title = config.models["info"][model_key]["name"]
        if model_key == current_model:
            title = "✅ " + title

        buttons.append(
            InlineKeyboardButton(title, callback_data=f"set_settings|{model_key}")
        )
    reply_markup = InlineKeyboardMarkup([buttons])

    return text, reply_markup


async def settings_handle(update: Update, context: CallbackContext):
    await register_user_if_not_exists(update, context, update.message.from_user)
    if await is_previous_message_not_answered_yet(update, context): return

    user_id = update.message.from_user.id
    db.set_user_attribute(user_id, "last_interaction", datetime.now())

    text, reply_markup = get_settings_menu(user_id)
    await update.message.reply_text(text, reply_markup=reply_markup, parse_mode=ParseMode.HTML)


async def set_settings_handle(update: Update, context: CallbackContext):
    await register_user_if_not_exists(update.callback_query, context, update.callback_query.from_user)
    user_id = update.callback_query.from_user.id

    query = update.callback_query
    await query.answer()

    _, model_key = query.data.split("|")
    db.set_user_attribute(user_id, "current_model", model_key)
    db.start_new_dialog(user_id)

    text, reply_markup = get_settings_menu(user_id)
    try:                    
        await query.edit_message_text(text, reply_markup=reply_markup, parse_mode=ParseMode.HTML)
    except telegram.error.BadRequest as e:
        if str(e).startswith("Сообщение не изменено"):
            pass
    

async def show_balance_handle(update: Update, context: CallbackContext):
    await register_user_if_not_exists(update, context, update.message.from_user)

    user_id = update.message.from_user.id
    db.set_user_attribute(user_id, "last_interaction", datetime.now())

    # count total usage statistics
    total_n_spent_dollars = 0
    total_n_used_tokens = 0

    n_used_tokens_dict = db.get_user_attribute(user_id, "n_used_tokens")
    n_transcribed_seconds = db.get_user_attribute(user_id, "n_transcribed_seconds")

    details_text = "🏷️ Details:\n"
    for model_key in sorted(n_used_tokens_dict.keys()):
        n_input_tokens, n_output_tokens = n_used_tokens_dict[model_key]["n_input_tokens"], n_used_tokens_dict[model_key]["n_output_tokens"]
        total_n_used_tokens += n_input_tokens + n_output_tokens

        n_input_spent_dollars = config.models["info"][model_key]["price_per_1000_input_tokens"] * (n_input_tokens / 1000)
        n_output_spent_dollars = config.models["info"][model_key]["price_per_1000_output_tokens"] * (n_output_tokens / 1000)
        total_n_spent_dollars += n_input_spent_dollars + n_output_spent_dollars

        details_text += f"- {model_key}: <b>{n_input_spent_dollars + n_output_spent_dollars:.03f}$</b> / <b>{n_input_tokens + n_output_tokens} tokens</b>\n"

    voice_recognition_n_spent_dollars = config.models["info"]["whisper"]["price_per_1_min"] * (n_transcribed_seconds / 60)
    if n_transcribed_seconds != 0:
        details_text += f"- Whisper (voice recognition): <b>{voice_recognition_n_spent_dollars:.03f}$</b> / <b>{n_transcribed_seconds:.01f} seconds</b>\n"
    
    total_n_spent_dollars += voice_recognition_n_spent_dollars    

    text = f"Вы потратили <b>{total_n_spent_dollars:.03f}$</b>\n"
    text += f"Вы использовали <b>{total_n_used_tokens}</b> tokens\n\n"
    text += details_text

    await update.message.reply_text(text, parse_mode=ParseMode.HTML)
    
#Проверка не наступила ли дата обновления количества запросов
async def check_user_request(user_id: int):
    user_request_count = db.get_user_attribute(user_id, "user_request_count")
    user_request_date = db.get_user_attribute(user_id, "user_request_date")

    if user_request_count is None:
        db.add_new_message_count(user_id, 0)

    if user_request_date is None or not isinstance(user_request_date, datetime):
        user_request_date = datetime.now()
        db.add_new_message_date(user_id, user_request_date)
    else:
        # user_request_date is already a datetime object, no need to convert it
        pass

    if (user_request_date.date() == datetime.now().date()):
        return
            
    

    if user_request_date.date() < datetime.now().date():
        is_premium = check_prem_pay(user_id)
        if is_premium:
            db.add_new_message_count(user_id, 100)
        else:
            db.add_new_message_count(user_id, 3)

        db.add_new_message_date(user_id)

        
        

async def check_prem_pay(_user_id: int):
    user_id = _user_id
    db.set_user_attribute(user_id, "last_interaction", datetime.now())
    
    
    prem_datetime = db.get_user_attribute(user_id, "prem_datetime")
    if prem_datetime is None:
        db.add_new_prem_date(user_id)
    
    prem_mounts = db.get_user_attribute(user_id, "prem_mounts")
    if prem_mounts is None:
        db.add_new_prem_days(user_id)
    
    prem_datetime = db.get_user_attribute(user_id, "prem_datetime")
    print(prem_datetime)
    label = str(user_id)
    pay_list = checkpay.ympaycheck(label)
    if len(pay_list) > 0:
        for pay_date in pay_list:
            if pay_date > prem_datetime:
                db.add_new_prem_date(user_id, pay_date)
                db.add_new_prem_days(user_id, 30)
                return True
            else:
                return False
            
        
    
async def check_prem(_user_id: int):
    
    user_id = _user_id
    
    prem_datetime = db.get_user_attribute(user_id, "prem_datetime")
    
    if prem_datetime is None:
        db.add_new_prem_date(user_id)
    
    prem_mounts = db.get_user_attribute(user_id, "prem_mounts")
    if prem_mounts is None:
        db.add_new_prem_days(user_id)
    
    # преобразование строки даты и времени в объект datetime
    if isinstance(prem_datetime, str):
        prem_datetime = datetime.strptime(prem_datetime, "%Y-%m-%d %H:%M:%S")

    # вычисление даты окончания премиум-подписки
    prem_end_date = prem_datetime + timedelta(days=prem_mounts)
    # вывод даты окончания премиум-подписки
    #print(prem_end_date.strftime("%Y-%m-%d %H:%M:%S"))
    if (prem_end_date > datetime.now()):
        return True, prem_end_date
    else:
        return False, datetime.now()

async def show_premium_handle(update: Update, context: CallbackContext):
    await register_user_if_not_exists(update, context, update.message.from_user)

    user_id = update.message.from_user.id 
    db.set_user_attribute(user_id, "last_interaction", datetime.now())
    
    check_pr = await check_prem_pay(user_id)
    prem_status, prem_end = await check_prem(user_id)
    
    
    if (prem_status):
        keyboard = []
        keyboard.append([InlineKeyboardButton(text="Картой 250р на 1 месяц", url=ymquickpay.ympay1m(user_id))])
        reply_markup = InlineKeyboardMarkup(keyboard)
        prem_end_date_str = prem_end.date().strftime("%Y-%m-%d")
        await check_user_request(user_id)
        user_request_count = db.get_user_attribute(user_id, "user_request_count")
        if check_pr:
            db.add_new_message_count(user_id, 100)
            db.add_new_message_date(user_id)
            await update.message.reply_html("Поздравляем! Платеж успешно проведён. У вас активная подписка до <b>{date}</b>\nКоличество оставшихся сообщений в эти сутки <b>{request}</b>".format(date=prem_end_date_str, request=user_request_count), reply_markup=reply_markup)
            
        else:
            await update.message.reply_html("У вас активная подписка до <b>{date}</b>\nКоличество оставшихся сообщений в эти сутки <b>{request}</b>. Нажмите /premuim для проверки новых платежей".format(date=prem_end_date_str, request=user_request_count), reply_markup=reply_markup)
    else:
        keyboard = []
        keyboard.append([InlineKeyboardButton(text="Картой 250р на 1 месяц", url=ymquickpay.ympay1m(user_id))])
        reply_markup = InlineKeyboardMarkup(keyboard)
        await update.message.reply_text("Нажмите /premuim для проверки новых платежей или Выберите подписку и способ оплаты:", reply_markup=reply_markup)


async def edited_message_handle(update: Update, context: CallbackContext):
    text = "🥲 К сожалению, <b>редактирование</b> сообщений не поддерживается"
    await update.edited_message.reply_text(text, parse_mode=ParseMode.HTML)


async def error_handle(update: Update, context: CallbackContext) -> None:
    logger.error(msg="Исключение при обработке обновления:", exc_info=context.error)

    try:
        # collect error message
        tb_list = traceback.format_exception(None, context.error, context.error.__traceback__)
        tb_string = "".join(tb_list)
        update_str = update.to_dict() if isinstance(update, Update) else str(update)
        message = (
            f"При обработке обновления возникло исключение\n"
            f"<pre>update = {html.escape(json.dumps(update_str, indent=2, ensure_ascii=False))}"
            "</pre>\n\n"
            f"<pre>{html.escape(tb_string)}</pre>"
        )

        # split text into multiple messages due to 4096 character limit
        for message_chunk in split_text_into_chunks(message, 4096):
            try:
                await context.bot.send_message(update.effective_chat.id, message_chunk, parse_mode=ParseMode.HTML)
            except telegram.error.BadRequest:
                # answer has invalid characters, so we send it without parse_mode
                await context.bot.send_message(update.effective_chat.id, message_chunk)
    except:
        await context.bot.send_message(update.effective_chat.id, "Какая-то ошибка в обработчике ошибок")

async def post_init(application: Application):
    await application.bot.set_my_commands([
        BotCommand("/new", "Начать новый диалог"),
        BotCommand("/mode", "Выбрать режим работы бота"),
        BotCommand("/retry", "Регенерировать предидущий запрос"),
        BotCommand("/premium", "Подписка на премиум"),
        BotCommand("/balance", "Показать баланс"),
        BotCommand("/settings", "Показать настройки"),
        BotCommand("/help", "Помощь"),
    ])

def run_bot() -> None:
    application = (
        ApplicationBuilder()
        .token(config.telegram_token)
        .concurrent_updates(True)
        .rate_limiter(AIORateLimiter(max_retries=5))
        .post_init(post_init)
        .build()
    )

    # add handlers
    user_filter = filters.ALL
    if len(config.allowed_telegram_usernames) > 0:
        usernames = [x for x in config.allowed_telegram_usernames if isinstance(x, str)]
        user_ids = [x for x in config.allowed_telegram_usernames if isinstance(x, int)]
        user_filter = filters.User(username=usernames) | filters.User(user_id=user_ids)

    application.add_handler(CommandHandler("start", start_handle, filters=user_filter))
    application.add_handler(CommandHandler("help", help_handle, filters=user_filter))

    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND & user_filter, message_handle))
    application.add_handler(CommandHandler("retry", retry_handle, filters=user_filter))
    application.add_handler(CommandHandler("new", new_dialog_handle, filters=user_filter))

    application.add_handler(MessageHandler(filters.VOICE & user_filter, voice_message_handle))
    
    application.add_handler(CommandHandler("mode", show_chat_modes_handle, filters=user_filter))
    application.add_handler(CallbackQueryHandler(set_chat_mode_handle, pattern="^set_chat_mode"))

    application.add_handler(CommandHandler("settings", settings_handle, filters=user_filter))
    application.add_handler(CallbackQueryHandler(set_settings_handle, pattern="^set_settings"))

    application.add_handler(CommandHandler("balance", show_balance_handle, filters=user_filter))
    
    application.add_handler(CommandHandler("premium", show_premium_handle, filters=user_filter))
    application.add_error_handler(error_handle)
    
    # start the bot
    application.run_polling()


if __name__ == "__main__":
    run_bot()