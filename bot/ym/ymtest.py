from yoomoney import Client
token = "4100118162676757.3AE1F4D6A8D90ADD92C6E60CA1987E82617335750DEFA82106B021CC2992D169C9C33FC4407BEFEBAB429B7F7BC1D48ECF8DF809B4C7347FD7E243D87DECC003E5A2626231D7849A3FB8E31E237FC24F9D89B96F370D23EF1636173FDDF5454DCD8238C7FF7D6145556F788E82A8908E6C3F2712BE7882AFDAE4011291982469"
client = Client(token)
user = client.account_info()
print("Account number:", user.account)
print("Account balance:", user.balance)
print("Account currency code in ISO 4217 format:", user.currency)
print("Account status:", user.account_status)
print("Account type:", user.account_type)
print("Extended balance information:")
for pair in vars(user.balance_details):
    print("\t-->", pair, ":", vars(user.balance_details).get(pair))
print("Information about linked bank cards:")
cards = user.cards_linked
if len(cards) != 0:
    for card in cards:
        print(card.pan_fragment, " - ", card.type)
else:
    print("No card is linked to the account")