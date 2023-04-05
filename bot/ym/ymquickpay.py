from yoomoney import Quickpay
def ympay1m(_label: str):
    _labelstr =str(_label)
    quickpay = Quickpay(
            receiver="4100118162676757",
            quickpay_form="shop",
            targets="Podpiska1m",
            paymentType="SB",
            sum=10,
            label= _labelstr
            )
    return quickpay.base_url
    #print(quickpay.base_url)
    #print(quickpay.redirected_url)