from yoomoney import Client
token = "4100118162676757.3AE1F4D6A8D90ADD92C6E60CA1987E82617335750DEFA82106B021CC2992D169C9C33FC4407BEFEBAB429B7F7BC1D48ECF8DF809B4C7347FD7E243D87DECC003E5A2626231D7849A3FB8E31E237FC24F9D89B96F370D23EF1636173FDDF5454DCD8238C7FF7D6145556F788E82A8908E6C3F2712BE7882AFDAE4011291982469"
client = Client(token)
history = client.operation_history(label="a1b2c3d4e5")
print("List of operations:")
print("Next page starts with: ", history.next_record)
for operation in history.operations:
    print()
    print("Operation:",operation.operation_id)
    print("\tStatus     -->", operation.status)
    print("\tDatetime   -->", operation.datetime)
    print("\tTitle      -->", operation.title)
    print("\tPattern id -->", operation.pattern_id)
    print("\tDirection  -->", operation.direction)
    print("\tAmount     -->", operation.amount)
    print("\tLabel      -->", operation.label)
    print("\tType       -->", operation.type)