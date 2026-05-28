from monolith.trans_inf import generate_response

conversation = ""

print("TinyGPT Chat")
print("Type 'exit' to quit.\n")

while True:

    user_input = input("You: ")

    if user_input.lower() == "exit":
        break

    conversation += f"\nUser: {user_input}\nAssistant:"

    response = generate_response(
        conversation,
        max_new_tokens=150
    )

    print(f"\nBot:{response}\n")

    conversation += response