def calculator():
    while True:
        num1 = int(input("Enter the first number:"))
        num2 = int(input("Enter the second number:"))
        print(
            """Select one of the following operators:
            1. Addition
            2. Substraction
            3. Multiply
            4. Divide
            """
        )
        operation = input("Enter the operator number from the list above: ")
        if operation == "1":
            result = num1 + num2
            print(f"{num1} + {num2} is equal to: {result}")
        elif operation == "2":
            result = num1 - num2
            print(f"{num1} - {num2} is equal to: {result}")
        elif operation == "3":
            result = num1 * num2
            print(f"{num1} * {num2} is equal to: {result}")
        elif operation == "4":
            if num2 != 0:
                result = num1 / num2
                print(f"{num1} ÷ {num2} is equal to: {result}")
            else:
                print(
                    "You can never divide a number by 0. You should know this by now you moron "
                )

        exit = input("Do you wish to perfomrm another calculation? yes/no: ")
        if exit == "no":
            print("We are sad to see you go. Goodbye :)")
            break
        elif exit == "yes":
            continue
        else:
            print("Are you dumb or something? Write yes/no: ")
            #  Your code goes here


def introduction(name, age):
    print(f"My name is {name} and I am {age} years old")


variable = "jambo"
