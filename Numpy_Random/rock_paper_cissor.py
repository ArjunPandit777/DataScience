import numpy as np

choiceArray = ["Rock", "Paper", "Scissor"]

print("1. Rock\n2. Paper\n3. Scissor")
player_choice = int(input("Enter your choice: "))  # 2

print("Your choice: " + choiceArray[player_choice-1])

computer_choice = np.random.choice(choiceArray)
print("Computer choice: " + computer_choice)  # "Paper"
computer_choice_int = choiceArray.index(computer_choice)


computer_choice_int = computer_choice_int+1

if player_choice == computer_choice_int:
    print("Draw")
elif player_choice == 1 and computer_choice_int == 2:
    print("Computer wins")
elif player_choice == 2 and computer_choice_int == 3:
    print("Computer wins")
elif player_choice == 3 and computer_choice_int == 1:
    print("Computer wins")
elif player_choice == 1 and computer_choice_int == 3:
    print("You win")
elif player_choice == 2 and computer_choice_int == 1:
    print("You win")
elif player_choice == 3 and computer_choice_int == 2:
    print("You win")
else :
    print("Invalid input")