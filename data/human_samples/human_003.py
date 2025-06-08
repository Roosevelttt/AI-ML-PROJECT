import random

def guess_number_game():
    number = random.randint(1, 100)
    attempts = 0
    max_attempts = 7
    
    print("Welcome to the Number Guessing Game!")
    print(f"I'm thinking of a number between 1 and 100.")
    print(f"You have {max_attempts} attempts to guess it.")
    
    while attempts < max_attempts:
        try:
            guess = int(input("Enter your guess: "))
            attempts += 1
            
            if guess == number:
                print(f"Congratulations! You guessed it in {attempts} attempts!")
                return True
            elif guess < number:
                print("Too low!")
            else:
                print("Too high!")
                
            print(f"Attempts remaining: {max_attempts - attempts}")
            
        except ValueError:
            print("Please enter a valid number!")
    
    print(f"Game over! The number was {number}")
    return False

if __name__ == "__main__":
    guess_number_game()