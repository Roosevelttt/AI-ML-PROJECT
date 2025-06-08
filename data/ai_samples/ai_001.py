def calculate_factorial(n):
    """
    Calculate the factorial of a given number.
    
    Args:
        n (int): The number to calculate factorial for
        
    Returns:
        int: The factorial of n
    """
    if n == 0 or n == 1:
        return 1
    else:
        return n * calculate_factorial(n - 1)

def main():
    """Main function to demonstrate factorial calculation."""
    number = 5
    result = calculate_factorial(number)
    print(f"The factorial of {number} is {result}")

if __name__ == "__main__":
    main()