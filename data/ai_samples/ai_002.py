class BankAccount:
    """
    A class to represent a bank account.
    
    Attributes:
        account_number (str): The account number
        balance (float): The current balance
        account_holder (str): The name of the account holder
    """
    
    def __init__(self, account_number: str, account_holder: str, initial_balance: float = 0.0):
        """
        Initialize a new bank account.
        
        Args:
            account_number (str): The account number
            account_holder (str): The name of the account holder
            initial_balance (float): The initial balance (default: 0.0)
        """
        self.account_number = account_number
        self.account_holder = account_holder
        self.balance = initial_balance
    
    def deposit(self, amount: float) -> bool:
        """
        Deposit money into the account.
        
        Args:
            amount (float): The amount to deposit
            
        Returns:
            bool: True if successful, False otherwise
        """
        if amount > 0:
            self.balance += amount
            return True
        return False
    
    def withdraw(self, amount: float) -> bool:
        """
        Withdraw money from the account.
        
        Args:
            amount (float): The amount to withdraw
            
        Returns:
            bool: True if successful, False otherwise
        """
        if amount > 0 and amount <= self.balance:
            self.balance -= amount
            return True
        return False
    
    def get_balance(self) -> float:
        """
        Get the current balance.
        
        Returns:
            float: The current balance
        """
        return self.balance