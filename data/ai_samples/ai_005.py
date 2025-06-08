def validate_email(email: str) -> bool:
    """
    Validate an email address using basic pattern matching.
    
    Args:
        email (str): Email address to validate
        
    Returns:
        bool: True if email is valid, False otherwise
    """
    import re
    
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))

def test_email_validation():
    """Test the email validation function."""
    test_emails = [
        "user@example.com",
        "invalid.email",
        "test@domain.co.uk",
        "not_an_email",
        "another@test.org"
    ]
    
    for email in test_emails:
        is_valid = validate_email(email)
        status = "Valid" if is_valid else "Invalid"
        print(f"{email}: {status}")

if __name__ == "__main__":
    test_email_validation()