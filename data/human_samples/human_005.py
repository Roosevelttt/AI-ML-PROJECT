def read_file_safely(filename):
    try:
        with open(filename, 'r') as file:
            content = file.read()
            return content
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        return None
    except PermissionError:
        print(f"Error: Permission denied to read '{filename}'.")
        return None
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None

# Example usage
content = read_file_safely("example.txt")
if content:
    print("File content:", content[:100])  # First 100 characters
else:
    print("Failed to read file.")