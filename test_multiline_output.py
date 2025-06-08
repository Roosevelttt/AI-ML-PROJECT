# Test script to verify multi-line output display
print("Line 1: This is the first line of output")
print("Line 2: This is the second line")
print("Line 3: Here's a third line")
print()
print("Line 5: After an empty line")

# Test with loops
for i in range(3):
    print(f"Loop iteration {i + 1}")

print()
print("Multi-line string test:")
multiline_text = """This is a
multi-line string
that spans several
lines of text."""
print(multiline_text)

# Test with lists
numbers = [1, 2, 3, 4, 5]
print("\nPrinting list elements:")
for num in numbers:
    print(f"Number: {num}")

print("\nFinal line of output")
