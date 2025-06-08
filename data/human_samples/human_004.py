def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

def fibonacci_iterative(n):
    if n <= 1:
        return n
    
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b

# Compare both methods
n = 10
print(f"Fibonacci({n}) recursive: {fibonacci(n)}")
print(f"Fibonacci({n}) iterative: {fibonacci_iterative(n)}")