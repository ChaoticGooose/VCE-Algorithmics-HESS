def factorial(m: int) -> int:
    if m == 0: # Base case
        return 1 # 0! = 1
    return m * factorial(m - 1) # Recurse down to the largest known value (0) then use results to calculate the factorial of m

print(factorial(10))
