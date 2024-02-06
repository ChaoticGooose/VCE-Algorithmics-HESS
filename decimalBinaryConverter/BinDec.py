"""
This can be done using the built in functions:
bin(x)
int(x, 2)
"""

def bin(x): # Devision method is a little faster
    binary = ""
    while x > 0:
        remander = x%2
        binary += str(remander)
        x = x//2
    return binary

def dec(x):
    decimal = 0
    for i, n in enumerate(x):
        # Multiply digit by 2^digit number
        decimal += int(n)*(2**int(i))
    return decimal

print(bin(9))
print(dec("1001"))