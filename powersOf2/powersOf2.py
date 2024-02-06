def largestPower(x):
    power = 0
    while (x > 1):
        if x%2 != 0:
            x -= 1
        x = x/2
        power += 1
    return power

def numDigits(x):
    return len(str(x))

x = 17
print(numDigits(x))
