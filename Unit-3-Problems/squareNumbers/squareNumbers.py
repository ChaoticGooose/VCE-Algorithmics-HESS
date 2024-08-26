from math import sqrt

def devider(x):
    devides = 0
    while x > 1:
        x = x/2
        devides += 1
    return devides

def isPrime(x):
    for i in range(2, int(sqrt(x)+1)):
        if x%i == 0:
            print(i)
            return False
    return True

#  Print the first 30 square numbers
start = 1
end = 30
print(*[number**2 for number in range(start,end+1)])

#  Take an input number and divide it by 2 until the answer is less than 1. 
#  Print the number of divisions.
print(devider(int(input("Number: "))))


#  Take an input number and check to see if it is prime, by considering its remainder on 
#  division by every positive whole number less than itself but greater than 1.
print(isPrime(int(input("Number: "))))