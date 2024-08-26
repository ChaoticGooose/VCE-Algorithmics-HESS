import random
from math import sqrt

# This code creates two random numbers between 1 and 50 << Ths comment is redundent the code is self commentimg
first = random.randint(1,50)
second = random.randint(1,50)


# 1. Print the two numbers with the sentence "My random numbers are:" << so is this
print(f"My random numbers are: {first} and {second}")

# 2. Store the numbers in variables 'larger' and 'smaller'. State the larger number. << and this
# (if they are the same then larger = smaller.) << potentially this
larger = max(first, second)
smaller = min(first, second)

# 3. State which of your numbers are even. << and this
if first % 2 == 0:
    print(f"{first} is even")
if second % 2 == 0:
    print(f"{second} is even")

# 4. State whether or not the larger number is exactly divisible by the smaller << this too
#    number. If not, state the remainder. << and this
if larger % smaller == 0:
    print(f"{larger} is divisible by {smaller}")
else:
    print(f"{larger} is not divisible by {smaller} and the remainder is {larger % smaller}")

# 5. If the two sides are the hypotenuse and shorter side of a right angled triangle, << could be less verbose but actually makes sense
#    find the length of the other shorter side.
side = sqrt((larger ** 2) - (smaller ** 2))
print(f"The length of the other shorter side is {'%.02f' % side}")