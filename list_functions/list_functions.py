from statistics import mean
my_numbers = [3,6,12,4,65,23,54,76,3,8,5,32]

# Find the maximum number in this list.
def find_max(my_numbers):
    return max(my_numbers)


# Find the mean average of the numbers in this list.
def find_mean(my_numbers):
    return mean(my_numbers)



# Use a while loop to remove numbers from the end of this list until only two are left.
def reduce_until_two(my_numbers):
    while len(my_numbers) > 2:
        my_numbers.pop()
    return my_numbers



# Run the following code to test your functions.

max = find_max(my_numbers)
if max == 76:
        print("You correctly found the maximum of 76.")
elif max is not None:
    print(f"Your maximum function returned {max} instead of 76.")

mean = find_mean(my_numbers)
if mean == 24.25:
    print("You correctly found the mean of 24.25.")
elif mean is not None:
    print(f"Your mean function returned {mean} instead of 24.25.")

last = reduce_until_two(my_numbers)
if last is not None and len(last) == 2 and last[0] == 3 and last[1] == 6:
    print("You correctly reduced the list.")
elif last is not None:
    print(f"Your list function returned {last} instead of [3,6]")
