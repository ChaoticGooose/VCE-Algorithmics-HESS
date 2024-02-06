# Write a program which takes the user’s name and age.
# If their name is Bob, print “Hey Bob!” otherwise print “Hello.”
name = input("What is your name? ")
age = int(input("What is your age? "))
if name == "Bob":
    print("Hey Bob!")
else:
    print("Hello.")

# Using if/elif statements:
# If they are younger than 13, print “You are too young for Facebook.”
# If they are between 13 and 17, print “You can use Facebook but tell your parents.”
# If they are 18 or older, print “You can use Facebook as much as you like!”
if age < 13:
    print("You are too young for Facebook.")
elif age < 18:
    print("You can use Facebook but tell your parents.")
else:
    print("You can use Facebook as much as you like!")

# In addition to the above, if they are over 100, print “Are you sure you still want to use Facebook?”
if age > 100:
    print("Are you sure you still want to use Facebook?")