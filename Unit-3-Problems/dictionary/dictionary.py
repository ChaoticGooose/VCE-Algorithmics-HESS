# The dictionary “friends_heights” contains my friends and their heights.

friends_heights = {"Allan": 158, "Barbara": 154, "Chris": 176, "Devlin": 190, "Lilette": 154, "Kayshaan": 160, "Cormac": 159}

# Loop through the dictionary and print “### is ### cm tall.” for each friend
for i in friends_heights:
    print(f"{i} is {friends_heights[i]} cm tall.")



#	Now create a dictionary containing at least five names as keys, and a tuple containing their age and height as the values.
height_age = {"alice": (13, 122), "bob": (14, 134), "charlie": (15, 145), "david": (16, 156), "edward": (1, 120)}


#	Loop through this dictionary and print “### is ### years old and ### cm tall.”
for i in height_age:
    print(f"{i} is {height_age[i][0]} years old and {height_age[i][1]} cm tall.")



#	Find the minimum age and maximum height. Print these along with the name of the relevant person.
min_height = min(height_age, key=lambda k: height_age[k][1])
max_height = max(height_age, key=lambda k: height_age[k][1])

print(f"Max Height: {max_height}")
print(f"Min Height: {min_height}")

#	Add a new person to your dictionary and repeat the last two questions.
height_age["fred"] = (13, 300)

for i in height_age:
    print(f"{i} is {height_age[i][0]} years old and {height_age[i][1]} cm tall.")

min_height = min(height_age, key=lambda k: height_age[k][1])
max_height = max(height_age, key=lambda k: height_age[k][1])

print(f"Max Height: {max_height}")
print(f"Min Height: {min_height}")