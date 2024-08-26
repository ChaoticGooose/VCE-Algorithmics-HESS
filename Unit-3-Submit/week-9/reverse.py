def reverse(text):
    if len(text) <= 1: # Base case for last character
        return text
    return reverse(text[1:]) + text[0] # Append the first character to the end of the reversed string and recurse down to the last character

print(reverse("hello"))
