# ravi2.py - Interchange First and Last Characters of a String

s = input("Enter a string: ")
print("Result:", s[-1] + s[1:-1] + s[0] if len(s) > 1 else s)
