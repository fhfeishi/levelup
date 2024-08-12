# .split  .rsplit()

s = "one_two_three_four"

sl = s.split('_')
print("left spilt '_':", sl)  # left spilt '_': ['one', 'two', 'three', 'four']
sr = s.rsplit('_')
print("right spilt '_':", sr)  # right spilt '_': ['one', 'two', 'three', 'four']

slx = s.split('_', 2)
print("left spilt '_' *2 :", slx)  # left spilt '_' *2 : ['one', 'two', 'three_four']

srx = s.rsplit('_', 1)
print("right spilt '_' *1:", srx)  # right spilt '_' *1: ['one_two_three', 'four']
