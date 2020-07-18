todays_rice = [1]
total_rice = [1]
day = 100

for i in range(1, day):
    new_rice = todays_rice[-1] * 2
    todays_rice.append(new_rice)


print(todays_rice)

r1 = todays_rice[0]

for i in range(1, day):
    r1 += todays_rice[i]
    total_rice.append(r1)
    
print(total_rice)


# print(total_rice)


