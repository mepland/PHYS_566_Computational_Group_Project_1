import random

x = 0
y = 0
for i in range(5):
    r = random.seed()
    r=random.random()
    print r
    if r <= 0.25:
        print 1
    elif r <= 0.5:
        print 2
    elif r <= 0.75:
        print 3
    else:
        print 4