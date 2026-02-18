import random
import csv
import math

L1 = 0.3
L2 = 0.25
L3 = 0.15

def solve_ik(x, y, z):
    t1 = math.atan2(y, x)
    r = math.sqrt(x*x + y*y)
    zp = z - L1
    d = math.sqrt(r*r + zp*zp)

    if d > (L2 + L3) or d < abs(L2 - L3):
        return None

    cos_t3 = (d*d - L2*L2 - L3*L3)/(2*L2*L3)
    t3 = math.acos(cos_t3)

    a = math.atan2(zp, r)
    b = math.atan2(L3*math.sin(t3), L2 + L3*math.cos(t3))
    t2 = a - b

    return t1, t2, t3

with open("dataset.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["x","y","z","t1","t2","t3"])

    count = 0
    while count < 50000:
        x = random.uniform(0.1, 0.6)
        y = random.uniform(-0.6, 0.6)
        z = random.uniform(0.1, 0.6)

        sol = solve_ik(x,y,z)
        if sol:
            writer.writerow([x,y,z,*sol])
            count += 1

print("Dataset generated!")
