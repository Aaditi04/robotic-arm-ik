import numpy as np
import matplotlib.pyplot as plt

# Hardcoded example data from your runs
# Analytical trajectory (from robotic_arm output)
analytical_t1 = [0,0.004,0.016,0.036,0.061,0.092,0.127,0.166,0.207,0.250,0.294,0.338,0.381,0.422,0.461,0.496,0.527,0.552,0.572,0.584,0.588]
analytical_t2 = [0,-0.00004,-0.00015,-0.00033,-0.00057,-0.00086,-0.00119,-0.00155,-0.00193,-0.00234,-0.00275,-0.00316,-0.00356,-0.00395,-0.00431,-0.00464,-0.00493,-0.00516,-0.00534,-0.00546,-0.00549]
analytical_t3 = [0,0.005,0.021,0.045,0.078,0.117,0.161,0.211,0.263,0.318,0.374,0.430,0.484,0.537,0.586,0.631,0.670,0.702,0.727,0.742,0.748]

# ML output (single-shot prediction)
ml_output = [0.57, 0.018, 0.69]

steps = range(len(analytical_t1))

plt.figure(figsize=(10,6))

plt.plot(steps, analytical_t1, label="Analytical θ1")
plt.plot(steps, analytical_t2, label="Analytical θ2")
plt.plot(steps, analytical_t3, label="Analytical θ3")

plt.axhline(ml_output[0], linestyle="--", label="ML θ1")
plt.axhline(ml_output[1], linestyle="--", label="ML θ2")
plt.axhline(ml_output[2], linestyle="--", label="ML θ3")

plt.xlabel("Trajectory Step")
plt.ylabel("Joint Angle (radians)")
plt.title("Analytical vs ML Inverse Kinematics")
plt.legend()
plt.grid()

plt.savefig("ik_comparison.png")
plt.show()

print("Plot saved as ik_comparison.png")
