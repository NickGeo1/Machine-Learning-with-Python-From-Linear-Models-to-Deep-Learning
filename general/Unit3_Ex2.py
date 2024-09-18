import numpy as np
import matplotlib.pyplot as plt

W = np.array([[1, 0, -1],[0, 1, -1],[-1, 0, -1],[0, -1, -1]])

#Ax + By + C = 0
#y = -(A/B)x - C/B

x = np.linspace(-5, 5, 200)
for line in W:
    plt.plot(x, -line[0]/line[1]*x - line[2]/line[1])