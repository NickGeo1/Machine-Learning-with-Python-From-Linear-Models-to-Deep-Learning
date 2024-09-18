import numpy as np


def main():
    points = np.array([[-1, 0], [0,1]])
    lbls = np.array([1, 1])
    mistakes = np.zeros(shape=2)

    theta = np.array([0,0])
    #theta_0 = 0

    while True:
        for point,lbl in zip(points, lbls):
            if lbl*(np.dot(point, theta)) <= 0:
                mistakes[np.where(points == point)[0][1]] += 1
                theta = theta + lbl*point
                #theta_0 = theta_0 + lbl



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

