import time
import numpy as np
from scipy.spatial import KDTree
import matplotlib.pyplot as plt

#basic implementation
if __name__ == '__main__':
    drone_list = [[2, 1], [5, 5], [4, 8]]
    obstacle_list = [[1, 5], [7, 1], [8, 7]]

    #For clarity, visualizing drone and obstacle locations
    for drone in drone_list:
        plt.scatter(*drone, marker='^', color='r')
    for obstacle in obstacle_list:
        plt.scatter(*obstacle, marker='o', color='k')
    plt.grid()
    plt.draw()

    tree = KDTree(obstacle_list, balanced_tree=True, leafsize=1) #leafsize can be changed depending on # of data pts

    k = 1
    dist, index = tree.query(drone_list, k=k) #parameter eps can perform approximate NN, distance_upper_bound for pruning
    for i, drone in enumerate(drone_list):
        print('Closest obstacle to drone ', drone, ' is obstacle ', obstacle_list[index[i]], ', distance = ', dist[i])

    k = 2
    dist, index = tree.query(drone_list, k=k)  # kNN test
    print(index)
    for i, drone in enumerate(drone_list):
        print('Closest 2 obstacles to drone ', drone, ' are obstacles at ')
        for ind in range(k):
            print(obstacle_list[index[i][ind]])

    plt.show()


