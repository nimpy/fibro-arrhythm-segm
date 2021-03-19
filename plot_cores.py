import os
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt

path = sys.argv[1]

with open(os.path.join(path, "cores.pickle"), 'rb') as fp:
    cores = pickle.load(fp)
    plt.imshow(np.load(os.path.join(path, "texture.npy"))[1:-1, 1:-1])
    for core in cores:
        plt.plot(core[0], core[1], "r.")
        plt.plot(np.mean(core[0]), np.mean(core[1]), "w.")
    plt.show()
