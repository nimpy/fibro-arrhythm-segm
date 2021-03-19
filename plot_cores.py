import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir_arrhythm', default='/scratch/fibro_arrhythm_data/OriginalTextures/OriginalArrhythmogenic',
                    help="Directory containing the arrhythmogenic textures")


if __name__ == '__main__':
    args = parser.parse_args()

    base_dir = args.data_dir_arrhythm
    texture_dirs = os.listdir(base_dir)

    for texture_dir in texture_dirs:

        with open(os.path.join(base_dir, texture_dir, "cores.pickle"), 'rb') as fp:
            cores = pickle.load(fp)
            print("\n=========== TEXTURE " + texture_dir + " (#cores: " + str(len(cores)) + ") ===========")
            for core in cores:
                assert len(core) == 2, "There is more than 2 coordinates?"
                print(core[0])
                print(core[1])
                print()

            texture = np.load(os.path.join(base_dir, texture_dir, "texture.npy"))

            # dict of occurrence count of items, see https://stackoverflow.com/a/28663910/4031135
            texture_elem_count = dict(zip(*np.unique(texture, return_counts=True)))
            assert texture_elem_count[0.0] == 1020, "There are zeros not only on the border?"
            fibrotic_perc = texture_elem_count[2.0] / (texture_elem_count[1.0] + texture_elem_count[2.0])
            print("fibrotic: ",  fibrotic_perc)

            # showing the texture with the cores
            plt.imshow(texture[1:-1, 1:-1])  # TODO (i.e. to ask) why crop? why is there a border of zeros?
            for core in cores:
                plt.plot(core[0], core[1], "ro")
                plt.plot(np.mean(core[0]), np.mean(core[1]), "wo")
            plt.show()

