import os
import pickle
import numpy as np
import imageio
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir_arrhythm', default='/scratch/fibro_arrhythm_data/OriginalTextures/OriginalArrhythmogenic',
                    help="Directory containing the arrhythmogenic textures")

parser.add_argument('--goal_data_dir', default='/scratch/fibro_arrhythm_data/ds210320',
                    help="Directory where the dataset ready for training will be stored")

parser.add_argument('--zfill_param', default=4,
                    help="Parameter for zfill, should be >= than the order of magnitude of how many textures there are")


def gaus(x, m, s):
    return 1 / (np.sqrt(2 * np.pi * s ** 2)) * np.exp(-(x - m) ** 2 / (2 * s ** 2))


def gaus_2D(x, y, s):
    xx, yy = np.meshgrid(np.arange(254), np.arange(254))
    return gaus(xx, x, s) * gaus(yy, y, s)


if __name__ == '__main__':
    args = parser.parse_args()

    base_dir = args.data_dir_arrhythm
    texture_dirs = os.listdir(base_dir)

    for texture_dir in texture_dirs:

        # texture
        texture = np.load(os.path.join(base_dir, texture_dir, "texture.npy"))
        texture = texture[1:-1, 1:-1]
        texture = texture - 1
        assert np.min(texture) - 0.0 < 0.00001, "The textures have some unexpected values"
        assert np.max(texture) - 1.0 < 0.00001, "The textures have some unexpected values"

        texture_ID = texture_dir[3:]

        print(texture_ID.zfill(args.zfill_param))
        print(texture[:,:6])

        texture_filename = 'texture_' + texture_ID.zfill(args.zfill_param) + '.npy'
        print(os.path.join(args.goal_data_dir, 'textures', texture_filename))

        # with open(os.path.join(args.goal_data_dir, 'textures', texture_filename), 'wb') as f:
        #     np.save(f, texture)


        # cores
        with open(os.path.join(base_dir, texture_dir, "cores.pickle"), 'rb') as fp:
            cores = pickle.load(fp)

            plt.imshow(texture[1:-1, 1:-1])  # TODO (i.e. to ask) why crop? why is there a border of zeros?
            for core in cores:
                plt.plot(core[0], core[1], "ro")
                plt.plot(np.mean(core[0]), np.mean(core[1]), "wo")
            # plt.show()


            labels = np.zeros_like(texture)

            for core in cores:
                assert len(core) == 2, "There is more than 2 coordinates?"
                print(core[0])
                print(core[1])

                core_center_x = np.mean(core[0])
                core_center_y = np.mean(core[1])
                core_center = np.array(core_center_x, core_center_y)

                average_dist_from_core = 0
                for i in range(len(core[0])):
                    average_dist_from_core += np.linalg.norm(np.array(core[0][i], core[1][i]) - core_center)

                average_dist_from_core /= len(core[0])
                print(average_dist_from_core)
                core_radius = 1.5 * average_dist_from_core  # TODO use this for gaus_2D function instead


                gaussian_core = gaus_2D(core_center_x, core_center_y, average_dist_from_core)
                # plt.imshow(gaussian_core)
                # plt.show()

                # TODO make the "Gaussian" balls have the same magniture,
                # TODO threshold everything outside of e.g. 2 x radius to zero

                labels += gaussian_core

            plt.imshow(labels)
            plt.show()



