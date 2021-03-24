import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--data_dir_arrhythm', default='/scratch/fibro_arrhythm_data/OriginalTextures/OriginalArrhythmogenic',
                    help="Directory containing the arrhythmogenic textures")
parser.add_argument('--data_dir_nonarrhythm', default='/scratch/fibro_arrhythm_data/OriginalTextures/OriginalNonArrhythmogenic',
                    help="Directory containing the non-arrhythmogenic textures")

parser.add_argument('--goal_data_dir', default='/scratch/fibro_arrhythm_data/ds210324',
                    help="Directory where the dataset ready for training will be stored")

parser.add_argument('--zfill_param', default=4,
                    help="Parameter for zfill, should be >= than the order of magnitude of how many textures there are")


# gaus() and gaus_2D() are from https://stackoverflow.com/a/55382660/4031135
def gaus(x, m, s):
    return 1 / (np.sqrt(2 * np.pi * s ** 2)) * np.exp(-(x - m) ** 2 / (2 * s ** 2))


def gaus_2D(x, y, gaus_size, image_size):
    xx, yy = np.meshgrid(np.arange(image_size), np.arange(image_size))
    return gaus(xx, x, gaus_size) * gaus(yy, y, gaus_size)


def generate_dataset(base_dir, goal_dir, dataset_name, arrhythm, visualise, saving=False):
    assert not (visualise and saving), "This is a bad idea."

    texture_dirs = os.listdir(base_dir)

    for texture_dir in texture_dirs:

        # texture
        texture = np.load(os.path.join(base_dir, texture_dir, "texture.npy"))
        texture = texture[1:-1, 1:-1]
        texture = texture - 1
        assert np.min(texture) - 0.0 < 0.00001, "The textures have some unexpected values"
        assert np.max(texture) - 1.0 < 0.00001, "The textures have some unexpected values"

        texture_ID = texture_dir[3:]
        # texture_ID = texture_ID.zfill(args.zfill_param)

        print(dataset_name + '_' + texture_ID)
        
        # if texture_ID == '944_0':  # problematic
        #     print()

        texture_filename = 'texture_' + dataset_name + '_' + texture_ID + '.npy'
        # print(os.path.join(args.goal_data_dir, 'textures', texture_filename))

        label_image = np.zeros_like(texture)

        if arrhythm:

            # cores
            with open(os.path.join(base_dir, texture_dir, "cores.pickle"), 'rb') as fp:
                cores = pickle.load(fp)

                if visualise:
                    plt.imshow(texture[1:-1, 1:-1])  # TODO (i.e. to ask) why crop? why is there a border of zeros?
                    for core in cores:
                        plt.plot(core[0], core[1], "ro")
                        plt.plot(np.mean(core[0]), np.mean(core[1]), "bo")

                for core in cores:
                    assert len(core) == 2, "There is more than 2 coordinates?"

                    core_center_x = np.mean(core[0])
                    core_center_y = np.mean(core[1])
                    core_center = np.array(core_center_x, core_center_y)

                    distances_from_core = []
                    for i in range(len(core[0])):
                        distance_from_core = np.linalg.norm(np.array(core[0][i], core[1][i]) - core_center)
                        distances_from_core.append(distance_from_core)

                    max_distance_from_core = np.max(np.array(distances_from_core))

                    assert label_image.shape[0] == label_image.shape[1], \
                        "Generating non-square label images not (yet?) supported..."
                    gaussian_core = gaus_2D(core_center_x, core_center_y, max_distance_from_core, label_image.shape[0])

                    gaussian_core_normalised = (gaussian_core - gaussian_core.min()) / (
                                gaussian_core.max() - gaussian_core.min())

                    gaussian_core_normalised[gaussian_core_normalised < 0.01] = 0  # set to 1 to visualise the thresholding

                    label_image += gaussian_core_normalised

                label_image[label_image > 1.0] = 1  # since these gaussian balls were additive, they need to be cut off at 1

                if visualise:
                    plt.imshow(label_image, cmap='gray')
                    plt.show()

        label_filename = 'label_' + dataset_name + '_' + texture_ID + '.npy'
        # print(os.path.join(args.goal_data_dir, 'labels', label_filename))

        assert np.min(label_image) > -0.00001, "The label images have some unexpected values"
        assert np.max(label_image) < 1.00001, "The label images have some unexpected values"

        # print(np.min(label_image), np.max(texture))
        # print(np.min(label_image), np.max(label_image))

        if saving:
            with open(os.path.join(goal_dir, 'textures', texture_filename), 'wb') as f:
                np.save(f, texture)
            with open(os.path.join(goal_dir, 'labels', label_filename), 'wb') as f:
                np.save(f, label_image)


if __name__ == '__main__':
    args = parser.parse_args()

    # TODO convert both texture and label image array types to float32 (now it's done when loading the dataset)

    saving = True

    generate_dataset('/scratch/fibro_arrhythm_data/OriginalTextures/OriginalArrhythmogenic', args.goal_data_dir, "orig", arrhythm=True, visualise=False, saving=saving)
    generate_dataset('/scratch/fibro_arrhythm_data/OriginalTextures/OriginalNonArrhythmogenic', args.goal_data_dir, "orig", arrhythm=False, visualise=False, saving=saving)

    generate_dataset('/scratch/fibro_arrhythm_data/ExtraData/Dataset_ap6_yes', args.goal_data_dir, "dsap6", arrhythm=True, visualise=False, saving=saving)
    generate_dataset('/scratch/fibro_arrhythm_data/ExtraData/Dataset_ap6_no', args.goal_data_dir, "dsap6", arrhythm=False, visualise=False, saving=saving)

    generate_dataset('/scratch/fibro_arrhythm_data/ExtraData/Dataset_ap24_yes', args.goal_data_dir, "dsap24", arrhythm=True, visualise=False, saving=saving)
    generate_dataset('/scratch/fibro_arrhythm_data/ExtraData/Dataset_ap24_no', args.goal_data_dir, "dsap24", arrhythm=False, visualise=False, saving=saving)

    generate_dataset('/scratch/fibro_arrhythm_data/ExtraData/Dataset_other/Dataset_ap6_31_yes', args.goal_data_dir, "dsoap631", arrhythm=True, visualise=False, saving=saving)
    generate_dataset('/scratch/fibro_arrhythm_data/ExtraData/Dataset_other/Dataset_ap6_31_no', args.goal_data_dir, "dsoap631", arrhythm=False, visualise=False, saving=saving)

    generate_dataset('/scratch/fibro_arrhythm_data/ExtraData/Dataset_other/Dataset_ap6_35_yes', args.goal_data_dir, "dsoap635", arrhythm=True, visualise=False, saving=saving)
    generate_dataset('/scratch/fibro_arrhythm_data/ExtraData/Dataset_other/Dataset_ap6_35_no', args.goal_data_dir, "dsoap635", arrhythm=False, visualise=False, saving=saving)

    # total number of data samples: 15637

