import numpy as np
from random import sample, seed
import torch
from torch.utils.data import Dataset
from typing import List, Tuple


DATA_FILE = 'Data/data_dictionary.npy'

def weights_2_image(data: np.ndarray, weights: np.ndarray, locations: np.ndarray, grid: List[Tuple]):
    """
    This function creates a heat map from the weights and their locations
    :param weights:
    :param locations:
    :return:
    """
    locations = locations.tolist()
    heat_image = np.zeros((28, 28))
    image = np.zeros((data.shape[1], 28, 28))
    tile_size = 7
    for idx, loc in enumerate(grid):
        try:
            array_loc = locations.index(idx)
            weight = weights[array_loc]
            heat_image[loc[0]: loc[0] + tile_size, loc[1]: loc[1] + tile_size] = weight
            image[:, loc[0]: loc[0] + tile_size, loc[1]: loc[1] + tile_size] = data[array_loc, :, :, :]
        except:
            image[:, loc[0]: loc[0] + tile_size, loc[1]: loc[1] + tile_size] = 255

    return heat_image, image


def device_gpu_cpu():
    USE_GPU = True
    if USE_GPU and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    return device


def get_data(train_samples: int = 250) -> dict:
    """
    This function prepares data for training and testing.
    The function return images of numbers 3 and 8 from MNIST dataset.

    :param train_samples: number of training samples from each class
    :return:

    """

    seed(0)

    data_dict = np.load(DATA_FILE, allow_pickle=True).item()

    picked_labels = (3, 8)

    all_images_0 = data_dict[picked_labels[0]]
    all_images_1 = data_dict[picked_labels[1]]

    # Create random indices from which we'll choose later the indices of the train set:
    idx_0 = sample(range(all_images_0.shape[0]), all_images_0.shape[0])
    idx_1 = sample(range(all_images_1.shape[0]), all_images_1.shape[0])

    # Picking train and test images:
    train_images_0 = all_images_0[idx_0[:train_samples]]
    train_images_1 = all_images_1[idx_1[:train_samples]]

    test_images_0 = all_images_0[idx_0[train_samples:]]
    test_images_1 = all_images_1[idx_1[train_samples:]]

    images_train = np.vstack((train_images_0, train_images_1))
    labels_train = np.hstack((np.zeros(train_samples, dtype=np.uint8), np.ones(train_samples, dtype=np.uint8)))

    images_test = np.vstack((test_images_0, test_images_1))
    labels_test_0 = np.zeros(test_images_0.shape[0], dtype=np.uint8)
    labels_test_1 = np.ones(test_images_1.shape[0], dtype=np.uint8)
    labels_test = np.hstack((labels_test_0, labels_test_1))

    # Let's mix up the train samples:
    idx = sample(range(images_train.shape[0]), images_train.shape[0])
    images_train = images_train[idx]
    labels_train = labels_train[idx]



    # Creating the grid:
    tile_size = 7
    grid = _get_tile_grid((28, 28), tile_size)
    # Creating an array of all tiles:
    all_train_tiles = _make_tiles_all(images_train, 16, grid, tile_size)
    all_test_tiles = _make_tiles_all(images_test, 16, grid, tile_size)

    # Putting all train and test data in Dictionary
    data = {'train_images': all_train_tiles,
            'train_labels': labels_train,
            'test_images': all_test_tiles,
            'test_labels': labels_test}
    return data, grid


def _make_bag(img_tiles: np.ndarray, num_tiles: tuple = (16, 16), label: np.ndarray = None):
    """
    This function create one bag from one image.
    the bag will consist a random number of tiles ranging between num_til[0] and num_til[1]
    :param img_tiles: all tiles of the images in one nd.array. data is of shape (Tiles, C, H, W)
    :param num_tiles: tuple with range for random number of number of tiles in each bag
    :param label: if not None than contains the true label of the data. label is of shape (N,)
    :return:
    """

    # Removing tiles containing only background:
    img_tiles, idxs = _remove_bakground_tiles(img_tiles)
    img_tile_num = img_tiles.shape[0]

    # if the number of tiles for the current image is less than the number of tiles we want inside the bag,
    # we'll need to update the number of tiles in the bag:
    delta = num_tiles[1] - img_tile_num
    num_tiles = (num_tiles[0] - delta, num_tiles[1] - delta)
    if num_tiles[0] < 0:
        num_tiles = (0, num_tiles[1] - delta)

    # Randomly creating indices of tiles to be inserted into bag:
    instances_num = np.random.randint(num_tiles[0], num_tiles[1] + 1)  # Number of instances to be in bag
    idx = sample(range(img_tile_num), instances_num)  # Create the indices of the instances to be inserted
    instance_location_in_bag = np.array(idxs)[idx]

    # We'll now put the tiles in a bag:
    bag = img_tiles[idx, :, :, :]

    return bag, instance_location_in_bag


def _remove_bakground_tiles(tiles: np.ndarray) -> np.ndarray:
    """
    This function gets an array of tiles of size (Tiles, C, H, W) and removes tiles with background only
    :param tiles: input tiles
    :return:
    """
    mean_val = tiles.mean(axis=(1, 2, 3))
    idx = np.where(mean_val != 255)[0].tolist()
    new_tiles = tiles[idx, :, :, :]
    return new_tiles, idx


def _make_tiles_all(data: np.ndarray, instance_num: int, grid: List[Tuple], tile_size: int) -> np.ndarray:
    N, C, _, _ = data.shape
    all_tiles = np.zeros((N, instance_num, C, tile_size, tile_size))
    for idx, loc in enumerate(grid):
        tile = data[:, :, loc[0]:loc[0] + tile_size, loc[1]:loc[1] + tile_size]
        all_tiles[:, idx, :, :, :] = tile
    return all_tiles


def _get_tile_grid(data_size: tuple, tile_size :int = 7) -> List[Tuple]:
    """
    This function creates a list of tuples containing the location of top left corner of tiles.
    :param data_size: tuple (cols, rows) dictating the size of input image
    :param tile_size: int dictating the size of tiles
    :return: a tuple containing the grid coordinates - (x, y)
    """

    grid = [(x, y) for x in range(0, data_size[0], tile_size) for y in range(0, data_size[1], tile_size)]
    return grid


class MnistMILdataset(Dataset):
    def __init__(self, all_image_tiles, labels, transform=None):
        self.all_image_tiles = all_image_tiles
        self.labels = torch.LongTensor(labels)
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        tiles = self.all_image_tiles[idx]
        x, instance_locations = _make_bag(tiles)
        y = self.labels[idx]

        shape = x.shape
        X = torch.zeros(shape)
        if self.transform:
            x = x.transpose(0, 2, 3, 1)

            for i in range(x.shape[0]):
                X[i] = self.transform(x[i])

        return X, y, instance_locations
