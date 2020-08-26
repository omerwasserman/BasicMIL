import utils
from torchvision import transforms
# Get data:
data = utils.get_data()

train_image_tiles = data['train_images']
train_labels = data['train_labels']


# transformations to data
# TODO: check about normalization (transformation)
trans = transforms.Compose([transforms.ToTensor()])
train_dset = utils.MnistMILdataset(train_image_tiles, train_labels, trans)

train_dset[5]

print('a')