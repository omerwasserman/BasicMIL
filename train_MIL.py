import utils
import model
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

def train(model: nn.Module, data_loader: DataLoader):
    """
    This function trains the model
    :return:
    """
    print('Start Training...')
    model.train()
    best_train_error = data_loader.dataset.labels.shape[0]
    best_train_loss = 1e5
    best_model = None

    for e in range(epoch):
        train_error, train_loss = 0, 0
        for batch_idx, (data, target) in enumerate(data_loader):
            data, target = data.to(DEVICE), target.to(DEVICE)
            prob, label, weights = model(data)

            # Check training error
            error = 1. - label.eq(target).cpu().float().mean().item()
            train_error += error

            prob = torch.clamp(prob, min=1e-5, max=1. - 1e-5)
            neg_log_likelihood = -1. * (target * torch.log(prob) + (1. - target) * torch.log(1. - prob))  # negative log bernoulli
            loss = neg_log_likelihood
            train_loss += loss.data[0]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if train_error < best_train_loss:
            best_train_loss = train_loss
            best_model = model
        print('Epoch: {}, Loss: {:.2f}, Train error: {:.0f}'.format(e, train_loss.cpu().numpy()[0], train_error))

    return best_model, best_train_error, best_train_loss.cpu().numpy()[0]

def check_accuracy(model: nn.Module, data_loader: DataLoader):
    model.eval()
    for batch_idx, (data, target) in enumerate(data_loader):
        data, target = data.to(DEVICE), target.to(DEVICE)
        prob, label, weights = model(data_loader)


def test():
    """
    This function test the model
    :return:
    """
    model.eval()

    pass


##################################################################################################

# Device definition:
DEVICE = utils.device_gpu_cpu()

# Model upload:
# TODO: upload model
net = model.GatedAttention()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=1e-4, weight_decay=1e-4)
# optimizer = optim.Adadelta(net.parameters())



# Get data:
data = utils.get_data()
train_image_tiles = data['train_images']
train_labels = data['train_labels']

# transformations to data
# TODO: check about normalization (transformation)

trans = transforms.Compose([transforms.ToTensor()])
train_dset = utils.MnistMILdataset(train_image_tiles, train_labels, trans)

train_loader = DataLoader(train_dset, batch_size=1, shuffle=False)

epoch = 3
best_model, best_train_error, best_train_loss = train(net, train_loader)

print()
print('We have a model with {:.0f}/500 training error and {:.2f} training loss'.format(best_train_error, best_train_loss))

print('Done!')