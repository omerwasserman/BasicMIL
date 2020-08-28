import utils
import model
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from typing import List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns


def train(model: nn.Module, dloader_train: DataLoader, dloader_test: DataLoader):
    """
    This function trains the model
    :return:
    """
    print('Start Training...')
    best_train_error = dloader_train.dataset.labels.shape[0]
    best_train_loss = 1e5
    best_model = None

    for e in range(epoch):
        train_error, train_loss = 0, 0
        model.train()
        for batch_idx, (data, target, _) in enumerate(dloader_train):
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
            best_train_error = train_error
            best_model = model

        model.eval()
        test_acc = check_accuracy(model, dloader_test)
        model.train()
        print('Epoch: {}, Loss: {:.2f}, Train error: {:.0f}'.format(e, train_loss.cpu().numpy()[0], train_error))

    return best_model, best_train_error, best_train_loss.cpu().numpy()[0]


def check_accuracy(model: nn.Module, data_loader: DataLoader):
    """
       if loader.dataset.train:
        print('Checking accuracy on validation set')
    else:
        print('Checking accuracy on test set')
    """
    num_correct = 0
    num_samples = 0
    model.eval()  # set model to evaluation mode
    with torch.no_grad():
        for data, target, _ in data_loader:
            data, target = data.to(device=DEVICE), target.to(device=DEVICE)
            _, label, _ = model(data)
            num_correct += (label == target).cpu().int().item()
            num_samples += label.size(0)
        acc = float(num_correct) / num_samples
        print('Got {} / {} correct {:.2f} %'.format(num_correct, num_samples, 100 * acc))
    return acc


def inference_print_weights(model: nn.Module, data_loader: DataLoader, grid: List[Tuple]):
    model.eval()
    with torch.no_grad():
        data, target, instance_locations = next(iter(data_loader))
        data, target = data.to(device=DEVICE), target.to(device=DEVICE)
        _, label, weights = model(data)
        num_correct = (label == target).item()
        weights = weights.cpu().numpy()
        instance_locations = instance_locations.cpu().numpy()

    print('Correct Labeling') if num_correct else print('Wrong Labeling')
    heat_image, image = utils.weights_2_image(data.squeeze(0).cpu().numpy(), weights[0], instance_locations[0], grid)

    plt.figure(0)
    plt.subplot(1, 2, 1)
    plt.imshow(image[0, :, :], cmap='gray')
    plt.subplot(1, 2, 2)
    plt.imshow(heat_image, cmap='hot', interpolation='nearest')
    plt.colorbar()  # TODO: THIS line is new . CHECK it!

    plt.figure(1)
    sns.heatmap(heat_image, linewidth=0.5)
    plt.show()





##################################################################################################


# Device definition:
DEVICE = utils.device_gpu_cpu()

# Get data:
data, grid = utils.get_data()
train_image_tiles = data['train_images']
train_labels = data['train_labels']
test_image_tiles = data['test_images']
test_labels = data['test_labels']

# transformations to data
# TODO: check about normalization (transformation)

trans = transforms.Compose([transforms.ToTensor()])
train_dset = utils.MnistMILdataset(train_image_tiles, train_labels, trans)
train_loader = DataLoader(train_dset, batch_size=1, shuffle=False)

test_dset = utils.MnistMILdataset(test_image_tiles, test_labels, trans)
test_loader = DataLoader(test_dset, batch_size=1, shuffle=False)

infer = False

# Model upload:
if not infer:
    net = model.GatedAttention()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=1e-4, weight_decay=1e-4)
    # optimizer = optim.Adadelta(net.parameters())

    epoch = 20
    best_model, best_train_error, best_train_loss = train(net, train_loader, test_loader)

    print()
    print('We have a model with {:.0f}/500 training error and {:.2f} training loss'.format(best_train_error, best_train_loss))

    # Checking test set accuracy:
    test_acc_final = check_accuracy(best_model, test_loader)
    print()
    print('The model gets test set accuracy of {:.2f} %'.format(test_acc_final * 100))

    # Save model to file:
    torch.save(best_model.state_dict(), 'models/model.pt')
    print(' Training is done!')
    print('model is saved')
else:
    # load model:
    print('Loading pre trained model')
    model = model.GatedAttention()
    model.load_state_dict(torch.load('models/model.pt'))
    model.eval()
    # Print image with weights:
    inference_print_weights(model, test_loader, grid)
