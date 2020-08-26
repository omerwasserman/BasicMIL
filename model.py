import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


def device_gpu_cpu():
    USE_GPU = True
    if USE_GPU and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    return device


class Flatten(nn.Module):
    """
    This class flattens an array to a vector
    """
    def forward(self, x):
        N, C, H, W = x.size()  # read in N, C, H, W
        return x.view(N, -1)  # "flatten" the C * H * W values into a single vector per image


class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.L = 500
        self.D = 128
        self.K = 1

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # This layer don't change the size of input tiles.
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),   # This layer don't change the size of input tiles.
            nn.ReLU(),
            nn.BatchNorm2d(num_features=64),
            nn.Dropout(0.25),
            Flatten(),  # flattening from 7 X 7 X 64
            nn.Linear(7 * 7 * 64, self.L),
            nn.ReLU()
        )

        self.feature_extractor_part1 = nn.Sequential(
            nn.Conv2d(1, 20, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(20, 50, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2)
        )

        self.feature_extractor_part2 = nn.Sequential(
            nn.Linear(50 * 4 * 4, self.L),
            nn.ReLU(),
        )

        self.attention = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh(),
            nn.Linear(self.D, self.K)
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.L*self.K, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.squeeze(0)

        H = self.feature_extractor_part1(x)
        H = H.view(-1, 50 * 4 * 4)
        H = self.feature_extractor_part2(H)  # NxL

        A = self.attention(H)  # NxK
        A = torch.transpose(A, 1, 0)  # KxN
        A = F.softmax(A, dim=1)  # softmax over N

        M = torch.mm(A, H)  # KxL

        Y_prob = self.classifier(M)
        Y_hat = torch.ge(Y_prob, 0.5).float()

        return Y_prob, Y_hat, A

def model_1():
    num_classes = 2

    model = nn.Sequential( nn.Conv2d(1, 32, kernel_size=3),     # This layer makes the images 26X26
                           nn.ReLU(),
                           nn.Conv2d(32, 64, kernel_size=3),    # This layer makes the images 24X24
                           nn.ReLU(),
                           nn.MaxPool2d(kernel_size=2),         # This layer makes the images 12X12
                           nn.Dropout(0.25),
                           Flatten(),                           # flattening from 12X12X64
                           nn.Linear(12 * 12 * 64, 128),
                           nn.ReLU(),
                           nn.Linear(128, num_classes) )

    return model


model = nn.Sequential( )