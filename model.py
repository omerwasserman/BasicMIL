import torch
import torch.nn as nn
import torch.nn.functional as F


class Flatten(nn.Module):
    """
    This class flattens an array to a vector
    """
    def forward(self, x):
        N, C, H, W = x.size()  # read in N, C, H, W
        return x.view(N, -1)  # "flatten" the C * H * W values into a single vector per image


class GatedAttention(nn.Module):
    def __init__(self):
        super(GatedAttention, self).__init__()
        self.M = 500
        self.L = 128
        self.K = 1    # in the paper referred a 1.

        self.feature_extractor_basic_2 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # This layer don't change the size of input tiles.
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # This layer don't change the size of input tiles.
            nn.ReLU(),
            nn.BatchNorm2d(num_features=64),
            nn.Dropout(0.25),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # This layer don't change the size of input tiles.
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),  # This layer don't change the size of input tiles.
            nn.ReLU(),
            nn.BatchNorm2d(num_features=128),
            nn.Dropout(0.25),
            Flatten(),  # flattening from 7 X 7 X 64
            nn.Linear(7 * 7 * 128, self.M),
            nn.ReLU()
        )

        self.feature_extractor_basic_1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # This layer don't change the size of input tiles.
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # This layer don't change the size of input tiles.
            nn.ReLU(),
            nn.BatchNorm2d(num_features=64),
            nn.Dropout(0.25),
            Flatten(),  # flattening from 7 X 7 X 64
            nn.Linear(7 * 7 * 64, self.M),
            nn.ReLU()
        )

        self.attention_V = nn.Sequential(
            nn.Linear(self.M, self.L),
            nn.Tanh()
        )

        self.attention_U = nn.Sequential(
            nn.Linear(self.M, self.L),
            nn.Sigmoid()
        )

        self.attention_weights = nn.Linear(self.L, self.K)

        self.classifier = nn.Sequential(
            nn.Linear(self.M * self.K, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.squeeze(0)

        H = self.feature_extractor_basic_2(x)  # NxM
        """H = H.view(-1, 50 * 4 * 4) 
        H = self.feature_extractor_part2(H)  # NxL """

        A_V = self.attention_V(H)  # NxL
        A_U = self.attention_U(H)  # NxL
        A = self.attention_weights(A_V * A_U)  # element wise multiplication # NxK
        A = torch.transpose(A, 1, 0)  # KxN
        A = F.softmax(A, dim=1)  # softmax over N

        M = torch.mm(A, H)  # KxM

        # Because this is a binary classifier, the output of it is one single number which can be interpreted as the
        # probability that the input belong to class 1/TRUE (and not 0/FALSE)
        Y_prob = self.classifier(M)

        # The following line just turns probability to class.
        Y_class = torch.ge(Y_prob, 0.5).float()
        #Y_class = torch.tensor(Y_prob.data[0][0] < Y_prob.data[0][1]).float()

        return Y_prob, Y_class, A

    # AUXILIARY METHODS
    def calculate_classification_error(self, X, Y):
        Y = Y.float()
        _, Y_hat, _ = self.forward(X)
        error = 1. - Y_hat.eq(Y).cpu().float().mean().item()

        return error, Y_hat

    def calculate_objective(self, X, Y):
        Y = Y.float()
        Y_prob, _, A = self.forward(X)
        Y_prob = torch.clamp(Y_prob, min=1e-5, max=1. - 1e-5)
        neg_log_likelihood = -1. * (Y * torch.log(Y_prob) + (1. - Y) * torch.log(1. - Y_prob))  # negative log bernoulli

        return neg_log_likelihood, A

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


