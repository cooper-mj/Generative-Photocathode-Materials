import numpy as np
import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, p, dropout):
        """
        Initializes three four linear layers to transform a 72-dimensional
        representation of a particle into a binary classification
        """
        super(Discriminator, self).__init__()
        self.p = p
        self.dropout = dropout

        # Transform 72-dimensional particle to 32-dimensional particle
        self.h0 = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LeakyReLU(self.p),
            nn.Dropout(self.dropout)
        )
        # Transform 32-dimensional particle to 16-dimensional particle
        self.h1 = nn.Sequential(
            nn.Linear(hidden_size, int(hidden_size/2)),
            nn.LeakyReLU(self.p),
            nn.Dropout(self.dropout)
        )
        # Transform 16-dimensional particle to 8-dimensional particle
        self.h2 = nn.Sequential(
            nn.Linear(int(hidden_size/2), int(hidden_size/4)),
            nn.LeakyReLU(self.p),
            nn.Dropout(self.dropout)
        )
        # Sigmoid activation for binary prediction
        self.out = nn.Sequential(
            torch.nn.Linear(int(hidden_size/4), output_size),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        x = self.h0(x)
        x = self.h1(x)
        x = self.h2(x)
        x = self.out(x)
        return x
