import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, p):
        """
        Generates a 72-dimensional particle vector given a latent representation
        vector
        """
        super(Generator, self).__init__()
        self.p = p

        self.h0 = nn.Sequential(
            nn.Linear(input_size, int(hidden_size/4)),
            nn.LeakyReLU(self.p)
        )
        self.h1 = nn.Sequential(
            nn.Linear(int(hidden_size/4), int(hidden_size)),
            nn.LeakyReLU(self.p)
        )
        self.h2 = nn.Sequential(
            nn.Linear(int(hidden_size), int(hidden_size/2)),
            nn.LeakyReLU(self.p)
        )
        self.h3 = nn.Sequential(
            nn.Linear(int(hidden_size/2), hidden_size)
        )
        self.out = nn.Sequential(
            nn.Linear(hidden_size, output_size)
        )
        
    def forward(self, x):
        x = self.h0(x)
        x = self.h1(x)
        x = self.h2(x)
        x = self.h3(x)
        x = self.out(x)
        return x
