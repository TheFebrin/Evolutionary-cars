import torch
import torch.nn as nn
from functools import reduce


class Network(nn.Module):
    def __init__(self, in_dim, h1, h2, out_dim):
        super(Network, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_dim, h1),
            nn.Tanh(),
            nn.Linear(h1, h2),
            nn.Tanh(),
            nn.Linear(h2, out_dim),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.layers(x)


'''
if __name__ == '__main__':
    model = Network(in_dim=3, h1=4, h2=4, out_dim=2)
    p = sum(reduce( lambda a, b: a*b, x.size()) for x in model.parameters())
    print(p)

    with torch.no_grad():
        for name, p in model.named_parameters():
            print(name, p)

            if 'weight' in name:
                p.copy_(torch.ones(p.shape))
            elif 'bias' in name:
                p.zero_()
            else:
                raise ValueError('Unknown parameter name "%s"' % name)
'''
