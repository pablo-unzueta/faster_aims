import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from torch import Tensor
from torch.nn import (
    Sequential,
    Linear,
    ReLU,
    GRU,
    Embedding,
    BatchNorm1d,
    Dropout,
    LayerNorm,
)


class SoapMLP(nn.Module):
    def __init__(self, data, dim1, fc_count, **kwargs):
        """

        :type data: torch.Dataset containing the structures and potentials
        :type dim1: int number of neurons in each layer
        :type fc_count: int number of layers in NN
        """
        super(SoapMLP, self).__init__()

        self.lin1 = nn.Linear(data.shape[2], dim1)  # Assumes 3D Tensor where 2 is the size of SOAP
        self.lin_list = nn.ModuleList(
            [nn.Linear(dim1, dim1) for i in range(fc_count)]
        )
        self.lin2 = nn.Linear(dim1, 1)

    def forward(self, data):
        out = F.relu(self.lin1(data.representation))
        for layer in self.lin_list:
            out = F.relu(layer(out))
        out = self.lin2(out)
        if out.shape[1] == 1:
            return out.view(-1)
        else:
            return out





# model = MLP()
# model[0]
# model = torch.nn.Sequential(
#     torch.nn.Linear(3, 1),
#     torch.nn.Flatten(0, 1)
# )
# model[0]
# x = torch.linspace(-math.pi, math.pi, 2000)
# y = torch.sin(x)
#
# p = torch.tensor([1, 2, 3])
# xx = x.unsqueeze(-1).pow(p)
#
# loss_fn = nn.MSELoss(reduction='sum')
# learning_rate = 1e-6
#
# for t in range(2000):
#     y_pred = model(xx)
#     loss = loss_fn(y_pred, y)
#     # if t % 100 == 99:
#         # print(t, loss.item())
#
#     model.zero_grad()
#     loss.backward()
#
#     with torch.no_grad():
#         for param in model.parameters():
#             param -= learning_rate * param.grad
#
#     linear_layer = model[0]

    # print(f'Result: y = {linear_layer.bias.item()} + {linear_layer.weight[:, 0].item()} x + {linear_layer.weight[:, 1].item()} x^2 + {linear_layer.weight[:, 2].item()} x^3')
