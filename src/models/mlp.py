import pytorch_lightning as pl
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


# class LitSoapMLP(pl.LightningModule):
#     def __init__(self, soap_mlp):
#         super().__init__()
#         self.soap_mlp = soap_mlp
#         self.save_hyperparameters()
#
#     def training_step(self, batch, batch_idx):
#         # training_step defines the training loop
#         x, y = batch
#         x = x.view(x.size(0), -1)
#         y_pred = self.soap_mlp(x)
#         train_loss = F.mse_loss(y_pred, y)
#         self.log('train_loss', train_loss)
#         return train_loss
#
#     def validation_step(self, val_batch, batch_idx):
#         # validation_step defines the val loop
#         x, y = val_batch
#         x = x.view(x.size(0), -1)
#         y_pred = self.soap_mlp(x)
#         val_loss = F.mse_loss(y_pred, y)
#         self.log('val_loss', val_loss)
#         return val_loss
#
#     def test_step(self, batch, batch_idx):
#         # test_step defines the val loop
#         x, y = batch
#         x = x.view(x.size(0), -1)
#         y_pred = self.soap_mlp(x)
#         test_loss = F.mse_loss(y_pred, y)
#         self.log('test_loss', test_loss)
#         return test_loss
#
#     def configure_optimizers(self):
#         print("⚡", "using LitSoapMLP", "⚡")
#         return torch.optim.Adam(self.parameters(), lr=1e-3)

