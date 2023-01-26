from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import pytorch_lightning as pl
from src.models.mlp import SoapMLP, LitSoapMLP
from torch.utils.data import DataLoader


def train():
    train_loader = DataLoader(train_set)
    val_loader = DataLoader(val_loader)
    model = LitSoapMLP(SoapMLP())
    trainer = pl.Trainer(callbacks=EarlyStopping(monitor='val_loss', mode='min'),
                         check_val_every_n_epoch=10)
    trainer.fit(model, train_loader, valid_loader)