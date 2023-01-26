import torch
from pytorch_lightning.cli import LightningCLI
from src.process.aims_data import LitAimsDataModule, AimsParser, AimsData
from src.models.mlp import LitSoapMLP, SoapMLP


def cli_main():
    cli = LightningCLI()

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print(f"{torch.__version__}")
    print("Welcome to Super sIMPle Neural network Demo")
    print("SIMPN-Demo")
