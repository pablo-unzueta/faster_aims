import os
from pathlib import Path
from src.process.aims_data import AimsData, AimsParser
import ase
import numpy

# Recreate dataset from given aims runs
# learning with holes


def process(path, ex_shift, cutoff):
    aims_parser = AimsParser(path)
    # aims_parser.create_train_val_test_splits_dir()
    aims_data = AimsData(aims_parser, ex_shift, cutoff)
    aims_data.descriptor
    # aims_data.remove_coupling_region()


if __name__ == "__main__":
    data_dir = '//data/unique_structures/splits'
    data_path = Path(data_dir)
    if data_path.exists():
        print(f"{data_path} exists")

    train_path = os.path.join(data_dir, "train")
    val_path = os.path.join(data_dir, "val")
    test_path = os.path.join(data_dir, "test")
    sample_path = os.path.join(data_dir, "sample")

    process(sample_path, ex_shift=78.5, cutoff=0.2)
