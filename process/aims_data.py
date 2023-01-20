from typing import List
from ase.io import read
from ase import Atoms
import os
import numpy as np
from dataclasses import dataclass
from torch.utils.data import Dataset
from src.representations.representations import SoapDescriptor
# https://singroup.github.io/dscribe/latest/tutorials/descriptors/soap.html
from sklearn.preprocessing import StandardScaler

HARTREE_TO_KCAL_MOL = 627.509608
HARTREE_TO_KJ_MOL = 2625.5002
HARTREE_TO_EV = 27.211399


@dataclass
class AimsParser:
    path: str

    @property
    def potential_files(self):
        """

        :return: sorted list of potential files from directory
        """
        return self.get_files_from_label("PotEn")

    @property
    def structure_files(self):
        return self.get_files_from_label("positions")

    def get_files_from_label(self, label, scr_dir="scr."):
        files = []
        for root, dirs, file in os.walk(self.path):
            # for dir in dirs:
            for f in file:
                if label in f and scr_dir not in root:
                    files.append(os.path.join(root, f))
        files.sort()
        return files

    # def create_train_val_test_splits_dir(self):

    @property
    def potentials(self):
        """
        Need to delete first and last column (time and classical energy)
        :return: np.ndarray of potentials from aims
        """
        return np.delete(np.delete(np.vstack([np.genfromtxt(file)
                                              for file in self.potential_files]), 0, axis=1), -1, axis=1)

    @property
    def structures(self):
        # Equivalent to the following two lines :)
        #  atoms = [read(file, index=":") for file in self.structure_files]
        #  return [structure for inner_list in atoms for structure in inner_list]
        return [structure for inner_list in [read(file, index=":")
                                             for file in self.structure_files] for structure in inner_list]


@dataclass
class AimsData(Dataset):
    """ AIMS PyTorch DATASET """
    aims_parser: AimsParser
    ExShift: float = 0
    cutoff: float = 0  # in eV for simplicity

    @property
    def descriptor(self):
        return self.descriptor.representation

    @property
    def potentials(self):
        return self.aims_parser.potentials

    @property
    def structures(self):
        return self.aims_parser.structures

    @property
    def single_structure(self):
        return self.aims_parser.structures[0]

    @property
    def unique_atoms(self):
        unique_atoms = []
        atoms = self.single_structure.get_chemical_symbols()
        [unique_atoms.append(atom) for atom in atoms if atom not in unique_atoms]
        unique_atoms.sort()
        return unique_atoms

    def __len__(self):
        return len(self.potentials)

    def __getitem__(self, idx):
        return {'potentials': self.potentials[idx], 'structure': self.structures[idx]}

    def normalize_energies(self):
        """
        Each column is normalized relative to itself, since energy ranges vary
        index 0 is S0
        :return: normalized energies
        """
        potentials = self.potentials.copy()

        _, states = potentials.shape
        normalized_energies = np.empty_like(potentials)
        for i in range(states):
            scaler = StandardScaler()
            relevant_pot = potentials[:, i].reshape(-1, 1)
            scaler.fit(relevant_pot)
            normalized_energies[:, i] = scaler.transform(relevant_pot).reshape(1, -1)

    # def remove_coupling_region(self):
    #     potentials = self.potentials.copy()
    #     _, states = potentials.shape
    #     shape = potentials.shape
    #     shape = shape + (3)
    #     energy_diff = np.empty(shape)
    #     print(energy_diff.shape)
    #     exit()
    #     print(energy_diff.shape)
    #     for i in range(states):
    #         energy_diff[:, i] = potentials.T - potentials[:, i]
    #     energy_diff *= HARTREE_TO_EV
    #     # TODO: Start here
    #     count = 0
    #     for i in energy_diff[:, 1]:
    #         if abs(i) < 0.2:
    #             count += 1
    #     print(len(energy_diff))
    #     print(count)
