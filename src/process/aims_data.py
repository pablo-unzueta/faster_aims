import os
from dataclasses import dataclass
import pytorch_lightning as pl
from src.representations.representations import SoapDescriptor
import numpy as np
from ase.io import read
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader

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

    def __post_init__(self):
        self.descriptor = SoapDescriptor(self.structures, species=self.unique_atoms)

    @property
    def compute_descriptor(self):
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

    # def normalize_energies(self):
    #     """
    #     Each column is normalized relative to itself, since energy ranges vary
    #     index 0 is S0
    #     :return: normalized energies
    #     """
    #     potentials = self.potentials.copy()
    #
    #     _, states = potentials.shape
    #     normalized_energies = np.empty_like(potentials)
    #     for i in range(states):
    #         scaler = StandardScaler()
    #         relevant_pot = potentials[:, i].reshape(-1, 1)
    #         scaler.fit(relevant_pot)
    #         normalized_energies[:, i] = scaler.transform(relevant_pot).reshape(1, -1)

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


@dataclass
class LitAimsDataModule(pl.LightningDataModule):
    """
    Aims Pytorch Dataset using Pytorch Lightning DataModule.
    """

    def __init__(self, aims_parser, batch_size=32, ex_shift=0, cutoff=0.2):
        self.batch_size = batch_size
        self.ex_shift = ex_shift  # ExShift in Control.dat of aims run
        self.cutoff = cutoff  # Cutoff for nonadiabiatic coupling region in eV
        self.aims_parser = aims_parser
        # TODO change hard dependency of SoapDescriptor class
        # Need to make it more FLEXIBLE
        self.descriptor = SoapDescriptor
        # I know Alessio is going to hate this, but my IDE keeps yelling at me unless I put this here for now
        # Dio Maiale
        self.structures = None
        self.potentials = None
        self.representation = None
        self.normalized_energies = None
        self.normalized_energies_without_coupling_region = None

    def prepare_data(self):
        # Load in Data
        self.structures = self.aims_parser.structures
        self.potentials = self.aims_parser.potentials
        self.representation = self.descriptor(self.structures, self.get_unique_atoms()).representation

    def get_unique_atoms(self):
        unique_atoms = []
        atoms = self.structures[0].get_chemical_symbols()
        [unique_atoms.append(atom) for atom in atoms if atom not in unique_atoms]
        unique_atoms.sort()
        return unique_atoms

    def setup(self):
        self.normalize_energies()
        self.remove_coupling_region()
        self.split_data()
        # Split Data

        return None

    def split_data(self):
        self.train = None

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

        self.normalized_energies = normalized_energies

    def remove_coupling_region(self):
        coupling_structures = []
        potentials = self.potentials.copy()
        n_structures, states = potentials.shape
        energy_diff = np.empty([n_structures, states, states])  # n x states x states to determine coup

        for i in range(states):
            energy_diff[:, i] = (potentials.T - potentials[:, i]).T
        energy_diff *= HARTREE_TO_EV

        count = 0
        for state in range(states):
            coupling_structures_per_state = []
            for j, energies in enumerate(energy_diff[:, state]):
                energies = np.delete(energies, state, 0)
                indices = np.argwhere(abs(energies) < 0.2)
                if len(indices) > 0:
                    count += 1
                    coupling_structures_per_state.append(j)
            coupling_structures.append(coupling_structures_per_state)

        self.coupling_structures = coupling_structures

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size)
    #
    # def val_dataloader(self):
    #     return DataLoader(self.val, batch_size=self.batch_size)
    #
    # def test_dataloader(self):
    #     return DataLoader(self.test, batch_size=self.batch_size)
