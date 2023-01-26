from ase.build import molecule  # just for testing, remove this line later
from dscribe.descriptors import SOAP, CoulombMatrix
from dataclasses import dataclass
from ase import Atoms
from typing import List
import torch

# Need to make generic descriptor class that can hold anything?
# @dataclass
# class Descriptor():

@dataclass
class CoulombMatrixDescriptor(CoulombMatrix):
    """Generate CM Descriptor using dscribe backend

    CM Descriptor is outdated but is useful for learning purposes.

    Attributes:
        n_atoms_max (int): Number of atoms in your largest molecule. CM is zero padded
        flatten (bool): Flatten to 1D Array?
        permutation (str): Defines the method for handling permutational invariance. Can be one of the following:
            none: The matrix is returned in the order defined by the Atoms.
            sorted_l2: The rows and columns are sorted by the L2 norm.
            eigenspectrum: Only the eigenvalues are returned sorted by their absolute value in descending order.
    """
    n_atoms_max: int
    flatten: bool
    permutation: str = "none"

    # def __post_init__(self):
    #     self.descriptor = CoulombMatrix(
    #         n_atoms_max=self.n_atoms_max,
    #         flatten=self.flatten,
    #         permutation=self.permutation
    #     )


@dataclass
class SoapDescriptor:
    """ Class to return SOAP Descriptor """
    atoms: Atoms
    species: List[str]
    r_cut: float = 6.0
    n_max: int = 8
    l_max: int = 6
    periodic: bool = False
    average: str = "off"

    def __post_init__(self):
        self.descriptor = SOAP(
            species=self.species,
            periodic=self.periodic,
            r_cut=self.r_cut,
            n_max=self.n_max,
            l_max=self.l_max,
        )

    @property
    def representation(self):
        return torch.Tensor(self.descriptor.create(self.atoms, n_jobs=-1))

    @property
    def shape(self):
        return self.representation.shape
