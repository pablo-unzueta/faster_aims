from src.representations import SoapDescriptor, CoulombMatrixDescriptor
from ase.build import molecule


def test_soap_descriptor_default() -> None:
    water = molecule("H2O")
    unique_atoms = []
    atoms = water.get_chemical_symbols()
    [unique_atoms.append(atom) for atom in atoms if atom not in unique_atoms]
    soap_descriptor = SoapDescriptor(water, species=unique_atoms)
    assert soap_descriptor.representation.shape[0] == 3
    assert soap_descriptor.representation.shape[1] == 952


def test_soap_descriptor() -> None:
    water = molecule("H2O")
    unique_atoms = []
    atoms = water.get_chemical_symbols()
    [unique_atoms.append(atom) for atom in atoms if atom not in unique_atoms]
    soap_descriptor = SoapDescriptor(water, species=unique_atoms, n_max=4, l_max=4)
    assert soap_descriptor.representation.shape[0] == 3
    assert soap_descriptor.representation.shape[1] == 180
