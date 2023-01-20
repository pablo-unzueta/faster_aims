from src.models.mlp import SoapMLP
from src.representations.representations import SoapDescriptor
from ase.build import molecule


def test_soap_mlp() -> None:
    water = molecule("H2O")
    unique_atoms = []
    atoms = water.get_chemical_symbols()
    [unique_atoms.append(atom) for atom in atoms if atom not in unique_atoms]
    waters = [water, water]
    soap_descriptor = SoapDescriptor(waters, species=unique_atoms)
    soap_mlp = SoapMLP(soap_descriptor, 64, 1)
    y_pred = soap_mlp(soap_descriptor)
    assert y_pred.shape == 0

