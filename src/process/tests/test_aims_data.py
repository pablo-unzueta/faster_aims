from pathlib import Path
import numpy as np
from ase import Atoms
from ase.build import molecule
from src.process.aims_data import AimsParser, AimsData, LitAimsDataModule
from src.representations.representations import SoapDescriptor


def test_aims_parser() -> None:
    errors = []
    path = Path.cwd() / "sample_data"
    full_path = path.resolve()
    aims_parser = AimsParser(full_path)

    if len(aims_parser.potential_files) == 0:
        errors.append("Cannot find potential files")
    if len(aims_parser.structure_files) == 0:
        errors.append("Cannot find structure files")
    if not isinstance(aims_parser.potentials, np.ndarray):
        errors.append("Cannot read potential files")
    if not isinstance(aims_parser.structures[0], Atoms):  # Only need to check first Atoms object
        errors.append("Cannot read structure files")

    assert not errors, "errors occurred:\n{}".format("\n".join(errors))


def test_aims_data() -> None:
    errors = []
    path = Path.cwd() / "sample_data"
    full_path = path.resolve()
    aims_parser = AimsParser(full_path)
    aims_data = AimsData(aims_parser)
    shape = [aims_data.compute_descriptor.size(dim=0),
             aims_data.compute_descriptor.size(dim=1),
             aims_data.compute_descriptor.size(dim=2)]

    if not shape == [507, 6, 952]:
        errors.append("Could not compute descriptor")
    if not isinstance(aims_data.structures[0], Atoms):
        errors.append("Empty list of structures")
    if not isinstance(aims_data.potentials, np.ndarray):
        errors.append("Empty array of potentials")
    if len(aims_data.unique_atoms) == 0:
        errors.append("No Atoms found")
    if aims_data.__len__() == 0:
        errors.append("Empty Dataset")
    if not isinstance(aims_data.__getitem__(0), dict):
        errors.append("Cannot get items in dataset")


    # TODO
    # normalize_energies
    # remove_coupling_region
    # single structure
    # descriptor

    assert not errors, "errors occurred:\n{}".format("\n".join(errors))


def test_lit_aims_data_module() -> None:
    errors = []
    path = Path.cwd() / "sample_data"
    full_path = path.resolve()
    aims_parser = AimsParser(full_path)
    lit_aims_data = LitAimsDataModule(aims_parser)
    # PL has two main calls. First is prepare_data()
    # Second is setup(). Once these two are defined, everything else
    # Should behave smoothly
    lit_aims_data.prepare_data()

    if lit_aims_data.structures is None:
        errors.append("Could not load in structures")
    if lit_aims_data.potentials is None:
        errors.append("Could not load in potentials")
    if lit_aims_data.representation is None:
        errors.append("Could not compute descriptor")

    lit_aims_data.setup()
    if lit_aims_data.normalized_energies is None:
        errors.append("Could not normalize energies")
    if lit_aims_data.coupling_structures is None:
        errors.append("Could not remove coupling region")
    if lit_aims_data.train is None:
        errors.append("Could not split data")
    assert not errors, "errors occurred:\n{}".format("\n".join(errors))