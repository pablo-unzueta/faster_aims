from src.process.aims_data import AimsData, AimsParser
from pathlib import Path
import numpy as np
from ase import Atoms


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
