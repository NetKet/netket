import pytest
import netket as nk


@pytest.fixture
def _mpi_size(request):
    return nk.utils.n_nodes


@pytest.fixture
def _mpi_rank(request):
    return nk.utils.rank


@pytest.fixture
def _mpi_comm(request):
    from netket.utils import MPI_py_comm

    return MPI_py_comm


@pytest.fixture
def MPI(request):
    from mpi4py import MPI

    return MPI


def pytest_addoption(parser):
    parser.addoption(
        "--legacy",
        action="store_true",
        dest="legacy",
        default=False,
        help="enable legacy tests",
    )

    parser.addoption(
        "--legacy-only",
        action="store_true",
        dest="only_legacy",
        default=False,
        help="enable legacy tests and disable everything else",
    )


def pytest_configure(config):
    if not config.option.legacy:
        setattr(config.option, "markexpr", "not legacy")

    if config.option.only_legacy:
        setattr(config.option, "markexpr", "legacy")

    if nk.utils.n_nodes > 1:
        setattr(config.option, "markexpr", "not legacy")
