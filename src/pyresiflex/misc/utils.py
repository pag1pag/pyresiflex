"""Library of useful helper functions."""

from pathlib import Path

import numpy as np

ROOT_FOLDER_PATH: Path = Path(__file__).parent.parent


def gaussian_sigma_from_fwhm(fwhm: float) -> float:
    r"""Return the Gaussian width ``sigma`` for a given FWHM.

    The pulse is parametrised as :math:`\exp(-((t - \mu) / \sigma)^2)`
    (note: no factor of two in the exponent denominator), for which the
    full width at half maximum is :math:`2 \sqrt{\ln 2}\, \sigma`.

    Parameters
    ----------
    fwhm : float
        Full width at half maximum of the Gaussian pulse.

    Returns
    -------
    float
        The corresponding standard-deviation-like width ``sigma``.

    Examples
    --------
    >>> import numpy as np
    >>> from pyresiflex.misc.utils import gaussian_sigma_from_fwhm
    >>> bool(np.isclose(gaussian_sigma_from_fwhm(5.0), 3.0028, atol=1e-4))
    True
    """
    return fwhm * np.sqrt(2) / (np.sqrt(2 * np.log(2)) * 2)


def gaussian_fwhm_from_sigma(sigma: float) -> float:
    r"""Return the FWHM for a given Gaussian width ``sigma``.

    Inverse of :func:`gaussian_sigma_from_fwhm`.

    Parameters
    ----------
    sigma : float
        Standard-deviation-like width of the Gaussian pulse.

    Returns
    -------
    float
        The corresponding full width at half maximum.

    Examples
    --------
    >>> import numpy as np
    >>> from pyresiflex.misc.utils import (
    ...     gaussian_fwhm_from_sigma,
    ...     gaussian_sigma_from_fwhm,
    ... )
    >>> sigma = gaussian_sigma_from_fwhm(5.0)
    >>> bool(np.isclose(gaussian_fwhm_from_sigma(sigma), 5.0))
    True
    """
    return 2 * np.sqrt(np.log(2)) * sigma


def get_root() -> Path:
    """Return the full path of the pyresiflex folder.

    Used not to worry about the project architecture.

    Returns
    -------
    Path
        the abspath to root folder (should end with 'pyresiflex')

    Examples
    --------
    >>> from pyresiflex.misc.utils import get_root
    >>> path = get_root() / "data"
    """
    return ROOT_FOLDER_PATH


def get_path_to_data(*paths: str, force_return: bool = False) -> Path:
    """Return the absolute path to the data folder, or file inside.

    Parameters
    ----------
    *paths : str
        You can add a path to precise the folder inside.
    force_return : bool, optional
        If True, return path even if does not exists, by default False.

    Returns
    -------
    Path
        The abspath to the data (or file).

    Examples
    --------
    >>> from pyresiflex.misc.utils import get_path_to_data
    >>> path = get_path_to_data()
    >>> path = get_path_to_data("Minesi2022")

    Raises
    ------
    FileNotFoundError
        If the file or folder is not found.
    """
    path_to_data_folder = get_root().joinpath("data", *paths)

    if not (path_to_data_folder.exists() or force_return):
        raise FileNotFoundError(
            f"File or folder not found: {path_to_data_folder}"
        )

    return path_to_data_folder.resolve()
