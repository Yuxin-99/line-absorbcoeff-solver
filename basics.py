'''
Code based on Radis Source code "radis.lbl.base", "radis.lbl.factory"
'''

import drjit as dr
import mitsuba as mi
import numpy as np
import pandas as pd

from numpy import arange
from os.path import join
from radis.api.hitranapi import hit2df

# %% HITRAN ids

trans = {
    "1": "H2O",
    "2": "CO2",
    "3": "O3",
    "4": "N2O",
    "5": "CO",
    "6": "CH4",
    "7": "O2",
    "8": "NO",
    "9": "SO2",
    "10": "NO2",
    "11": "NH3",
    "12": "HNO3",
    "13": "OH",
    "14": "HF",
    "15": "HCl",
    "16": "HBr",
    "17": "HI",
    "18": "ClO",
    "19": "OCS",
    "20": "H2CO",
    "21": "HOCl",
    "22": "N2",
    "23": "HCN",
    "24": "CH3Cl",
    "25": "H2O2",
    "26": "C2H2",
    "27": "C2H6",
    "28": "PH3",
    "29": "COF2",
    "30": "SF6",
    "31": "H2S",
    "32": "HCOOH",
    "33": "HO2",
    "34": "O",
    "35": "ClONO2",
    "36": "NO+",
    "37": "HOBr",
    "38": "C2H4",
    "39": "CH3OH",
    "40": "CH3Br",
    "41": "CH3CN",
    "42": "CF4",
    "43": "C4H2",
    "44": "HC3N",
    "45": "H2",
    "46": "CS",
    "47": "SO3",
    "48": "C2N2",
    "49": "COCl2",
    "50": "SO",
    "51": "CH3F",
    "52": "GeH4",
    "53": "CS2",
    "54": "CH3I",
    "55": "NF3",
}
HITRAN_MOLECULES = list(trans.values())
""" str: list of [HITRAN-2020]_ molecules. """


# parameters used in current test settings
Tref = 296      # unit: K
# pressure_mbar = 1013.25     # unit: mbar
# pressure_atm = pressure_mbar / 1e3      # unit: bar/ 1e5 pa / 1e3 hpa


drop_columns = ['ierr', 'iref', 'lmix', 'gpp']

drop_all_but_these = [
    "id",
    "iso",
    "wav",
    "int",
    "A",
    "airbrd",
    "selbrd",
    "Tdpair",
    "Tdpsel",
    "Pshft",
    "El",
    "gp",
]


data_path = "~/Desktop/SP/Database"


def read_mol_mass_df(file):
    df = pd.read_csv(file, comment="#", delim_whitespace=True)
    df = df.set_index(["id", "iso"])

    return df


molparam_file = join(data_path, "molparam.txt")
molparam_df = read_mol_mass_df(molparam_file)


def get_molecule_identifier(molecule_name):
    r"""
    For a given input molecular formula, return the corresponding
    :py:data:`~radis.db.classes.HITRAN_MOLECULES` identifier number [1]_.

    Parameters
    ----------
    molecule_name : str
        The string describing the molecule.

    Returns
    -------
    M: int
        The HITRAN molecular identified number.

    References
    ----------
    .. [1] `HITRAN 1996, Rothman et al., 1998 <https://www.sciencedirect.com/science/article/pii/S0022407398000788>`__

    Function is from from https://github.com/nzhagen/hitran/blob/master/hitran.py

    """

    # Invert the dictionary.
    trans2 = {v: k for k, v in trans.items()}

    try:
        return int(trans2[molecule_name])
    except KeyError:
        raise NotImplementedError(
            "Molecule '{0}' not supported. Choose one of {1}".format(
                molecule_name, sorted(list(trans2.keys()))
            )
        )


# helper functions of reading the data frame
def remove_unecessary_columns(df):
    """Remove unecessary columns and add values as attributes

    Returns
    -------
    None: DataFrame updated inplace
    """

    # Discard molecule column if unique
    if "id" in df.columns:
        id_set = df.id.unique()
        if len(id_set) != 1:  # only 1 molecule supported ftm
            raise NotImplementedError(
                "Only 1 molecule at a time is currently supported "
                + "in SpectrumFactory. Use radis.calc_spectrum, which "
                + "calculates them independently then use MergeSlabs"
            )

        df.drop("id", axis=1, inplace=True)
        # df_metadata.append("id")
        df.attrs["id"] = id_set[0]
    else:
        assert "id" in df.attrs or "molecule" in df.attrs

    if "iso" in df.columns:
        isotope_set = df.iso.unique()

        if len(isotope_set) == 1:
            df.drop("iso", axis=1, inplace=True)
            # df_metadata.append("iso")
            df.attrs["iso"] = isotope_set[0]
    else:
        assert "iso" in df.attrs


def load_df(file, wavenum_min, wavenum_max, isotope, db_use_cached, verbose):
    df = hit2df(
        file,
        cache=db_use_cached,
        # load_columns=columns,  # not possible with "pytables-fixed"
        verbose=verbose,
        drop_non_numeric=True,
        load_wavenum_min=wavenum_min,
        load_wavenum_max=wavenum_max,
        engine="pytables",
    )

    # Drop columns (helps fix some Memory errors)
    dropped = []
    for col in df.columns:
        if col in drop_columns or (
                drop_columns == "all" and col not in drop_all_but_these
        ):
            del df[col]
            dropped.append(col)
    if verbose >= 2 and len(dropped) > 0:
        print("Dropped columns: {0}".format(dropped))

    # Crop to the wavenumber of interest
    # TODO : is it still needed since we use load_only_wavenum_above ?
    df = df[(df.wav >= wavenum_min) & (df.wav <= wavenum_max)]
    # Select correct isotope(s)
    if isotope != "all":
        isotope = [float(k) for k in isotope.split(",")]
        df = df[df.iso.isin(isotope)]

    # Finally: Concatenate all
    frames = [df]
    df = pd.concat(frames, ignore_index=True)  # reindex

    remove_unecessary_columns(df)
    return df


def get_indices(arr_i, axis):
    pos = np.interp(arr_i, axis, np.arange(axis.size))
    index = pos.astype(np.int32)
    return index, index + 1, pos - index


def dr_get_indices(arr_i, axis):
    # TODO: drjit version
    pos = np.interp(arr_i, axis, np.arange(axis.size))
    index = pos.astype(np.int32)
    return index, index + 1, pos - index


def get_molar_mass(df):
    """Returns molar mass for all lines of DataFrame ``df``.

    Parameters
    ----------
    df: pd.DataFrame

    Returns
    -------
    The molar mass of all the isotopes in the dataframe

    """
    # molpar = self.molparam

    if "id" in df.columns:
        raise NotImplementedError(
            "'id' still in DataFrame columsn. Is there more than 1 molecule ?"
        )
    if "id" in df.attrs:
        molecule = df.attrs["id"]
    else:
        molecule = df.attrs["molecule"]

    if "iso" in df.columns:
        iso_set = df.iso.unique()
        molar_mass_dict = {}
        for iso in iso_set:
            molar_mass_dict[iso] = molparam_df.loc[(molecule, iso), "molar_mass"]
        molar_mass = df["iso"].map(molar_mass_dict)
    else:
        iso = df.attrs["iso"]
        molar_mass = molparam_df.loc[(molecule, iso), "molar_mass"]

    return molar_mass


def dr_get_molar_mass(dfcolumn_dict, molecule, isotope, num_of_lines):
    """Returns molar mass for all lines of DataFrame ``df``.

    Parameters
    ----------
    dfcolumn_dict: dataFrame in dictionary of mitsuba array format
    molecule: molecule set of all molecule ids
    isotope: iso set of all isotopes
    num_of_lines: total number of lines (length of the dfcolumn_dict)

    Returns
    -------
    The molar mass of all the isotopes in the dataframe

    """
    # case1: only one molecule
    if len(molecule) == 1:
        molecule_id = molecule[0]
        if len(isotope) == 1:
            molar_mass = molparam_df.loc[(molecule_id, isotope[0]), "molar_mass"]
        else:
            molar_mass = dr.ones(mi.Float64, num_of_lines)
            for iso in isotope:
                mask = dr.eq(dfcolumn_dict["iso"], iso)
                mass = molparam_df.loc[(molecule_id, iso), "molar_mass"]
                molar_mass = dr.select(mask, mass, molar_mass)
        return molar_mass

    # case 2: multiple molecules
    molar_mass = dr.ones(mi.Float64, num_of_lines)
    for molecule_id in molecule:
        mask_mol = dr.eq(dfcolumn_dict["id"], molecule_id)
        for iso in isotope:
            mask_iso = dr.eq(dfcolumn_dict["iso"], iso)
            mask = mask_mol & mask_iso
            if mask == False:
                # there is no line data for this combination (molecule_id & iso)
                continue
            mass = molparam_df.loc[(molecule_id, iso), "molar_mass"]
            molar_mass = dr.select(mask, mass, molar_mass)
    return molar_mass


def generate_wavenumber_range(wavenum_min, wavenum_max, wstep, neighbour_lines=0):
    """define waverange vectors, with ``wavenumber`` the output spectral range
    and ``wavenumber_calc`` the spectral range used for calculation, that
    includes neighbour lines within ``neighbour_lines`` distance.

    Parameters
    ----------
    wavenum_min, wavenum_max: float
        wavenumber range limits (cm-1)
    wstep: float
        wavenumber step (cm-1)
    neighbour_lines: float
        wavenumber full width of broadening calculation: used to define which
        neighbour lines shall be included in the calculation
        only consider the case as 0 for now

    Returns
    -------
    wavenumber: numpy array
        an evenly spaced array between ``wavenum_min`` and ``wavenum_max`` with
        a spacing of ``wstep``
    wavenumber_calc: numpy array
        an evenly spaced array between ``wavenum_min-neighbour_lines`` and
        ``wavenum_max+neighbour_lines`` with a spacing of ``wstep``
    woutrange: (wmin, wmax)
        index to project the full range including neighbour lines `wavenumber_calc`
        on the final range `wavenumber`, i.e. : wavenumber_calc[woutrange[0]:woutrange[1]] = wavenumber
    """
    assert wavenum_min < wavenum_max
    assert wstep > 0

    # not consider the case of wstep = 'auto' for now

    # Output range
    # generate the final vector of wavenumbers (shape M)
    wavenumber = arange(wavenum_min, wavenum_max + wstep, wstep)

    # generate the calculation vector of wavenumbers (shape M + space on the side)
    # ... Calculation range
    wavenum_min_calc = wavenumber[0] - neighbour_lines  # cm-1
    wavenum_max_calc = wavenumber[-1] + neighbour_lines  # cm-1
    # TODO: drjit version arange?
    w_out_of_range_left = arange(
        wavenumber[0] - wstep, wavenum_min_calc - wstep, -wstep
    )[::-1]
    w_out_of_range_right = arange(
        wavenumber[-1] + wstep, wavenum_max_calc + wstep, wstep
    )

    # ... deal with rounding errors: 1 side may have 1 more point
    if len(w_out_of_range_left) > len(w_out_of_range_right):
        w_out_of_range_left = w_out_of_range_left[1:]
    elif len(w_out_of_range_left) < len(w_out_of_range_right):
        w_out_of_range_right = w_out_of_range_right[:-1]

    wavenumber_calc = np.hstack((w_out_of_range_left, wavenumber, w_out_of_range_right))
    woutrange = len(w_out_of_range_left), len(w_out_of_range_left) + len(wavenumber)

    assert len(w_out_of_range_left) == len(w_out_of_range_right)
    assert len(wavenumber_calc) == len(wavenumber) + 2 * len(w_out_of_range_left)

    return wavenumber, wavenumber_calc, woutrange


def dr_generate_wavenumber_range(wavenum_min, wavenum_max, wstep, neighbour_lines=0):
    """define waverange vectors, with ``wavenumber`` the output spectral range
    and ``wavenumber_calc`` the spectral range used for calculation, that
    includes neighbour lines within ``neighbour_lines`` distance.

    Parameters
    ----------
    wavenum_min, wavenum_max: float
        wavenumber range limits (cm-1)
    wstep: float
        wavenumber step (cm-1)
    neighbour_lines: float
        wavenumber full width of broadening calculation: used to define which
        neighbour lines shall be included in the calculation
        only consider the case as 0 for now

    Returns
    -------
    wavenumber: numpy array
        an evenly spaced array between ``wavenum_min`` and ``wavenum_max`` with
        a spacing of ``wstep``
    wavenumber_calc: numpy array
        an evenly spaced array between ``wavenum_min-neighbour_lines`` and
        ``wavenum_max+neighbour_lines`` with a spacing of ``wstep``
    woutrange: (wmin, wmax)
        index to project the full range including neighbour lines `wavenumber_calc`
        on the final range `wavenumber`, i.e. : wavenumber_calc[woutrange[0]:woutrange[1]] = wavenumber
    """
    assert wavenum_min < wavenum_max
    assert wstep > 0

    # not consider the case of wstep = 'auto' for now

    # Output range
    # generate the final vector of wavenumbers (shape M)
    wavenumber = dr.arange(dtype=mi.Float64, start=wavenum_min, stop=wavenum_max + wstep, step=wstep)

    # generate the calculation vector of wavenumbers (shape M + space on the side)
    # ... Calculation range
    wavenum_min_calc = wavenumber[0] - neighbour_lines  # cm-1
    wavenum_max_calc = wavenumber[-1] + neighbour_lines  # cm-1
    w_out_of_range_left = dr.arange(
        dtype=mi.Float64, start=wavenumber[0] - wstep, stop=wavenum_min_calc - wstep, step=-wstep
    )[::-1]
    w_out_of_range_right = dr.arange(
        dtype=mi.Float64, start=wavenumber[-1] + wstep, stop=wavenum_max_calc + wstep, step=wstep
    )

    # ... deal with rounding errors: 1 side may have 1 more point
    if len(w_out_of_range_left) > len(w_out_of_range_right):
        w_out_of_range_left = w_out_of_range_left[1:]
    elif len(w_out_of_range_left) < len(w_out_of_range_right):
        w_out_of_range_right = w_out_of_range_right[:-1]

    len_wavenum = len(wavenumber)
    len_left = len(w_out_of_range_left)
    len_right = len(w_out_of_range_right)

    wavenumber_calc = dr.zeros(mi.Float64, len_left + len_wavenum + len_right)
    wavenumber_calc[:len_left] = w_out_of_range_left
    wavenumber_calc[len_left:(len_left + len_wavenum)] = wavenumber
    wavenumber_calc[(len_left + len_wavenum):] = w_out_of_range_right
    woutrange = len_left, len_left + len_wavenum

    assert len_left == len_right
    assert len(wavenumber_calc) == len_wavenum + 2 * len_left

    return wavenumber, wavenumber_calc, woutrange
