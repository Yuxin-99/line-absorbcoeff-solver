"""
Code based on Radis Source code "radis.lbl.factory", "radis.lbl.base"
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import time

from basics import get_molecule_identifier, Tref, generate_wavenumber_range, get_molar_mass
from broadening import calc_lorentz_hwhm, calc_gauss_hwhm, calc_lineshape_LDM, apply_lineshpe_LDM, calc_continuum_absorb
from constants import c_2, k_b
from loader import load_hitran


def calc_lineshift(df, pressure_mbar):
    """Calculate lineshift due to pressure.

    Returns
    -------
    None: ``df`` is updated directly with new column ``shiftwav``

    References
    ----------
    Shifted line center based on pressure shift coefficient :math:`lambda_{shift}`
    and pressure :math:`P`.

    .. math::
        \\omega_{shift}=\\omega_0+\\lambda_{shift} P

    See Eq.(A13) in [Rothman-1998]_
    """

    start = time.time()
    air_pressure = pressure_mbar / 1013.25  # convert from mbar to atm

    if "Pshft" in df.columns:
        df["shiftwav"] = df.wav + (df.Pshft * air_pressure)
    else:
        print(
            "Pressure-shift coefficient not given in database: assumed 0 pressure shift"
        )
        df["shiftwav"] = df.wav

    # Sorted lines is needed for sparse wavenumber range algorithm.
    df.sort_values("shiftwav", inplace=True)

    elapsed = time.time() - start
    return elapsed


# compute S
def calc_linestrength(df, Tgas, molecule):
    """References
    ----------

    ..math::
    S(T) = S_0 \\frac{Q_{ref}}{Q_{gas}}
    \\operatorname{exp}\\left(-E_l \\left(\\frac{1}{T_{gas}}-\\frac{1}{T_{ref}}\\right)\\right)
    \\frac{1 -\\operatorname{exp}\\left(\\frac{-\\omega_0}{Tgas}\\right)}
    {1 -\\operatorname{exp}\\left(\\frac{-\\omega_0}{T_{ref}}\\right)}

    see notes: https://www.notion.so/RADIS-calc_spectrum-eedb4276deb343329a950b125a3bc542?pvs=4#04eaed5f30954e82ad0272e7f3d54acc
    """
    # compute S based on https://hitran.org/docs/definitions-and-units/#:~:text=Temperature%20dependence%20of%20the%20line%20intensity
    start = time.time()
    df['S'] = (
            df.int * Qref_Qgas_ratio(df, Tgas, molecule)
            * (np.exp(-c_2 * df.El * (1 / Tgas - 1 / Tref)))
            * ((1 - np.exp(-c_2 * df.wav / Tgas)) / (1 - np.exp(-c_2 * df.wav / Tref)))
    )
    elapsed = time.time() - start
    return elapsed


def Qref_Qgas_ratio(df, Tgas, molecule):
    """Calculate Qref/Qgas at temperature ``Tgas``, ``Tref``, for all lines
    of ``df1``. Returns a single value if all lines have the same Qref/Qgas ratio,
    or a column if they are different

    See Also
    --------
    :py:meth:`~radis.lbl.base.BaseFactory.Qgas`
    """

    if "id" in df.columns:
        id_set = df.id.unique()
        if len(id_set) > 1:
            raise NotImplementedError(">1 molecule.")
        else:
            df.attrs["id"] = int(id_set)

    # if "molecule" in df.attrs:
    #     molecule = df.attrs["molecule"]  # used for ExoMol, which has no HITRAN-id
    # else:
    #     molecule = get_molecule(df1.attrs["id"])

    molecule_id = get_molecule_identifier(molecule)
    from hapi import partitionSum

    if "iso" in df:
        iso_set = df.iso.unique()
        if len(iso_set) == 1:
            # if single isotope
            print("There shouldn't be a Column 'iso' with a unique value")
            isotope = int(iso_set)

            Q_ref = partitionSum(molecule_id, isotope, Tref)
            Q_gas = partitionSum(molecule_id, isotope, Tgas)
            df.attrs["Qgas"] = Q_gas
            df.attrs["Qref"] = Q_ref
            Qref_Qgas = Q_ref / Q_gas

        else:
            # if multiple isotopes
            Qref_Qgas_ratio_map = {}
            # {1: 8, 2: 10}
            for iso in iso_set:
                Q_ref = partitionSum(molecule_id, iso, Tref)
                Q_gas = partitionSum(molecule_id, iso, Tgas)
                Qref_Qgas_ratio_map[iso] = Q_ref / Q_gas
            Qref_Qgas = df["iso"].map(Qref_Qgas_ratio_map)

    else:
        # if single isotope
        iso = df.attrs["iso"]
        Q_ref = partitionSum(molecule_id, iso, Tref)
        Q_gas = partitionSum(molecule_id, iso, Tgas)
        df.attrs["Qgas"] = Q_gas
        df.attrs["Qref"] = Q_ref
        Qref_Qgas = Q_ref / Q_gas
    return Qref_Qgas


def get_spectrum(
        wmin,
        wmax,
        Tgas,
        molecule,
        isotope="all",
        mole_fraction=1.0,
        diluent="air",
        pressure=1.01325,
        wstep=0.01,
        # optimization="simple",
        # broadening_method="fft"
):
    """
    :param wmin: the minimum wavenumber, unit: cm-1
    :param wmax: the maximum wavenumber, unit: cm-1
    :param Tgas: the temperature, unit: K
    :param molecule: the queried molecule(s), str or list
    :param isotope: the queried isotope(s), str or dict
    :param mole_fraction: the fraction of molecule(s), float or dict
    :param diluent: the diluent, default (and for now only as) air
    :param pressure: the total gas pressure, unit: bar

    """
    start_time = time.time()

    pressure_mbar = pressure * 1e3

    if type(molecule) != list:
        molecule = [molecule]

    molecule_dict = {}
    for mol in molecule:
        molecule_dict[mol] = {}

    # convert the isotope according to the molecule input
    if type(isotope) == dict:
        for mol in molecule:
            if mol not in isotope.keys():
                raise Exception("Error: Missing isotope input for the molecule " + mol + "!!!")
            else:
                molecule_dict[mol]["iso"] = isotope[mol]
    else:
        assert (isinstance(isotope, str))
        for mol in molecule:
            molecule_dict[mol]["iso"] = isotope  # every molecule has the same isotope

    # set the correct diluent
    # mole_fraction is dictionary of multiple molecules and have non air diluent -> NotImplementedError
    if diluent != "air":
        raise NotImplementedError("Currently only support the case that there is only air as diluent!")

    if len(molecule) != 1 and type(mole_fraction) != dict:
        raise Exception("When given multiple molecules, please provide the exact mol_fraction of each molecule!!!")
    elif len(molecule) == 1 and (isinstance(mole_fraction, int) or isinstance(mole_fraction, float)):
        molecule_dict[molecule[0]]["mol_fraction"] = mole_fraction
        # diluent_fraction = 1.0 - mole_fraction
        # diluent = {"air": diluent_fraction}
    else:
        # mol_frac_sum = 0.0
        for mol in molecule:
            if mol not in mole_fraction.keys():
                raise Exception("Error: Missing mole fraction input for the molecule " + mol + "!!!")
            else:
                molecule_dict[mol]["mol_fraction"] = mole_fraction[mol]
                # mol_frac_sum += mole_fraction[mol]
        # diluent_fraction = 1.0 - mol_frac_sum
        # diluent = {"air": diluent_fraction}

    # loop through all the molecules
    abscoeff_res = 0.0
    perf_dict_list = []
    for mol in molecule_dict.keys():
        diluent = {"air": 1.0 - molecule_dict[mol]["mol_fraction"]}
        abscoeff, wave_number, df, perf_dict = calc_molecule_eq_spectrum(
            Tgas,
            pressure_mbar,
            mol,
            molecule_dict[mol]["iso"],
            molecule_dict[mol]["mol_fraction"],
            diluent,
            wmin,
            wmax,
            wstep
        )
        # ARTS: As absorption is additive, the total absorption coefficient is derived by adding up the
        # absorption contributions of all spectral lines of all molecular species.
        abscoeff_res += abscoeff
        perf_dict_list.append(perf_dict)

    # if "H2O" in molecule_dict.keys():
    #     abscoeff_res += calc_continuum_absorb(wave_number, mole_fraction["H2O"], pressure)

    total_time = time.time() - start_time
    print_perf(perf_dict_list, total_time)

    # the df here is just the dataframe of the last molecule, might not be used for testing when there are multiple molecules
    return abscoeff_res, wave_number, df


def calc_molecule_eq_spectrum(
        Tgas,
        pressure_mbar,
        molecule,
        isotope,
        mole_fraction,
        diluent,
        wavenum_min,
        wavenum_max,
        wstep
):
    df = load_hitran(
        wavenum_min=wavenum_min,
        wavenum_max=wavenum_max,
        molecule=molecule,
        isotope=isotope,
    )

    wavenumber, wavenumber_calc, woutrange = generate_wavenumber_range(wavenum_min, wavenum_max, wstep)

    perf_dict = {"Line strength": 0.0, "Line Shift": 0.0, "Lorentz hwhm": 0.0, "Gaussian hwhm": 0.0,
                 "Calculate line profile LDM": 0.0, "Apply LDM": 0.0}
    if len(df) == 0:
        # TODO: when no data for this molecule, should deal carefully with the related return vals
        return 0.0, wavenumber, df, perf_dict

    # df['S'], unit: cm−1/(molecule·cm−2)
    S_time = calc_linestrength(df, Tgas, molecule)
    perf_dict["Line strength"] = S_time
    shift_time = calc_lineshift(df, pressure_mbar)
    perf_dict["Line Shift"] = shift_time

    molar_mass = get_molar_mass(df)  # g/mol
    pressure_atm = pressure_mbar / 1013.25
    lorentz_hwhm_time = calc_lorentz_hwhm(df, Tgas, diluent, mole_fraction, pressure_atm)
    perf_dict["Lorentz hwhm"] = lorentz_hwhm_time
    gauss_hwhm_time = calc_gauss_hwhm(df, Tgas, molar_mass)
    perf_dict["Gaussian hwhm"] = gauss_hwhm_time

    # calculate the broadening (line shapes)
    line_profile_LDM, wL, wG, wL_dat, wG_dat, calc_ldm_time = calc_lineshape_LDM(df, wstep, wavenumber_calc)
    perf_dict["Calculate line profile LDM"] = calc_ldm_time
    # compute S * F, unit of the convolution result: 1/(molecule·cm−2)
    sumoflines, apply_time = apply_lineshpe_LDM(
        df['S'],
        wavenumber,
        wavenumber_calc,
        woutrange,
        df['shiftwav'],
        wstep,
        line_profile_LDM,
        wL,
        wG,
        wL_dat,
        wG_dat
    )
    perf_dict["Apply LDM"] = apply_time

    # incorporate density of molecules (number of absorbing particles / cm³)
    # mole_fraction is the weight of the abscoeff of this molecule
    density = mole_fraction * ((pressure_mbar * 100) / (k_b * Tgas)) * 1e-6
    # n * S * F, unit: cm-1
    abscoeff = density * sumoflines
    # print("done!")

    return abscoeff, wavenumber, df, perf_dict


def print_perf(perf_dict_list, total_time):
    TAB = 4

    perf_sum_dict = {}
    for key, value in perf_dict_list[0].items():
        perf_sum_dict[key] = 0

    for key, value in perf_sum_dict.items():
        for perf_dict in perf_dict_list:
            perf_sum_dict[key] += perf_dict[key]

    print("Calculation time of the absorption coefficient (non optimization): %.5f s. \n" % total_time)
    for key, value in perf_sum_dict.items():
        print(" " * TAB + key + ": %.5f s. \n" % value)


def plot_absorb_coeff(x, y, x_label, y_label, title, smooth_window_size=5):
    sns.set_theme()
    smooth_window_size = np.ones(smooth_window_size) / float(smooth_window_size)
    smooth_y = np.convolve(y, smooth_window_size, 'same')

    # plot the original data and the smoothed data
    fig, ax = plt.subplots()

    # set the plot title and axis labels
    ax.plot(x, smooth_y)
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    # display the legend
    ax.legend()

    plt.show()


