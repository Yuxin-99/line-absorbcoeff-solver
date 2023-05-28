import drjit as dr
import numpy as np
import time

from basics import get_molecule_identifier, Tref, pressure_atm, generate_wavenumber_range, get_molar_mass
from broadening import calc_lorentz_hwhm, calc_gauss_hwhm, calc_lineshape_LDM, apply_lineshpe_LDM, calc_continuum_absorb
from constants import c_2, k_b
from loader import load_hitran


def calc_lineshift_opt(df, pressure_mbar):
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

    # Calculate
    air_pressure = pressure_mbar / 1013.25  # convert from mbar to atm

    start = time.time()
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
def calc_linestrength_opt(df_cols, Tgas, molecule):
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
            df.int * Qref_Qgas_ratio_opt(df, Tgas, molecule)
            * (np.exp(-c_2 * df.El * (1 / Tgas - 1 / Tref)))
            * ((1 - np.exp(-c_2 * df.wav / Tgas)) / (1 - np.exp(-c_2 * df.wav / Tref)))
    )
    elapsed = time.time() - start
    return elapsed


def Qref_Qgas_ratio_opt(df, Tgas, molecule):
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
            Qref_Qgas_ratio = dr.zeros(len(df_id))
            mask1 = dr.eq(df_id, molecule_id)
            for iso in iso_set:
                mask2 = dr.eq(df_iso, iso)
                Q_ref = partitionSum(molecule_id, iso, Tref)
                Q_gas = partitionSum(molecule_id, iso, Tgas)
                dr.select(mask1&mask2, Q_ref / Q_gas, Qref_Qgas_ratio)
                Qref_Qgas_ratio[iso] = Q_ref / Q_gas

            Qref_Qgas = df["iso"].map(Qref_Qgas_ratio)

    else:
        # if single isotope
        iso = df.attrs["iso"]
        Q_ref = partitionSum(molecule_id, iso, Tref)
        Q_gas = partitionSum(molecule_id, iso, Tgas)
        df.attrs["Qgas"] = Q_gas
        df.attrs["Qref"] = Q_ref
        Qref_Qgas = Q_ref / Q_gas
    return Qref_Qgas


def get_spectrum_opt(
        wmin,
        wmax,
        Tgas,
        molecule,
        isotope="all",
        mole_fraction=1.0,
        diluent="air",
        pressure=pressure_atm,
        wstep=0.01,
        # optimization="simple",
        # broadening_method="fft"
):
    """
    :param wmin: the minimum wavenumber, unit: cm-1
    :param wmax: the maximum wavenumber, unit: cm-1
    :param Tgas: the temperature, unit: K
    :param molecule: the queried molecule(s)
    :param isotope: the queried isotope(s)
    :param mole_fraction: the fraction of molecule(s)
    :param diluent: the dieluent, default (and for now only as) air
    :param pressure: the total gas pressure, unit: bar

    """
    pressure_mbar = pressure * 1e3

    # convert the molecule input as a list
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
        assert (type(isotope) == str)
        for mol in molecule:
            molecule_dict[mol]["iso"] = isotope  # every molecule has the same isotope

    # set the correct diluent
    # mole_fraction is dictionary of multiple molecules and have non air diluent -> NotImplementedError
    if diluent != "air":
        raise NotImplementedError("Currently only support the case that there is only air as diluent!")

    if len(molecule) != 1 and type(mole_fraction) != dict:
        raise Exception("When given multiple molecules, please provide the exact mol_fraction of each molecule!!!")
    elif len(molecule) == 1 and (type(mole_fraction) == float or type(mole_fraction) == int):
        molecule_dict[molecule[0]]["mol_fraction"] = mole_fraction
        diluent_fraction = 1.0 - mole_fraction
        diluent = {"air": diluent_fraction}
    else:
        mol_frac_sum = 0.0
        for mol in molecule:
            if mol not in mole_fraction.keys():
                raise Exception("Error: Missing mole fraction input for the molecule " + mol + "!!!")
            else:
                molecule_dict[mol]["mol_fraction"] = mole_fraction[mol]
                mol_frac_sum += mole_fraction[mol]
        diluent_fraction = 1.0 - mol_frac_sum
        diluent = {"air": diluent_fraction}

    # download data, and put all molecules & isotopes into one big table
    df_list = []
    for mol in molecule_dict.keys():
        df = load_hitran(
            wavenum_min=wmin,
            wavenum_max=wmax,
            molecule=mol,
            isotope=molecule_dict[mol]["iso"],
            drop_column=False
        )
        df_list.append(df)

    column_dict = {}
    # set the column we will need for absorbcoeff computation
    columns_to_extract = ['id', 'iso', 'EL', 'int', 'wav', 'Tdpair', 'airbrd', 'Tdpsel']

    for df in df_list:
        for column_name in columns_to_extract:
            if column_name not in list(df.keys()):
                continue
            # Extract the desired column values from each data frame
            column_values = df[column_name].tolist()

            # Create a DrJIT vector for the column values and store it in the dictionary
            if column_name not in column_dict:
                column_dict[column_name] = dr.Vector(column_values)
            else:
                # If the column already exists in the dictionary, append the values
                column_dict[column_name].extend(column_values)

    # compute line strength

    # loop through all the molecules
    abscoeff_res = 0.0
    for mol in molecule_dict.keys():
        abscoeff, wave_number, df, perf_dict = calc_molecule_eq_spectrum_opt(
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

    # if "H2O" in molecule_dict.keys():
    #     abscoeff_res += calc_continuum_absorb(wave_number, mole_fraction["H2O"], pressure)

    print_perf_opt(perf_dict)

    # the df here is just the dataframe of the last molecule, might not be used for testing when there are multiple molecules
    return abscoeff_res, wave_number, df


def calc_molecule_eq_spectrum_opt(
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
    # df = load_df(file, wavenum_min, wavenum_max, isotope, True, True)
    df = load_hitran(
        wavenum_min=wavenum_min,
        wavenum_max=wavenum_max,
        molecule=molecule,
        isotope=isotope,
    )

    perf_dict = {}

    # df['S'], unit: cm−1/(molecule·cm−2)
    S_time = calc_linestrength_opt(df, Tgas, molecule)
    perf_dict["Line strength"] = S_time
    shift_time = calc_lineshift_opt(df, pressure_mbar)
    perf_dict["Line Shift"] = shift_time

    # TODO: OPT version
    molar_mass = get_molar_mass(df)  # g/mol
    # TODO: opt version; mole_fraction
    lorentz_hwhm_time = calc_lorentz_hwhm(df, Tgas, diluent, mole_fraction)
    perf_dict["Lorentz hwhm"] = lorentz_hwhm_time
    gauss_hwhm_time = calc_gauss_hwhm(df, Tgas, molar_mass)
    perf_dict["Gaussian hwhm"] = gauss_hwhm_time

    # calculate the broadening (line shapes)
    wavenumber, wavenumber_calc, woutrange = generate_wavenumber_range(wavenum_min, wavenum_max, wstep)
    line_profile_LDM, wL, wG, wL_dat, wG_dat, calc_ldm_time = calc_lineshape_LDM(df, wstep, wavenumber_calc)
    perf_dict["Calculate line profile LDM"] = calc_ldm_time
    # commpute S * F, unit of the convolution result: 1/(molecule·cm−2)
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
    perf_dict["total"] = 0.0
    for key in perf_dict:
        perf_dict["total"] += perf_dict[key]

    # incorporate density of molecules (number of absorbing particles / cm³)
    # mole_fraction is the weight of the abscoeff of this molecule
    density = mole_fraction * ((pressure_mbar * 100) / (k_b * Tgas)) * 1e-6
    # n * S * F, unit: cm-1
    abscoeff = density * sumoflines
    print("done!")

    return abscoeff, wavenumber, df, perf_dict


def print_perf_opt(perf_dict_list):
    TAB = 4

    perf_sum_dict = {}
    for key, value in perf_dict_list[0].items():
        perf_sum_dict[key] = 0

    for key, value in perf_sum_dict.items():
        for perf_dict in perf_dict_list:
            perf_sum_dict[key] += perf_dict[key]

    print("Calculation time of the absorption coefficient: %.5f s. \n" % perf_sum_dict["total"])
    for key, value in perf_sum_dict.items():
        if key != "total":
            print(" " * TAB + key + ": %.5f s. \n" % value)

