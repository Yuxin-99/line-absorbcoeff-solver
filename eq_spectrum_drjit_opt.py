import drjit as dr
import mitsuba as mi
import numpy as np
import time

from basics import get_molecule_identifier, Tref, generate_wavenumber_range, dr_get_molar_mass
from broadening_drjit_opt import calc_lorentz_hwhm_opt, calc_gauss_hwhm_opt, calc_lineshape_LDM_opt, apply_lineshape_LDM_opt
from constants import c_2, k_b
from loader import load_hitran

mi.set_variant('llvm_ad_rgb')


def calc_lineshift_opt(dfcolumn_dict, pressure_mbar):
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

    # Calculate
    air_pressure = pressure_mbar / 1013.25  # convert from mbar to atm
    if "Pshft" in dfcolumn_dict.keys():
        shiftwav = dfcolumn_dict["wav"] + (dfcolumn_dict["Pshft"] * air_pressure)
    else:
        print(
            "Pressure-shift coefficient not given in database: assumed 0 pressure shift"
        )
        shiftwav = dfcolumn_dict["wav"]

    dfcolumn_dict["shiftwav"] = shiftwav
    # Sorted lines is needed for sparse wavenumber range algorithm.
    # TODO: sort in drjit?
    sort_start = time.time()
    sorted_indices = mi.UInt(np.argsort(shiftwav))
    sort_time = time.time() - sort_start
    print("Time spent on sorting the data after computing shift_wavs: %.5f s. \n" % sort_time)
    # df.sort_values("shiftwav", inplace=True)
    for key in dfcolumn_dict.keys():
        if key == "id" or key == "iso":
            dfcolumn_dict[key] = dr.gather(mi.UInt, dfcolumn_dict[key], sorted_indices)
        else:
            dfcolumn_dict[key] = dr.gather(mi.Float64, dfcolumn_dict[key], sorted_indices)

    elapsed = time.time() - start
    return dfcolumn_dict, elapsed


# compute S
def calc_linestrength_opt(dfcolumn_dict, Tgas, molecule, isotope, num_of_lines):
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
    df_int = dfcolumn_dict["int"]
    El = dfcolumn_dict["El"]
    wav = dfcolumn_dict["wav"]
    s = (
            df_int * Qref_Qgas_ratio_opt(dfcolumn_dict, Tgas, molecule, isotope, num_of_lines)
            * (dr.exp(-c_2 * El * (1 / Tgas - 1 / Tref)))
            * ((1 - dr.exp(-c_2 * wav / Tgas)) / (1 - dr.exp(-c_2 * wav / Tref)))
    )
    elapsed = time.time() - start
    return s, elapsed


def Qref_Qgas_ratio_opt(dfcolumn_dict, Tgas, molecule, isotope, num_of_lines):
    """Calculate Qref/Qgas at temperature ``Tgas``, ``Tref``, for all lines.
    Returns a single value if all lines have the same Qref/Qgas ratio,
    or a column if they are different.

    molecule: molecule set of all molecule ids
    isotope: iso set of all isotopes

    See Also
    --------
    :py:meth:`~radis.lbl.base.BaseFactory.Qgas`
    """

    # if "molecule" in df.attrs:
    #     molecule = df.attrs["molecule"]  # used for ExoMol, which has no HITRAN-id
    # else:
    #     molecule = get_molecule(df1.attrs["id"])

    # molecule_id = get_molecule_identifier(molecule)
    from hapi import partitionSum

    # case1: only one molecule
    if len(molecule) == 1:
        molecule_id = molecule[0]
        if len(isotope) == 1:
            Q_ref = partitionSum(molecule_id, isotope[0], Tref)
            Q_gas = partitionSum(molecule_id, isotope[0], Tgas)
            Qref_Qgas_ratio = Q_ref / Q_gas
        else:
            Qref_Qgas_ratio = dr.ones(mi.Float64, num_of_lines)
            for iso in isotope:
                mask = dr.eq(dfcolumn_dict["iso"], iso)
                Q_ref = partitionSum(molecule_id, iso, Tref)
                Q_gas = partitionSum(molecule_id, iso, Tgas)
                Qref_Qgas_ratio = dr.select(mask, Q_ref / Q_gas, Qref_Qgas_ratio)
        return Qref_Qgas_ratio

    # case 2: multiple molecules
    Qref_Qgas_ratio = dr.ones(mi.Float64, num_of_lines)
    for molecule_id in molecule:
        mask_mol = dr.eq(dfcolumn_dict["id"], molecule_id)
        for iso in isotope:
            mask_iso = dr.eq(dfcolumn_dict["iso"], iso)
            mask = mask_mol & mask_iso
            if mask == False:
                # there is no line data for this combination (molecule_id & iso)
                continue
            Q_ref = partitionSum(molecule_id, iso, Tref)
            Q_gas = partitionSum(molecule_id, iso, Tgas)
            Qref_Qgas_ratio = dr.select(mask, Q_ref/Q_gas, Qref_Qgas_ratio)
    return Qref_Qgas_ratio


def get_spectrum_drjit_opt(
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
    :param molecule: the queried molecule(s)
    :param isotope: the queried isotope(s)
    :param mole_fraction: the fraction of molecule(s)
    :param diluent: the dieluent, default (and for now only as) air
    :param pressure: the total gas pressure, unit: bar
    :param wstep: parameter for constructing the ldm grid

    """
    start_time = time.time()
    pressure_mbar = pressure * 1e3

    ''' convert the molecule input as a list'''
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

    '''set the correct diluent'''
    # mole_fraction is dictionary of multiple molecules and have non air diluent -> NotImplementedError
    if diluent != "air":
        raise NotImplementedError("Currently only support the case that there is only air as diluent!")
    diluent = [diluent]

    if len(molecule) != 1 and type(mole_fraction) != dict:
        raise Exception("When given multiple molecules, please provide the exact mol_fraction of each molecule!!!")
    elif len(molecule) == 1 and (isinstance(mole_fraction, int) or isinstance(mole_fraction, float)):
        molecule_dict[molecule[0]]["mol_fraction"] = mole_fraction
    else:
        for mol in molecule:
            if mol not in mole_fraction.keys():
                raise Exception("Error: Missing mole fraction input for the molecule " + mol + "!!!")
            else:
                molecule_dict[mol]["mol_fraction"] = mole_fraction[mol]

    '''download data, and put all molecules & isotopes into one big table (dfcolumn_dict)'''
    df_list = []
    diluent_dict = {}       # key: mol_id; val: 1 - mole fraction of mol_id
    num_of_lines = 0
    for mol in molecule_dict.keys():
        mol_id = get_molecule_identifier(mol)
        molecule_dict[mol]["id"] = mol_id
        diluent_dict[mol_id] = 1.0 - molecule_dict[mol]["mol_fraction"]
        df = load_hitran(
            wavenum_min=wmin,
            wavenum_max=wmax,
            molecule=mol,
            isotope=molecule_dict[mol]["iso"],
            drop_column=False
        )
        # Remark: if there is no data for this molecule (len(df) == 0): no data will be added to dfcolumn_dict
        if len(df) == 0:
            continue
        df_list.append(df)
        num_of_lines += len(df)

    dfcolumn_dict = {}
    # set the column we will need for absorbcoeff computation
    columns_to_extract = ['id', 'iso', 'El', 'int', 'wav', 'Pshft', 'Tdpair', 'Tdpsel', 'airbrd', 'selbrd']
    # concatenate all rows from dataframes and put each column as each (key, val) in dfcolumn_dict
    dfcolumn_dict["diluent_frac"] = []
    for df in df_list:
        mol_id = df.id.unique()
        assert (len(mol_id == 1), "Each dataframe should only contain data of one uniqe molecule!")
        mol_id = int(mol_id)
        diluent_frac = diluent_dict[mol_id]
        diluent_frac_list = [diluent_frac] * len(df)
        dfcolumn_dict["diluent_frac"].extend(diluent_frac_list)
        for column_name in columns_to_extract:
            if column_name not in list(df.keys()):
                continue
            # Extract the desired column values from each data frame
            column_values = df[column_name].tolist()

            # Create a DrJIT vector for the column values and store it in the dictionary
            if column_name not in dfcolumn_dict:
                dfcolumn_dict[column_name] = column_values
            else:
                # If the column already exists in the dictionary, append the values
                dfcolumn_dict[column_name].extend(column_values)
    # TODO: maybe should check that lists in all values have the same length
    # TODO: and maybe should check Assert that the mandatory columns are all converted! like int, wav, El...

    mole_set = list(set(dfcolumn_dict["id"]))  # all the unique molecule ids
    iso_set = list(set(dfcolumn_dict["iso"]))  # all the unique isotope ids

    # convert each list in the dict to drjit vector
    for column_name in dfcolumn_dict.keys():
        if column_name == "id" or column_name == "iso":
            dfcolumn_dict[column_name] = mi.UInt(dfcolumn_dict[column_name])
        else:
            dfcolumn_dict[column_name] = mi.Float64(dfcolumn_dict[column_name])

    perf_dict = {}

    '''compute line strength'''
    # df['S'], unit: cm−1/(molecule·cm−2)
    s, S_time = calc_linestrength_opt(dfcolumn_dict, Tgas, mole_set, iso_set, num_of_lines)
    dfcolumn_dict['S'] = s
    perf_dict["Line strength"] = S_time
    dfcolumn_dict, shift_time = calc_lineshift_opt(dfcolumn_dict, pressure_mbar)
    perf_dict["Line Shift"] = shift_time

    '''compute broadening related parameters'''
    molar_mass = dr_get_molar_mass(dfcolumn_dict, mole_set, iso_set, num_of_lines)  # g/mol
    # map the mole_fraction to each line if multiple molecules
    mol_fracs = mole_fraction
    if len(molecule) != 1:
        mol_fracs = dr.ones(mi.Float64, num_of_lines)
        for mol in mole_fraction.keys():
            molecule_id = molecule_dict[mol]["id"]
            mask = dr.eq(dfcolumn_dict["id"], molecule_id)
            frac = mole_fraction[mol]
            mol_fracs = dr.select(mask, frac, mol_fracs)

    pressure_atm = pressure_mbar / 1013.25
    gamma_lb, lorentz_hwhm_time = calc_lorentz_hwhm_opt(dfcolumn_dict, Tgas, diluent, mol_fracs, pressure_atm)
    dfcolumn_dict['hwhm_lorentz'] = gamma_lb
    perf_dict["Lorentz hwhm"] = lorentz_hwhm_time
    gamma_doppler, gauss_hwhm_time = calc_gauss_hwhm_opt(dfcolumn_dict, Tgas, molar_mass)
    dfcolumn_dict['hwhm_gauss'] = gamma_doppler
    perf_dict["Gaussian hwhm"] = gauss_hwhm_time

    '''compute and apply broadening (line shapes)'''
    wavenumber, wavenumber_calc, woutrange = generate_wavenumber_range(wmin, wmax, wstep)
    perf_dict["Calculate line profile LDM"] = 0
    perf_dict["Apply LDM"] = 0

    # Remark: as the LDM of each molecule should be computed separately, I need to separate the huge data dictionary
    # into a smaller dictionary of each molecule respectively
    mol_data_dict = {}      # key as molecule_id, value as the corresponding columns dictionary of this molecule
    for mol_id in mole_set:
        mol_data = {}   # contains all column data of mol
        # TODO: filter rows belonging to mol_id in drjit version??
        mol_indices = mi.UInt([index for index, element in enumerate(dfcolumn_dict["id"]) if element == mol_id])
        for column_name in dfcolumn_dict.keys():
            if column_name == "id" or column_name == "iso":
                mol_data[column_name] = dr.gather(mi.UInt, dfcolumn_dict[column_name], mol_indices)
            else:
                mol_data[column_name] = dr.gather(mi.Float64, dfcolumn_dict[column_name], mol_indices)
        mol_data_dict[mol_id] = mol_data

    abscoeff_res = 0.0
    w_lineshape_ft = np.fft.rfftfreq(2 * len(wavenumber_calc), wstep)
    # loop through all the molecules to add the broadening effect to absorbcoeff
    for mol in molecule_dict.keys():
        mol_id = molecule_dict[mol]["id"]
        # Remark: skip the molecule that doesn't have data (no lines in this wavelength range)
        if mol_id not in mol_data_dict.keys():
            continue
        abscoeff, calc_ldm_time, apply_time = calc_molecule_broadening_opt(
            mol_data_dict[mol_id],
            Tgas,
            pressure_mbar,
            molecule_dict[mol]["mol_fraction"],
            wavenumber,
            wavenumber_calc,
            woutrange,
            mi.Float64(w_lineshape_ft),
            wstep
        )
        perf_dict["Calculate line profile LDM"] += calc_ldm_time
        perf_dict["Apply LDM"] += apply_time
        # ARTS: As absorption is additive, the total absorption coefficient is derived by adding up the
        # absorption contributions of all spectral lines of all molecular species.
        abscoeff_res += abscoeff

    total_time = time.time() - start_time
    print_perf_opt(perf_dict, total_time)

    # the df here is just the dataframe of the last molecule, might not be used for testing when there are multiple molecules
    return abscoeff_res, wavenumber, dfcolumn_dict


def calc_molecule_broadening_opt(
        df,
        Tgas,
        pressure_mbar,
        mole_fraction,
        wavenumber,
        wavenumber_calc,
        woutrange,
        w_lineshape_ft,
        wstep
):
    line_profile_LDM, wL, wG, wL_dat, wG_dat, calc_ldm_time = calc_lineshape_LDM_opt(df, w_lineshape_ft)
    # compute S * F, unit of the convolution result: 1/(molecule·cm−2)
    sumoflines, apply_time = apply_lineshape_LDM_opt(
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

    # incorporate density of molecules (number of absorbing particles / cm³)
    # mole_fraction is the weight of the abscoeff of this molecule
    density = mole_fraction * ((pressure_mbar * 100) / (k_b * Tgas)) * 1e-6
    # n * S * F, unit: cm-1
    abscoeff = density * sumoflines

    return abscoeff, calc_ldm_time, apply_time


def print_perf_opt(perf_dict, total_time):
    TAB = 4

    print("Calculation time of the absorption coefficient (with optimization): %.5f s. \n" % total_time)
    for key, value in perf_dict.items():
        print(" " * TAB + key + ": %.5f s. \n" % value)
