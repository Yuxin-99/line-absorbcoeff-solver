import numpy as np
import seaborn as sns

from matplotlib import pyplot as plt
from radis import calc_spectrum

from atmosphere import get_atm_gas_data
from main import get_absorbcoeff

epsilon = 5e-3  # the tolerant difference of the computed parameters between the RADIS library and our code


def test_opt_O2_750_900_T1000():
    print(
        "Testing the computation of Molecule: C02; isotope: 1; wavenum_min = 2380; wavenum_max = 4200; Tgas = 1000"
    )
    # Tref = 296  # unit: K
    pressure = 1.01325 * 1e5    # unit: pa
    Tgas = 1000  # unit: K

    molecules = "O2"
    isotope = "1"
    wavelength_min = 750
    wavelength_max = 900
    mole_fraction = 0.2
    # diluent = {'air': 0}
    wstep = 0.01

    run_test(wavelength_min, wavelength_max, Tgas, pressure, molecules, isotope, mole_fraction, wstep, 0.3)


def test_opt_COALL_1900_2300_T1000():
    print(
        "Testing the computation of Molecule: C02; isotope: 1; wavenum_min = 2380; wavenum_max = 4200; Tgas = 1000"
    )
    pressure = 1.01325 * 1e5
    Tgas = 1000  # unit: K

    molecule = "CO"
    isotope = "1, 2, 3"
    wavelength_min = 4347
    wavelength_max = 5263
    mole_fraction = 0.1
    # diluent = {'air': 0.9}
    wstep = 0.01

    run_test(wavelength_min, wavelength_max, Tgas, pressure, molecule, isotope, mole_fraction, wstep, 0.3)


def test_opt_multi_mols_1900_2300():
    print(
        "Testing the computation of Molecule: C02; isotope: 1; wavenum_min = 2380; wavenum_max = 4200; Tgas = 1000"
    )
    pressure = 1.01325 * 1e5
    Tgas = 700  # unit: K

    molecule = ["CO2", "CO"]
    isotope = {"CO2": "1", "CO": "3"}
    wavelength_min = 4166
    wavelength_max = 5263
    mole_fraction = {"CO2": 0.1, "CO": 0.2}
    # diluent = {'air': 0.9}
    wstep = 0.01

    run_test(wavelength_min, wavelength_max, Tgas, pressure, molecule, isotope, mole_fraction, wstep, 0.3)


def run_test(wavelength_min, wavelength_max, Tgas, pressure, molecule, isotope, mole_fraction, wstep, path_length):
    """
        :param wavelength_min: nm
        :param wavelength_max: nm
        :param Tgas: K
        :param pressure: pa
        :param molecule: str or list
        :param isotope: str or dict
        :param mole_fraction: float or dict
        :param wstep: float
        :param path_length: float
    """
    # create validation values
    pressure_bar = pressure * 1e-5
    s, sf = calc_spectrum(wavelength_min=wavelength_min,
                          wavelength_max=wavelength_max,
                          Tgas=Tgas,
                          pressure=pressure_bar,
                          molecule=molecule,
                          isotope=isotope,
                          mole_fraction=mole_fraction,
                          path_length=path_length,
                          wstep=wstep,
                          cutoff=0,
                          optimization="simple",
                          broadening_method="fft",
                          return_factory=True,
                          verbose=False)

    muti_mols = (type(molecule) == list)
    print("test begins!!!!")

    abscoeff, wavenumber, df = get_absorbcoeff(
        wavelen_min=wavelength_min,
        wavelen_max=wavelength_max,
        Tgas=Tgas,
        molecule=molecule,
        isotope=isotope,
        mol_frac=mole_fraction,
        pressure=pressure,
        wstep=wstep
    )

    # if it is the case of multiple molecules, do not compare the separate parameters (S, hwhm)
    # check the line strength S
    if not muti_mols:
        sf_df = sf.df1
        # make sure the wave numbers are the same
        wav_test = df['wav']
        wav_validation = sf_df.wav
        assert (len(wav_validation) == len(wav_test), "The length of the two parameters are not equal!")
        for i in range(len(df)):
            assert (
                np.absolute(wav_test[i] - wav_validation[i]) <= epsilon, "Errors in wavenumber! Not equal to RADIS!")

        S_validation = sf_df['S']
        S_test = df['S']
        compare_variable(S_validation, S_test, wav_validation, wav_test, "Line Strength", "cm−1/(molecule·cm−2)")

        # check the Lorentz hwhm
        lorentz_hwhm_validation = sf_df['hwhm_lorentz']
        lorentz_hwhm_test = df['hwhm_lorentz']
        compare_variable(lorentz_hwhm_validation, lorentz_hwhm_test, wav_validation, wav_test, "Lorentz HWHM",
                         "cm−1/atm")

        # check the Gauss hwhm
        gauss_hwhm_validation = sf_df['hwhm_gauss']
        gauss_hwhm_test = df['hwhm_gauss']
        compare_variable(gauss_hwhm_validation, gauss_hwhm_test, wav_validation, wav_test, "Gaussian HWHM", "cm−1/atm")

    # check the absorption coefficient
    wavespace_valid = s.get_wavenumber()
    wavespace_test = wavenumber
    assert (len(wavespace_valid) == len(wavespace_test), "The length of the wavespace are not equal!")
    for i in range(len(wavespace_valid)):
        assert (
            np.absolute(wavespace_valid[i] - wavespace_test[i]) <= epsilon, "Errors in wavespace! Not equal to RADIS!")
    abscoeff_validation = s.get('abscoeff', 'cm-1')[1]
    abscoeff_test = abscoeff
    compare_variable(abscoeff_validation, abscoeff_test, wavespace_valid, wavespace_test, "Absorption Coefficient",
                     "cm-1")


def compare_variable(val_validation, val_test, sf_wav, wav, val_name, val_unit):
    print("Validate the accuracy of " + val_name)

    val_test = np.array(val_test)
    wav = np.array(wav)
    # compare in numbers
    assert (len(val_test) == len(val_validation), "The length of the two parameters are not equal!")
    for i in range(len(val_test)):
        assert (
            np.absolute(val_test[i] - val_validation[i]) <= epsilon, "Errors in " + val_name + "! Not equal to RADIS!")

    # compare in plots: x axis as wav; S as y axis
    comparison_bar_plots(sf_wav, wav, val_validation, val_test, val_name, val_unit)


def comparison_bar_plots(wav_radis, wav_test, y_radis, y_test, y_label, y_unit):
    sns.set_theme(style="whitegrid")
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))  # row 1, col 2, index 1
    # draw the relative (error) betwee y_radis and y_test
    relative_error = np.abs((y_test - y_radis) / y_radis)

    axs[0].bar(wav_radis, relative_error, width=0.01)
    axs[0].set_title("Relative error of " + y_label)
    axs[0].set_xlabel('wavenumber (cm-1)')
    axs[0].set_ylabel("rel. err of " + y_label + ' [' + y_unit + ']')

    # draw the plots of wav and y_radis/y_test overlapped
    # first smooth the y value
    smooth_window_size = 10
    smooth_window_size = np.ones(smooth_window_size) / float(smooth_window_size)
    smooth_y_radis = np.convolve(y_radis, smooth_window_size, 'same')
    smooth_y_test = np.convolve(y_test, smooth_window_size, 'same')

    radis_color = np.array([252, 207, 3]) / 255
    test_color = np.array([3, 161, 252]) / 255
    axs[1].plot(wav_radis, smooth_y_radis, color=tuple(radis_color), linewidth=3, label="radis")
    axs[1].plot(wav_test, smooth_y_test, color=tuple(test_color), linestyle='dashed', linewidth=1.5, label="test")
    axs[1].set_title("Comparison of " + y_label + " to radis")
    axs[1].set_xlabel('wavenumber (cm-1)')
    axs[1].set_ylabel(y_label + ' [' + y_unit + ']')
    axs[1].legend()
    plt.show()
