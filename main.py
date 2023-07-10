import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from radis import calc_spectrum

from atmosphere import get_atm_gas_data
from eq_spectrum import get_spectrum, plot_absorb_coeff
from eq_spectrum_drjit_opt import get_spectrum_drjit_opt
from eq_spectrum_np_opt import get_spectrum_np_opt

from loader import nm_air2cm


def main():
    altitude = 1
    pressure, Tgas, molecules, mole_fraction = get_atm_gas_data(50)
    # pressure2, Tgas2, molecules, mole_fraction2 = get_atm_gas_data(60)
    #
    # mol = "H2O"
    # isotope = "all"
    # wavelen_min = 4160  # unit: nm
    # wavelen_max = 4200
    # wstep = 0.01

    #
    # molecules = ["CO2", "CO"]
    # isotope = {"CO2": "1,2", "CO": "1,2,3"}
    # wavelen_min = 4165
    # wavelen_max = 5000
    # pressure = 101325
    # absorbcoeff_1, wavenumber, _ = get_absorbcoeff(wavelen_min, wavelen_max, Tgas, pressure, "N2O", isotope,
    #                                                0.2, wstep, opt=False)
    # absorbcoeff_2, wavenumber, _ = get_absorbcoeff(wavelen_min, wavelen_max, Tgas, pressure, "O3", isotope,
    #                                                0.2, wstep, opt=False)
    # plot_result(wavenumber, absorbcoeff_1, absorbcoeff_2,
    #             "wavenumber (cm-1)", "AbsorbCoefficient (cm-1)", "Comparison between N2O and O3 (T=500K, P=101.325kPa)",
    #             smooth_window_size=8)

    # absorbcoeff, wavenumber, _ = get_absorbcoeff(wavelen_min, wavelen_max, Tgas, pressure, mol, isotope, mole_fraction[mol], wstep, opt=False)
    # plot_result(wavenumber, absorbcoeff, "wavenumber (cm-1)", "AbsorbCoefficient (cm-1)", "Molecules: " + mol
    #                   + f" Temperature: {Tgas: .2f} K Pressure: {pressure: .2f} Pa (at Altitude {altitude: .1f} km)", smooth_window_size=50)

    pressure = 1.01325
    Tgas = 500  # unit: K

    molecule = "CO2"
    isotope = "all"
    wavenum_min = 2380
    wavenum_max = 2400
    mole_fraction = 0.2
    # diluent = {'air': 0.9}
    wstep = 0.02

    s, sf = calc_spectrum(wavenum_min, wavenum_max,
                          Tgas=Tgas,
                          pressure=pressure,
                          molecule=molecule,
                          isotope=isotope,
                          mole_fraction=mole_fraction,
                          path_length=0.3,
                          wstep=wstep,
                          cutoff=0,
                          optimization="simple",
                          broadening_method="convolve",
                          return_factory=True,
                          verbose=False)

    sf.plot_broadening(400)


def get_absorbcoeff(wavelen_min, wavelen_max, Tgas, pressure, molecule, isotope, mol_frac, wstep, opt=True):
    # convert wavelength to wavenumber
    wavenum_min = nm_air2cm(wavelen_max)
    wavenum_max = nm_air2cm(wavelen_min)

    pressure = pressure * 1e-5  # unit: bar

    # convert the molecule input as a list
    if type(molecule) != list:
        molecule = [molecule]

    mol_str = ""
    for mol in molecule:
        mol_str = mol_str + mol + "&"
    mol_str = mol_str[:-1]
    print("Molecule: " + mol_str + "; wavelength: %.1f ~ %.1f; Tgas: %.1f K; pressure: %.1f Pa"
          % (wavelen_min, wavelen_max, Tgas, pressure))

    # drjit: flag to indicate whether to use drjit to accelerate
    if opt:
        return get_spectrum_np_opt(wavenum_min, wavenum_max, Tgas, molecule,
                                   isotope=isotope,
                                   mole_fraction=mol_frac, pressure=pressure, wstep=wstep)

    else:
        return get_spectrum(wavenum_min, wavenum_max, Tgas, molecule, isotope=isotope,
                            mole_fraction=mol_frac, pressure=pressure, wstep=wstep)


def plot_result(x, y1, y2, x_label, y_label, title, smooth_window_size=5):
    sns.set_theme()
    smooth_window_size = np.ones(smooth_window_size) / float(smooth_window_size)
    smooth_y1 = np.convolve(y1, smooth_window_size, 'same')
    if y2 is not None:
        smooth_y2 = np.convolve(y2, smooth_window_size, 'same')
    # smooth_y3 = np.convolve(y3, smooth_window_size, 'same')

    # plot the original data and the smoothed data
    fig, ax = plt.subplots()

    y1_color = np.array([252, 147, 66]) / 255  # orange
    y2_color = np.array([97, 171, 108]) / 255  # green
    y3_color = np.array([43, 97, 148]) / 255  # blue
    red_color = np.array([209, 71, 69]) / 255
    y1_label = "N2O"
    y2_label = "O3"
    y3_label = "isotope = '1'"
    # set the plot title and axis labels
    ax.plot(x, smooth_y1, color=tuple(y1_color), linewidth=2, label=y1_label)
    if y2 is not None:
        ax.plot(x, smooth_y2, color=tuple(y3_color), linewidth=2, label=y2_label)
    # ax.plot(x, smooth_y3, color=tuple(y3_color), linestyle='dashed', linewidth=1, label=y3_label)

    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    # display the legend
    ax.legend()

    plt.show()


if __name__ == '__main__':
    main()
