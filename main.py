from eq_spectrum import get_spectrum
from accelerate_opt import get_spectrum_opt

from radis import calc_spectrum


def main():
    # s1 = calc_spectrum(1900, 2300,  # cm-1
    #                   molecule='CO',
    #                   isotope='1,2,3',
    #                   pressure=1.01325,  # bar
    #                   Tgas=700,  # K
    #                   mole_fraction=0.1,
    #                   path_length=1,  # cm
    #                   databank='hitran',  # or 'hitemp', 'geisa', 'exomol'
    #                   )
    # s1.apply_slit(0.5, 'nm')  # simulate an experimental slit
    # s1.print_perf_profile()
    # print("################# Multiple isotopes! #################")

    # s2 = calc_spectrum(1900, 2300,  # cm-1
    #                    molecule='CO',
    #                    isotope='1',
    #                    pressure=1.01325,  # bar
    #                    Tgas=700,  # K
    #                    mole_fraction=0.1,
    #                    path_length=1,  # cm
    #                    databank='hitran',  # or 'hitemp', 'geisa', 'exomol'
    #                    )
    # s2.apply_slit(0.5, 'nm')  # simulate an experimental slit
    # s2.print_perf_profile()
    # print("################# Single isotope! #################")
    #

    mole_fraction = {"CO2": 0.1, "CO": 0.2}
    s3 = calc_spectrum(
        wavelength_min=4165,
        wavelength_max=5000,
        Tgas=700,
        path_length=0.1,
        molecule=["CO2", "CO"],
        mole_fraction=mole_fraction,
        isotope={"CO2": "1,2", "CO": "1,2,3"},
    )
    # wavenum_min=1999.4547859048316
    # wavenum_max=2400.3058004755053
    # s3.print_perf_profile()       # the profiler of spectrum of multiple molecules is NONE
    print("################# Multiple molecules! #################")

    # TODO: larger data, more molecules/isotopes, 考虑可以并行的地方
    # TODO: real example

    Tgas = 700  # unit: K

    molecule = ["CO2", "CO"]
    isotope = {"CO2": "1,2", "CO": "1,2,3"}
    wavelen_min = 4165
    wavelen_max = 5000
    pressure = 76
    get_absorbcoeff(wavelen_min, wavelen_max, Tgas, pressure, molecule, isotope, mole_fraction, opt=False)


def get_absorbcoeff(wavelen_min, wavelen_max, Tgas, pressure, molecule, isotope, mol_frac, opt=True):
    # convert wavelength to wavenumber
    wavenum_max = 1 / (wavelen_min * 1e-7)  # cm-1
    wavenum_min = 1 / (wavelen_max * 1e-7)

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

    # opt: flag to indicate whether to use mitsuba to accelerate
    if opt:
        absorb_coeff, wavenum, _ = get_spectrum_opt(wavenum_min, wavenum_max, Tgas, molecule, isotope=isotope,
                                                    mole_fraction=mol_frac, pressure=pressure)
    else:
        absorb_coeff, wavenum, _ = get_spectrum(wavenum_min, wavenum_max, Tgas, molecule, isotope=isotope,
                                                mole_fraction=mol_frac, pressure=pressure)


if __name__ == '__main__':
    main()
