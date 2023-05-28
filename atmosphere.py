import numpy as np
import xarray as xr

# Load dataset containing information
ds = xr.load_dataset('~/Desktop/SP/line-by-line-solver/Database/afgl_1986-us_standard.nc')
altitudes = np.linspace(0., 120., 121)


# level = 50.5
# 7 gases: ['H2O' 'O3' 'N2O' 'CO' 'CH4' 'CO2' 'O2']


def get_atm_gas_data(level):
    # units of each field can be found inside the dataset (print it using jupyter notebook)
    pressure = ds.interp(z_layer=level, kwargs={"fill_value": "extrapolate"}).p.data.item()     # unit: pa
    temperature = ds.interp(z_layer=level, kwargs={"fill_value": "extrapolate"}).t.data.item()  # unit: K
    mol_frac_list = ds.interp(z_layer=level, kwargs={"fill_value": "extrapolate"}).mr.data
    mole_fraction = {}
    index = 0
    for specie in ds.species.data:
        # specie contains the gas name
        mole_fraction[specie] = mol_frac_list[index]
        index += 1
        # With the three variables should be possible to compute the information that we want
    molecules = ds.species.data.tolist()

    return pressure, temperature, molecules, mole_fraction


# p, t, f = get_atm_gas_data(50)
# mole frac: mixing ratio
# iso: all
# loop through all altitudes
# save unweighted/weighted abscoeff:
#   120 x 7 x wavelength table
#   mole_fraction table

# 紫外线到红外线的range, infrared

# multiply/add: DrJit fma
