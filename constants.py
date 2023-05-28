# -*- coding: utf-8 -*-
"""Physical constants.

Note : more units available in :
- scipy.constants
- sympy.physics.units


-------------------------------------------------------------------------------
"""

# %% SI units

eV = 1.602176634e-19  # J
"""float: electron volt (J)

https://physics.nist.gov/cgi-bin/cuu/Value?e
"""

h = 6.62607015e-34
"""float: Planck constant (m2.kg.s-1 = J.s)

https://physics.nist.gov/cgi-bin/cuu/Value?h|search_for=planck"""

k_b = 1.3806490e-23
"""float: Boltzmann constant (m2.kg.s-2.K-1)

https://physics.nist.gov/cgi-bin/cuu/Value?k|search_for=boltzmann
"""

c = 2.99792458e8
"""float: light velocity (m/s)

https://physics.nist.gov/cgi-bin/cuu/Value?c
"""

a_0 = 5.29177e-11
"""float: First Bohr radius (m)"""

m_e = 9.1093837015e-31
"""float: Electron mass (kg)

https://physics.nist.gov/cgi-bin/cuu/Value?me|search_for=electron+mass"""

Ry = 13.605693122994
"""float: Rydberg unit of energy (eV)

https://physics.nist.gov/cgi-bin/cuu/Value?rydhcev|search_for=rydberg"""

Av = 6.02214076e23
"""float: Avogadro constant (molecules / mol)

https://physics.nist.gov/cgi-bin/cuu/Value?na|search_for=avogadro"""
Na = Av  # just an alias

eps_0 = 8.8541878128e-12
"""float: Vacuum permittivity (Farads / m)

https://physics.nist.gov/cgi-bin/cuu/Value?ep0|search_for=vacuum+permittivity"""

c_2 = h * c / k_b * 100  # the 'c2' on HITRAN parameter documentation (hc_k in radis code)
"""float:  cm to K conversion (~ 1.44 K/cm), used in Boltman factors"""


# %% HITRAN (CGS) units

k_b_CGS = 1.380648813e-16
"""float: Boltzman constant (erg / K)"""

c_CGS = 2.99792458e10
"""float: Light velocity (cm / s)"""

h_CGS = 6.626196e-27
"""float: Planck constant (erg * s)"""


""" Parameter sets for the continuum absorption coefficient """
THETA = 300
C_O_H2O = 6.5e-8
n_s = 7.3
C_O_d = 0.206e-8
n_d = 0

