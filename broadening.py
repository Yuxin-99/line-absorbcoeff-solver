import time

import numpy as np

from basics import get_indices, Tref, pressure_atm
from constants import Na, k_b_CGS, c_CGS, THETA, C_O_H2O, n_s, C_O_d, n_d


def lorentzian_step(res_L):
    log_pL = np.log((res_L / 0.20) ** 0.5 + 1)
    return log_pL


def gaussian_step(res_G):
    log_pG = np.log((res_G / 0.46) ** 0.5 + 1)
    return log_pG


# compute HWHM (half width at half of the maximum, gamma in ARTS) for lineshape later
def calc_lorentz_hwhm(df, Tgas, diluent, mole_fraction):
    start = time.time()
    diluent_molecules = diluent.keys()

    gamma_lb = 0

    # Adding air coefficient
    if "air" in diluent_molecules:
        gamma_lb += ((Tref / Tgas) ** df.Tdpair) * (
            (df.airbrd * pressure_atm * diluent["air"])
        )

    # Adding self coefficient
    # Check self broadening is here
    if not "Tdpsel" in list(df.keys()):
        Tdpsel = None  # if None, voigt_broadening_HWHM uses df.Tdpair
    else:
        Tdpsel = df.Tdpsel

    if not "selbrd" in list(df.keys()):
        selbrd = df.airbrd
    else:
        selbrd = df.selbrd

    if Tdpsel is None:  # use Tdpair instead
        gamma_lb += ((Tref / Tgas) ** df.Tdpair) * (
            (selbrd * pressure_atm * mole_fraction)
        )
    else:
        gamma_lb += ((Tref / Tgas) ** Tdpsel) * (
                selbrd * pressure_atm * mole_fraction
        )

    df['hwhm_lorentz'] = gamma_lb
    elapsed = time.time() - start
    return elapsed


def calc_lorentz_ft(w_centered, hwhm_lorentz):
    r"""Fourier Transform of a Lorentzian lineshape.

        .. math::
            \operatorname{exp}\left(-2\pi w_{centered} \gamma_{lb}\right)

        Parameters
        ----------
        w_centered: 2D array       [one per line: shape W x N]
            waverange (nm / cm-1) (centered on 0)
        hwhm_lorentz: array   (cm-1)        [length N]
            half-width half maximum coefficient (HWHM) for pressure broadening
            calculation

        Returns
        -------
        array

        See Also
        --------
        :py:func:`~radis.lbl.broadening.lorentzian_lineshape`
        """

    lorentz = np.exp(-2 * np.pi * w_centered * hwhm_lorentz)            # use Dr.Git's vectorization
    return lorentz


def calc_gauss_hwhm(df, Tgas, molar_mass):
    start = time.time()
    wav = df.wav
    gamma_doppler = (wav / c_CGS) * np.sqrt(
        (2 * Na * k_b_CGS * Tgas * np.log(2)) / molar_mass
    )

    df['hwhm_gauss'] = gamma_doppler
    elapsed = time.time() - start
    return elapsed


def calc_gauss_ft(w_centered, hwhm_gauss):
    r"""Fourier Transform of a Gaussian lineshape.

        .. math::
            \operatorname{exp}\left(\frac{-\left({\left(2\pi w_{centered} hwhm\right)}^2\right)}{4\ln2}\right)

        Parameters
        ----------
        w_centered: 2D array       [one per line: shape W x N]
            waverange (nm / cm-1) (centered on 0)
        hwhm_gauss:  array   [shape N = number of lines]
            Half-width at half-maximum (HWHM) of Gaussian

        Returns
        -------
        array

        See Also
        --------
        :py:func:`~radis.lbl.broadening.gaussian_lineshape`
        """

    gauss = np.exp(-((2 * np.pi * w_centered * hwhm_gauss) ** 2) / (4 * np.log(2)))
    return gauss


def calc_lorentz_profile():
    print("")


def calc_voigt_ft(w_lineshape_ft, hwhm_gauss, hwhm_lorentz):
    """Fourier Transform of a Voigt lineshape

        Parameters
        ----------
        w_lineshape_ft: 1D array       [shape W + 1]
            sampled fourier frequencies
        hwhm_gauss:  array   [shape N = number of lines]
            Half-width at half-maximum (HWHM) of Gaussian
        hwhm_lorentz: array   (cm-1)        [length N]
            Half-width at half-maximum (HWHM) of Lorentzian

        See Also
        --------

        :py:func:`~radis.lbl.broadening.gaussian_FT`
        :py:func:`~radis.lbl.broadening.lorentzian_FT`
        """

    IG_FT = calc_gauss_ft(w_lineshape_ft, hwhm_gauss)
    IL_FT = calc_lorentz_ft(w_lineshape_ft, hwhm_lorentz)
    return IG_FT * IL_FT


def calc_lineshape_LDM(df, wstep, wavenumber_calc):
    """
    LDM: line density map (https://github.com/radis/radis/issues/37)

    :return:
    line_profile_LDM: dict
            dictionary of Voigt profile template.
            If ``self.params.broadening_method == 'fft'``, templates are calculated
            in Fourier space.
    """
    start = time.time()
    dxL = lorentzian_step(0.01)
    dxG = gaussian_step(0.01)

    # Prepare steps for Lineshape database
    # ------------------------------------

    def _init_w_axis(w_dat, log_p):
        w_min = w_dat.min()
        if w_min == 0:
            print(f"{(w_dat == 0).sum()}"
                  + " line(s) had a calculated broadening of 0 cm-1. Check the database. At least this line is faulty: \n\n")
            w_min = w_dat[w_dat > 0].min()
        w_max = (
                w_dat.max() + 1e-4
        )  # Add small number to prevent w_max falling outside of the grid
        N = np.ceil((np.log(w_max) - np.log(w_min)) / log_p) + 1
        return w_min * np.exp(log_p * np.arange(N))

    log_pL = dxL  # LDM user params
    log_pG = dxG  # LDM user params

    wL_dat = df.hwhm_lorentz.values * 2  # FWHM
    wG_dat = df.hwhm_gauss.values * 2  # FWHM

    wL = _init_w_axis(wL_dat, log_pL)  # FWHM
    wG = _init_w_axis(wG_dat, log_pG)  # FWHM

    # Calculate the Lineshape
    # -----------------------

    line_profile_LDM = {}
    # broadening_method = self.params.broadening_method         # only consider "fft" for now
    # Unlike real space methods ('convolve', 'voigt'), here we calculate
    # the lineshape on the full spectral range.
    w = wavenumber_calc
    w_lineshape_ft = np.fft.rfftfreq(
        2 * len(w), wstep
    )  # TO-DO: add  + self.misc.zero_padding

    w_fold = (w_lineshape_ft, w_lineshape_ft[::-1])

    # Get all combinations of Voigt lineshapes (in Fourier space)
    for l in range(len(wG)):
        line_profile_LDM[l] = {}
        for m in range(len(wL)):
            line_profile_LDM[l][m] = calc_voigt_ft(
                w_lineshape_ft, wG[l] / 2, wL[m] / 2
            )  # compute the effect of hwhm to the line shape in fourier space directly

            # not consider folding for now
            # # Add folding until threshold is reached:
            # n = 1
            # while (
            #         voigt_FT(n / (2 * wstep), wG[l] / 2, wL[m] / 2)
            #         >= self.params.folding_thresh
            # ):
            #     line_profile_LDM[l][m] += voigt_FT(
            #         n / (2 * wstep) + w_fold[n & 1], wG[l] / 2, wL[m] / 2
            #     )
            #     n += 1

            # normalization based on the fourier frequency of 0
            line_profile_LDM[l][m] /= line_profile_LDM[l][m][0]

    elapsed = time.time() - start
    return line_profile_LDM, wL, wG, wL_dat, wG_dat, elapsed


def add_at_ldm(LDM, k, l, m, I):
    """Add the linestrengths on the LDM grid.

    Uses the numpy implementation of :py:func:`~numpy.add.at`, which
    add arguments element-wise.

    Parameters
    ----------
    LDM : ndarray
        LDM grid
    k, l, m : array
        index
    I : array
        intensity to add

    Returns
    -------
    add: ndarray
        linestrengths distributed over the LDM grid

    Notes
    -----
    Cython version implemented in https://github.com/radis/radis/pull/234

    See Also
    --------
    :py:func:`numpy.add.at`

    """
    # print('Numpy add.at()')
    return np.add.at(LDM, (k, l, m), I)


def apply_lineshpe_LDM(
        broadened_param,
        wavenumber,
        wavenumber_calc,
        woutrange,
        shifted_wavenum,
        wstep,
        line_profile_LDM,
        wL,
        wG,
        wL_dat,
        wG_dat):
    """Multiply `broadened_param` by `line_profile` and project it on the
    correct wavelength given by `shifted_wavenum` (actually not shifted for now)

    Parameters
    ----------
    broadened_param: pandas Series (or numpy array)   [size N = number of lines]
        Series to apply lineshape to. Typically linestrength `S` for absorption,
    wavenumber: the spectral range vector, vector of wavenumbers (shape W)
    wavenumber_calc: the spectral range used for calculation
    shifted_wavenum: (cm-1)     pandas Series (size N = number of lines)
        center wavelength (used to project broaded lineshapes )
    line_profile_LDM:  dict
        dict of line profiles ::

            lineshape = line_profile_LDM[gaussian_index][lorentzian_index]

        If ``self.params.broadening_method == 'fft'``, templates are given
        in Fourier space.
    wL: array       (size DL)
        array of all Lorentzian widths in LDM
    wG: array       (size DG)
        array of all Gaussian widths in LDM
    wL_dat: array    (size N)
        FWHM of all lines. Used to lookup the LDM
    wG_dat: array    (size N)
        FWHM of all lines. Used to lookup the LDM
    optimization :
        if ``"min-RMS"`` weights optimized by analytical minimization of the RMS-error.
        Otherwise, weights equal to their relative position in the grid.
        Only consider the case of "simple" for now

    Returns
    -------
    sumoflines: array (size W  = size of output wavenumbers)
        sum of (broadened_param x line_profile)

    Notes
    -----
    Units change during convolution::

        [sumoflines] = [broadened_param] * cm

    Reference
    ---------
    LDM implemented based on a code snippet from D.v.d.Bekerom.
    See: https://github.com/radis/radis/issues/37

    See Also
    --------
    :py:meth:`~radis.lbl.broadening.BroadenFactory._calc_lineshape_LDM`
    """
    start = time.time()
    # Get spectrum range
    broadening_method = "fft"  # only consider the case of "fft" for now

    # Get add-at method
    # ... 1. allow user to use non-cython method (useful for tests ?)
    # ... 2. write in the Spectrum object whether Cython was used or not
    # ...    (either because deactivated, or because not installed)
    # if self.use_cython and add_at != numpy_add_at:
    #     _add_at = add_at
    #     self.misc.add_at_used = "cython"
    # else:
    #     _add_at = numpy_add_at
    #     self.misc.add_at_used = "numpy"
    _add_at = add_at_ldm
    # Vectorize the chunk of lines
    S = broadened_param

    # ---------------------------
    # Apply line profile
    # ... First get closest matching spectral point  (on the left, and on the right)
    #         ... @dev: np.interp about 30% - 50% faster than np.searchsorted

    # LDM : Next calculate how the line is distributed over the 2x2x2 bins.
    ki0, ki1, tvi = get_indices(shifted_wavenum, wavenumber_calc)
    li0, li1, tGi = get_indices(np.log(wG_dat), np.log(wG))
    mi0, mi1, tLi = get_indices(np.log(wL_dat), np.log(wL))

    # Next assign simple weights:
    # Weights computed from the "min-RMS" optimization is not considered for now
    # Simple weigths:
    avi = tvi
    aGi = tGi
    aLi = tLi

    # ... fractions on LDM grid
    awV00 = (1 - aGi) * (1 - aLi)
    awV01 = (1 - aGi) * aLi
    awV10 = aGi * (1 - aLi)
    awV11 = aGi * aLi

    Iv0 = S * (1 - avi)
    Iv1 = S * avi   # interpolation of the line strength

    # ... Initialize array on which to distribute the lineshapes
    if broadening_method in ["voigt", "convolve"]:
        print("not considered ")
        # if self.params.sparse_ldm == True:
        #     # LDM is constructed in a sparse-way later
        #     pass
        # else:
        #     LDM = np.zeros((len(wavenumber_calc) + 2, len(wG), len(wL)))
        #     # +2 to allocate one empty grid point on each side : case where a line is on the boundary
        #     ki0 += 1
        #     ki1 += 1
    elif broadening_method == "fft":
        # if self.params.sparse_ldm == True:
        #     if self.verbose >= 2:
        #         print(
        #             "SPARSE optimisation not implemented with 'fft' mode. Use 'voigt' for analytical voigt, or radis.config['SPARSE_WAVERANGE'] = False"
        #         )
        LDM = np.zeros(
            (
                2 * len(wavenumber_calc),  # TO-DO: Add  + self.misc.zero_padding
                len(wG),
                len(wL),
            )
        )
    else:
        raise NotImplementedError(broadening_method)

    # Distribute all line intensities on the 2x2x2 bins.
    _add_at(LDM, ki0, li0, mi0, Iv0 * awV00)
    _add_at(LDM, ki0, li0, mi1, Iv0 * awV01)
    _add_at(LDM, ki0, li1, mi0, Iv0 * awV10)
    _add_at(LDM, ki0, li1, mi1, Iv0 * awV11)
    _add_at(LDM, ki1, li0, mi0, Iv1 * awV00)
    _add_at(LDM, ki1, li0, mi1, Iv1 * awV01)
    _add_at(LDM, ki1, li1, mi0, Iv1 * awV10)
    _add_at(LDM, ki1, li1, mi1, Iv1 * awV11)

    # All lines within each bins are convolved with the same lineshape.
    # ... Initialize array in FT space
    Ildm_FT = 1j * np.zeros(len(line_profile_LDM[0][0]))
    for l in range(len(wG)):
        for m in range(len(wL)):
            lineshape_FT = line_profile_LDM[l][m]
            Ildm_FT += np.fft.rfft(LDM[:, l, m]) * lineshape_FT
    # Back in real space:
    sumoflines_calc = np.fft.irfft(Ildm_FT)[: len(wavenumber_calc)]
    sumoflines_calc /= wstep
    # Get valid range (discard wings)
    sumoflines = sumoflines_calc[woutrange[0]: woutrange[1]]

    elapsed = time.time() - start

    return sumoflines, elapsed


def calc_continuum_absorb(wavenumber, h2o_fraction, pressure):
    # Formula:uih
    p_H2O = h2o_fraction * pressure     # unit: bar/ 1e3 hpa
    p_d = (1 - h2o_fraction) * pressure
    v = wavenumber * c_CGS      # s-1 / hz
    v = v * 1e-9        # Ghz
    con_absorbcoeff = (v ** 2) * (THETA ** 3) * (C_O_H2O * (p_H2O ** 2) * (THETA ** n_s) + C_O_d * p_H2O * p_d * (THETA ** n_d))  # dB/km
    con_absorbcoeff = np.log(con_absorbcoeff / 10, 10)     # cm-1
    return con_absorbcoeff
