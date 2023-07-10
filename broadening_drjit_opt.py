import drjit as dr
import mitsuba as mi
import numpy as np
import pyfftw
import time

from basics import get_indices, Tref
from constants import Na, k_b_CGS, c_CGS


def lorentzian_step_opt(res_L):
    log_pL = dr.log(dr.sqrt((res_L / 0.20)) + 1)
    return log_pL


def gaussian_step_opt(res_G):
    log_pG = dr.log(dr.sqrt((res_G / 0.46)) + 1)
    return log_pG


dxL = lorentzian_step_opt(0.01)
dxG = gaussian_step_opt(0.01)


# compute HWHM (half width at half of the maximum, gamma in ARTS) for lineshape later
def calc_lorentz_hwhm_opt(dfcolumn_dict, Tgas, diluent, mole_fraction, pressure_atm):
    start = time.time()
    # diluent_molecules = diluent.keys()

    gamma_lb = 0

    # Adding air coefficient
    if "air" in diluent:
        gamma_lb += dr.power(Tref / Tgas, dfcolumn_dict["Tdpair"]) * (
            (dfcolumn_dict["airbrd"] * pressure_atm * dfcolumn_dict["diluent_frac"])
        )

    # Adding self coefficient
    # Check self broadening is here
    if "Tdpsel" not in dfcolumn_dict.keys():
        Tdpsel = None  # if None, voigt_broadening_HWHM uses df.Tdpair
    else:
        Tdpsel = dfcolumn_dict["Tdpsel"]

    if "selbrd" not in dfcolumn_dict.keys():
        selbrd = dfcolumn_dict["airbrd"]
    else:
        selbrd = dfcolumn_dict["selbrd"]

    if Tdpsel is None:  # use Tdpair instead
        gamma_lb += dr.power(Tref / Tgas, dfcolumn_dict["Tdpair"]) * (
            (selbrd * pressure_atm * mole_fraction)
        )
    else:
        gamma_lb += dr.power(Tref / Tgas, Tdpsel) * (
                selbrd * pressure_atm * mole_fraction
        )

    elapsed = time.time() - start
    return gamma_lb, elapsed


def calc_lorentz_ft_opt(w_centered, hwhm_lorentz):
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

    lorentz = dr.exp(-2 * dr.pi * w_centered * hwhm_lorentz)  # use Dr.Git's vectorization
    return lorentz


def calc_gauss_hwhm_opt(dfcolumn_dict, Tgas, molar_mass):
    start = time.time()
    wav = dfcolumn_dict["wav"]
    gamma_doppler = (wav / c_CGS) * dr.sqrt(
        (2 * Na * k_b_CGS * Tgas * dr.log(2)) / molar_mass
    )

    elapsed = time.time() - start
    return gamma_doppler, elapsed


def calc_gauss_ft_opt(w_centered, hwhm_gauss):
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

    gauss = dr.exp(-dr.sqr(2 * dr.pi * w_centered * hwhm_gauss) / (4 * dr.log(2)))
    return gauss


def dr_calc_voigt_ft(w_lineshape_ft, hwhm_gauss, hwhm_lorentz):
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

    IG_FT = calc_gauss_ft_opt(w_lineshape_ft, hwhm_gauss)
    IL_FT = calc_lorentz_ft_opt(w_lineshape_ft, hwhm_lorentz)
    return IG_FT * IL_FT


def calc_lineshape_LDM_opt(dfcolumn_dict, w_lineshape_ft):
    """
    LDM: line density map (https://github.com/radis/radis/issues/37)

    :return:
    line_profile_LDM: dict
            dictionary of Voigt profile template.
            If ``self.params.broadening_method == 'fft'``, templates are calculated
            in Fourier space.
    """
    start = time.time()

    # Prepare steps for Lineshape database
    # ------------------------------------

    def _init_w_axis(w_dat, log_p):
        # TODO: [0] makes the value python float but not mi.Float64 any problem?? same with len() function later
        w_min = dr.min(w_dat)[0]
        if w_min == 0:
            print(" line(s) had a calculated broadening of 0 cm-1. Check the database. At least this line is faulty: \n\n")
            # TODO: filter in drjit version
            w_dat = dr.gather(mi.Float64, w_dat, dr.compress(w_dat > 0.0))
            w_min = dr.min(w_dat)[0]
        w_max = (
                dr.max(w_dat)[0] + 1e-4
        )  # Add small number to prevent w_max falling outside the grid
        N = dr.ceil((dr.log(w_max) - dr.log(w_min)) / log_p) + 1
        return w_min * dr.exp(log_p * dr.arange(mi.UInt, N))

    log_pL = dxL  # LDM user params
    log_pG = dxG  # LDM user params

    wL_dat = dfcolumn_dict["hwhm_lorentz"] * 2  # FWHM
    wG_dat = dfcolumn_dict["hwhm_gauss"] * 2  # FWHM

    wL = _init_w_axis(wL_dat, log_pL)  # FWHM
    wG = _init_w_axis(wG_dat, log_pG)  # FWHM

    # Calculate the Lineshape
    # -----------------------

    # broadening_method = self.params.broadening_method         # only consider "fft" for now
    # Unlike real space methods ('convolve', 'voigt'), here we calculate
    # the lineshape on the full spectral range.

    # Get all combinations of Voigt lineshapes (in Fourier space)
    wg_len = len(wG)
    wl_len = len(wL)
    w_lineshape_ft_len = len(w_lineshape_ft)
    ''' do an outer product of w_lineshape_ft[w] and wG[len(wG)] to get [len(wg) x w] 2d array'''
    # convert w_lineshape_ft to w_centered[len(wG) x w]
    w_centered_wg = dr.tile(w_lineshape_ft, wg_len)
    wg_ldm = dr.zeros(dr.llvm.ad.TensorXf64, [wg_len, w_lineshape_ft_len])
    dr.scatter(target=wg_ldm.array, value=w_centered_wg,
               index=dr.arange(mi.UInt, w_lineshape_ft_len * wg_len))

    wg_tensor = dr.ones(dr.llvm.ad.TensorXf64, [wg_len, 1])
    # TODO: check the mask of mi.Float64 here (gamma_lb in calc_lorentz_hwhm_opt)
    dr.scatter(target=wg_tensor.array, value=mi.Float64(wG/2), index=dr.arange(mi.UInt, wg_len))
    gauss_tensor = calc_gauss_ft_opt(wg_ldm, wg_tensor)  # shape: wg x w

    ''' do an outer product of w_lineshape_ft[w] and wL[len(wL)] to get [len(wL) x w] 2d array'''
    # convert w_lineshape_ft to w_centered[len(wG) x w]
    # ToDO: check if the wl_len also needs to be in drjit type
    w_centered_wl = dr.tile(w_lineshape_ft, wl_len)
    wl_ldm = dr.zeros(dr.llvm.ad.TensorXf64, [wl_len, w_lineshape_ft_len])
    dr.scatter(target=wl_ldm.array, value=w_centered_wl,
               index=dr.arange(mi.UInt, w_lineshape_ft_len * wl_len))

    wl_tensor = dr.ones(dr.llvm.ad.TensorXf64, [wl_len, 1])
    dr.scatter(target=wl_tensor.array, value=mi.Float64(wL/2), index=dr.arange(mi.UInt, wl_len))
    lorentz_tensor = calc_lorentz_ft_opt(wl_ldm, wl_tensor)  # shape: wl x w

    '''stack the 2d tensor corresponding to lorentz and gauss to get a 3d tensor[wg x wl x w]'''
    gauss_3d_tensor = dr.zeros(dr.llvm.ad.TensorXf64, [wg_len, wl_len, w_lineshape_ft_len])
    arr1 = [midx * wl_len * w_lineshape_ft_len + ki for midx in range(wg_len) for ki in range(w_lineshape_ft_len)]
    gauss_idx = [li * w_lineshape_ft_len + a for li in range(wl_len) for a in arr1]
    dr.scatter(target=gauss_3d_tensor.array, value=dr.tile(gauss_tensor.array, wl_len), index=mi.UInt(gauss_idx))

    lorentz_3d_tensor = dr.zeros(dr.llvm.ad.TensorXf64, [wg_len, wl_len, w_lineshape_ft_len])
    dr.scatter(target=lorentz_3d_tensor.array, value=dr.tile(lorentz_tensor.array, wg_len),
               index=dr.arange(mi.UInt, wg_len * wl_len * w_lineshape_ft_len))

    line_profile_LDM = gauss_3d_tensor * lorentz_3d_tensor  # shape: wg x wl x w

    normalize_factor = dr.ones(dr.llvm.ad.TensorXf64, (wg_len, wl_len, 1))
    dr.scatter(normalize_factor.array, line_profile_LDM[:, :, 0].array, dr.arange(mi.UInt, wg_len * wl_len * 1))
    line_profile_LDM /= normalize_factor

    elapsed = time.time() - start
    return line_profile_LDM, wL, wG, wL_dat, wG_dat, elapsed


def dr_add_at_ldm(LDM, k, l, m, I, wg_len, wl_len):
    """Add the linestrengths on the LDM grid.

    Uses the drjit implementation to realize :py:func:`~numpy.add.at`, which
    add arguments element-wise.

    Parameters
    ----------
    LDM : TensorXf64 []
        LDM grid
    k, l, m : array
        index
    I : array
        intensity to add
    wg_len, wl_len: dimension of LDM
    Returns
    -------
    add: ndarray
        linestrengths distributed over the LDM grid

    """
    # print('Numpy add.at()')
    indices = mi.UInt(k * wg_len * wl_len + l * wl_len + m)
    return dr.scatter_reduce(dr.ReduceOp.Add, LDM.array, I, indices)


def apply_lineshape_LDM_opt(
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
    woutrange:
    wstep:
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

    _add_at = dr_add_at_ldm
    # Vectorize the chunk of lines
    S = broadened_param

    # ---------------------------
    # Apply line profile
    # ... First get the closest matching spectral point  (on the left, and on the right)
    #         ... @dev: np.interp about 30% - 50% faster than np.searchsorted

    # LDM : Next calculate how the line is distributed over the 2x2x2 bins.
    # remark: since drjit doesn't have function like np.interp so still use the original version of get_indices
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
    Iv1 = S * avi  # interpolation of the line strength
    wg_len = len(wG)
    wl_len = len(wL)

    # ... Initialize array on which to distribute the lineshapes
    if broadening_method in ["voigt", "convolve"]:
        print("not considered ")
    elif broadening_method == "fft":
        LDM = dr.zeros(dr.llvm.ad.TensorXf64, [2 * len(wavenumber_calc), wg_len, wl_len])
    else:
        raise NotImplementedError(broadening_method)

    # Distribute all line intensities on the 2x2x2 bins.

    _add_at(LDM, ki0, li0, mi0, Iv0 * awV00, wg_len, wl_len)
    _add_at(LDM, ki0, li0, mi1, Iv0 * awV01, wg_len, wl_len)
    _add_at(LDM, ki0, li1, mi0, Iv0 * awV10, wg_len, wl_len)
    _add_at(LDM, ki0, li1, mi1, Iv0 * awV11, wg_len, wl_len)
    _add_at(LDM, ki1, li0, mi0, Iv1 * awV00, wg_len, wl_len)
    _add_at(LDM, ki1, li0, mi1, Iv1 * awV01, wg_len, wl_len)
    _add_at(LDM, ki1, li1, mi0, Iv1 * awV10, wg_len, wl_len)
    _add_at(LDM, ki1, li1, mi1, Iv1 * awV11, wg_len, wl_len)

    # All lines within each bins are convolved with the same lineshape.
    # ... Initialize array in FT space
    line_profile_LDM = np.array(line_profile_LDM)
    Ildm_FT = 1j * np.zeros(len(line_profile_LDM[0][0]))

    # change numpy.fft to pyfftw
    Ildm_FT += np.sum((pyfftw.interfaces.numpy_fft.rfft(np.array(LDM), axis=0).transpose(1, 2, 0)) * line_profile_LDM, axis=(0, 1))

    # Back in real space:
    sumoflines_calc = pyfftw.interfaces.numpy_fft.irfft(Ildm_FT)[: len(wavenumber_calc)]
    sumoflines_calc /= wstep
    # Get valid range (discard wings)
    sumoflines = sumoflines_calc[woutrange[0]: woutrange[1]]

    elapsed = time.time() - start

    return sumoflines, elapsed
