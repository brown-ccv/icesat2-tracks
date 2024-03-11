import numpy as np
from astropy.timeseries import LombScargle
import scipy.signal.windows as WINDOWS
from scipy.special import gammainc
import copy
import xarray as xr
from scipy.signal import detrend
import pandas as pd
from scipy import signal
import lmfit as LM
import icesat2waves.local_modules.m_general_ph3 as M
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt


def is_not_even(number):
    return True if (number % 2) else False


# basic functions
def create_chunk_boundaries(L, dsize, ov=None, iter_flag=True):
    """
    returns all need chunk boundaries  and center position given L, and ov
    inputs:
    L desired length of window,
    dsize size of the data

    if ov is None, = L/2
    if iter_flag True returns iter else it returns an ndarray

    """
    ov = int(np.round(L / 2)) if ov is None else ov

    xleft = np.arange(0, dsize - int(L - ov), int(L - ov))
    xright = np.arange(int(L - ov) * 2, dsize + 1, int(L - ov))
    xcenter_pos = np.arange(int(L - ov), dsize - int(L - ov) + 1, int(L - ov))
    max_size = min([xleft.size, xcenter_pos.size, xright.size])
    position_stancil = np.vstack(
        [xleft[0:max_size], xcenter_pos[0:max_size], xright[0:max_size]]
    )

    return iter(position_stancil.T.tolist()) if iter_flag else position_stancil


def create_chunk_boundaries_unit_lengths(L_unit, data_limits, ov=None, iter_flag=True):
    """
    returns all need chunk boundaries and center position given L, and ov
    inputs:
    L desired length of window in units of the x axis of the data,
    data_limits (x_min, x_max) tuple with the beginning and end the the derived window stancils

    if ov is None, = L/2
    if iter_flag True returns iter else it returns an ndarray

    """
    L = L_unit
    ov = np.round(L / 2) if ov is None else ov
    dl = L - ov
    xleft = np.arange(data_limits[0], data_limits[1] - dl, dl)
    xcenter_pos = np.arange(data_limits[0] + L / 2, data_limits[1] - dl + 1, dl)
    xright = np.arange(data_limits[0] + L, data_limits[1] + 1, dl)

    max_size = min([xleft.size, xcenter_pos.size, xright.size])

    position_stancil = np.vstack(
        [xleft[0:max_size], xcenter_pos[0:max_size], xright[0:max_size]]
    )

    return iter(position_stancil.T.tolist()) if iter_flag else position_stancil


def Z_to_power(Z, df, N):
    """compute the 1d spectrum of a field phi
    inputs:
    Z       complex fourier coefficients
    df      frequency / or wavenumber step
    N       length of data vector in real space (= L)
    """
    spec = 2.0 * (Z * Z.conj()).real / df / N**2
    neven = is_not_even(N)
    spec[0] = spec[0] / 2.0
    if neven:
        spec[-1] = spec[-1] / 2.0

    return spec


# 2nd cal spectra
def calc_spectrum_fft(phi, df, N):
    """compute the 1d spectrum of a field phi
    inputs:

    df      frequency / or wavenumber step
    N       length of data vector in real space (= L)
    neven   bool, if True
    """
    neven = is_not_even(N)

    phih = np.fft.rfft(phi)

    # the factor of 2 comes from the symmetry of the Fourier coeffs
    spec = 2.0 * (phih * phih.conj()).real / df / N**2

    # the zeroth frequency should be counted only once
    spec[0] = spec[0] / 2.0
    if neven:
        spec[-1] = spec[-1] / 2.0

    return spec


def LS_power_to_PSD(ls_power, L, dff):
    """
    returns Power spectral density (unit^2/dfreq)
    ls_power    output of astropy.timeseries.LombScargle.power with normalization='psd'
    """
    return 2 * ls_power / L / dff


def calc_spectrum_LS(x, y, k, err=None, LS=None, dk=None):
    """
    returns:
    Power spectral density of y given postitions x, for wanumbers k
    LS is a Lomb-scargel object. if None its initlized again
    and
    LS     LombScargle object
    """
    if LS is None:
        LS = LombScargle(x, y, dy=err, fit_mean=False, center_data=True)
    else:
        LS.t = x
        LS.y = y
        LS.dy = err

    ls_power = LS.power(k, normalization="psd", assume_regular_frequency="False")

    dk = np.diff(k).mean() if dk is None else dk
    return 2 * ls_power / y.size / dk, LS


def reconstruct_data_from_LS(LS, x_real_axis, freq):
    """
    This method return reconstructed field given a LombScargle object.
    LS          is the LombScargle object
    x_real_axis is the x coordinate of the original data (np.array)
    freq        is the frequency grid on which the field is reconstructed
    """

    y_reconstruct = LS.offset() * np.ones(len(x_real_axis))

    freq_seq = freq[1:] if freq[0] == 0 else freq
    freq_mask = freq <= 1 / 100

    for fi in freq_seq:
        try:
            theta = LS.model_parameters(fi)
        except:
            theta = [0, 0]
        y_reconstruct += theta[0] * np.sin(x_real_axis * 2 * np.pi * fi) + theta[
            1
        ] * np.cos(x_real_axis * 2 * np.pi * fi)

    return y_reconstruct


def calc_freq_fft(x_grid, N):
    """calculate array of spectral variable (frequency or
    wavenumber) in cycles per unit of L"""

    neven = is_not_even(N)

    dx = np.round(np.median(np.diff(x_grid)), 1)
    df = 1.0 / ((N - 1) * dx)
    f = df * np.arange(N / 2 + 1) if neven else df * np.arange((N - 1) / 2.0 + 1)
    return f, df


def calc_freq_fft_given_dx(dx, N):
    """
    calculate array of spectral variable (frequency or
            wavenumber) in cycles per unit of L
    dx      given resolution of the data
    N       number of datapoints used in window
    """

    neven = is_not_even(N)
    df = 1.0 / ((N - 1) * dx)
    f = df * np.arange(N / 2 + 1) if neven else df * np.arange((N - 1) / 2.0 + 1)
    return f, df


def calc_freq_LS(
    x,
    N,
    method="fftX2",
    dx=None,
    minimum_frequency=None,
    maximum_frequency=None,
    samples_per_peak=0.01,
):
    """
    calculate array of spectral variable (frequency or
    wavenumber) in cycles per unit of N (window length in number of data points)
    x can be unevenly spaced
    method:
        "fftX2"     defined the frequencyt grid as for FFT, but with double its resolution
        "LS_auto"   using LS algorithm with samples_per_peak=0.1

        minimum_frequency, maximum_frequency only used for LS_auto
    """

    if method == "fftX2":
        neven = is_not_even(N)
        dx = np.diff(x).mean() if dx is None else dx
        df = 1.0 / ((N - 1) * dx) / 2
        f = df * np.arange(df, N + 1) if neven else df * np.arange(df, N)

    elif method is "fft":
        neven = is_not_even(N)
        dx = np.diff(x).mean() if dx is None else dx
        df = 1.0 / ((N - 1) * dx)
        f = df * np.arange(N / 2 + 1) if neven else df * np.arange((N - 1) / 2.0 + 1)

    elif method is "LS_auto":

        f = LombScargle(x, np.random.randn(len(x)), fit_mean=True).autofrequency(
            minimum_frequency=minimum_frequency,
            maximum_frequency=maximum_frequency,
            samples_per_peak=samples_per_peak,
        )

        df = np.diff(f).mean()
        df = np.round(df, 5)

    elif method is "fixed_ratio":

        neven = is_not_even(N)
        dx = np.diff(x).mean() if dx is None else dx
        df = dx / 50
        f = df * np.arange(df, N + 1) if neven else df * np.arange(df, N)

    return f, df


def create_window(L, window=None):
    """
    define window function and weight it to conserve variance
    if window is not None it show have a length of N
    """
    if window is None:
        win = np.hanning(L)
    else:
        win = window

    factor = np.sqrt(L / (win**2).sum())
    win *= factor
    return win


def create_weighted_window(data, window=None):
    """
    define window function and weight it to conserve variance
    if window is not None it show have a length of N
    """

    L = data.size
    if window is None:
        win = WINDOWS.tukey(L, alpha=0.1, sym=True)
    else:
        win = window

    factor = np.sqrt(np.var(data) / np.var((data * win)))
    win *= factor
    return win


def spec_error(E, sn, ci=0.95):
    """Computes confidence interval for one-dimensional spectral
    estimate E (the power spectra).

    Parameters
    ===========
    - sn is the number of spectral realizations;
            it can be either an scalar or an array of size(E)
    - ci = .95 for 95 % confidence interval

    Output
    ==========
    lower (El) and upper (Eu) bounds on E"""

    def yNlu(sn, yN, ci):
        """compute yN[l] yN[u], that is, the lower and
        upper limit of yN"""

        # cdf of chi^2 dist. with 2*sn DOF
        cdf = gammainc(sn, sn * yN)

        # indices that delimit the wedge of the conf. interval
        fl = np.abs(cdf - ci).argmin()
        fu = np.abs(cdf - 1.0 + ci).argmin()

        return yN[fl], yN[fu]

    dbin = 0.005
    yN = np.arange(0, 2.0 + dbin, dbin)

    El, Eu = np.empty_like(E), np.empty_like(E)

    try:
        n = sn.size
    except AttributeError:
        n = 0

    if n:

        assert n == E.size, " *** sn has different size than E "

        for i in range(n):
            yNl, yNu = yNlu(sn[i], yN=yN, ci=ci)
            El[i] = E[i] / yNl
            Eu[i] = E[i] / yNu

    else:
        yNl, yNu = yNlu(sn, yN=yN, ci=ci)
        El = E / yNl
        Eu = E / yNu

    return El, Eu


def linear_gap_fill(F, key_lead, key_int):
    """
    F pd.DataFrame
    key_lead   key in F that determined the independent coordindate
    key_int     key in F that determined the dependent data
    """
    y_g = np.array(F[key_int])

    nans, x2 = np.isnan(y_g), lambda z: z.nonzero()[0]
    y_g[nans] = np.interp(x2(nans), x2(~nans), y_g[~nans])

    return y_g


def sub_sample_coords(X, lons, lats, stancils, map_func=None):
    """
    X            non-nan array of coodinate along beam
    lons, lats   arrays of postion data that should be mapped. must have same size as X
    stancils     interable stancil positions
    map_func (None) If not None the function is used to apply proceudre to each stancil

    returns
    nparray(3, N)   1st column is the stancil center, 2nd and 3rd collumn are the mapped lons and lats
    """

    def get_lon_lat_coords(stancil):

        x_mask = (stancil[0] <= X) & (X <= stancil[-1])
        if sum(x_mask) == 0:
            return np.array([stancil[1], np.nan, np.nan])

        lon_bin, lat_bin = lons[x_mask].mean(), lats[x_mask].mean()

        return np.array([stancil[1], lon_bin, lat_bin])

    map_func = map if map_func is None else map_func
    coord_positions = list(map_func(get_lon_lat_coords, copy.copy(stancils)))

    coord_positions = np.vstack(coord_positions)
    return coord_positions


class WavenumberSpectrogram:
    def __init__(self, x_grid, data, Lpoints, ov=None, window=None):
        """
        returns a wavenumber spectrogram with the resolution L-ov
        this uses standard fft and assumes equally gridded data

        inputs:
        data    data
        x       grid the fft is taken on
        L       window length in number of grid points
        ov      (default=None) number of grid points the windows should overlab
        window  (default=np.hanning) numpy window function

        returns:
        xr.Dataset with x, k as cooridates of the spectrogram and the mean error
            other arributes are in the .attr dict.
        """

        self.Lpoints = Lpoints
        # when not defined in create_chunk_boundaries then L/2
        self.ov = int(Lpoints / 2) if ov is None else ov

        self.data = data

        # create subsample k
        self.k, self.dk = calc_freq_fft(
            x_grid, Lpoints
        )  # return 1/ unit of frid points
        # to get the waveumber units (2  pi/ lambda), multiply by 2 pi
        self.k, self.dk = self.k * 2 * np.pi, self.dk * 2 * np.pi

        # create window
        self.win = create_window(Lpoints)

    def cal_spectrogram(self, data=None, name="power_spec"):
        """
        defines apply function and calculated all sub-sample sprectra using map
        """

        DATA = self.data if data is None else data
        Lpoints, dk = self.Lpoints, self.dk
        win = self.win

        def calc_spectrum_apply(stancil):

            "returns spectrum per stencil, detrends and windows the data"
            idata = DATA[stancil[0] : stancil[-1]]
            idata = detrend(idata) * win

            return stancil[1], calc_spectrum_fft(idata, dk, Lpoints)

        # derive L2 stancil
        stancil_iter = create_chunk_boundaries(Lpoints, DATA.size, ov=self.ov)
        # apply func to all stancils
        D_specs = dict(map(calc_spectrum_apply, stancil_iter))

        chunk_positions = np.array(list(D_specs.keys()))

        # number of spectal relazations
        self.N_stancils = len(chunk_positions)

        # repack data, create xarray
        self.spec_name = name
        G = {
            xi: xr.DataArray(
                I,
                dims=["k"],
                coords={"k": self.k, "x": xi},
                name=self.spec_name,
            )
            for xi, I in D_specs.items()
        }

        self.G = xr.concat(G.values(), dim="x").T
        if self.G.k[0] == 0:
            self.G = self.G[1:, :]

        self.G.attrs["ov"] = self.ov
        self.G.attrs["L"] = self.Lpoints

        return self.G

    def calc_var(self):
        """Compute total variance from spectragram"""
        # do not consider zeroth frequency
        return self.dk * self.G.mean("x").sum().data

    def mean_spectral_error(self, mask=None, confidence=0.95):
        "retrurns spetral error for the x-mean spectral estimate and stores it as coordindate in the dataarray"
        #  make error estimate
        if mask is not None:
            meanspec = self.G.isel(x=mask).mean("x")
            N = int(sum(mask))
        else:
            meanspec = self.G.mean("x")
            N = self.N_stancils

        El_of_mean, Eu_of_mean = spec_error(meanspec, N, confidence)
        El_of_mean.name = "El_mean"
        Eu_of_mean.name = "Eu_mean"

        self.G.coords["mean_El"] = (("k"), El_of_mean.data)
        self.G.coords["mean_Eu"] = (("k"), Eu_of_mean.data)

    def parceval(self, add_attrs=True):
        "test Parceval theorem"
        DATA = self.data
        L = self.Lpoints

        # derive mean variances of stancils
        stancil_iter = create_chunk_boundaries(L, DATA.size)

        def get_stancil_var_apply(stancil):

            "returns the variance of yy for stancil"
            idata = DATA[stancil[0] : stancil[-1]]
            idata = detrend(idata)
            return stancil[1], idata.var()

        D_vars = dict(map(get_stancil_var_apply, stancil_iter))

        stancil_vars = list()
        for I in D_vars.values():
            stancil_vars.append(I)

        print("Parcevals Theorem:")
        print("variance of unweighted timeseries: ", DATA.var())
        print("mean variance of detrended chunks: ", np.array(stancil_vars).mean())
        print("variance of the pwelch Spectrum: ", self.calc_var())

        if add_attrs:
            self.G.attrs["variance_unweighted_data"] = DATA.var()
            self.G.attrs["mean_variance_detrended_chunks"] = np.array(
                stancil_vars
            ).mean()
            self.G.attrs["mean_variance_pwelch_spectrum"] = self.calc_var()


class WavenumberSpectrogramLSEven:
    def __init__(
        self, x, data, L, waven_method="fftX2", dy=None, ov=None, window=None, kjumps=1
    ):
        """
        returns a wavenumber spectrogram with the resolution L-ov
        this uses Lombscargle

        inputs:
        data    data
        x       grid the fft is taken on
        dy      passed to LS object "error or sequence of observational errors associated with times t"
        waven_method     ('auto' (default), or )
        L       window length in number of grid points
        ov      (default=None) number of grid points the windows should overlab
        window  (default=np.hanning) numpy window function

        returns:
        xr.Dataset with x, k as cooridates of the spectrogram and the mean error
            other arributes are in the .attr dict.
        """

        self.L = L
        self.ov = (
            int(L / 2) if ov is None else ov
        )  # when not defined in create_chunk_boundaries then L/2

        self.x = x
        self.data = data
        self.dy = dy

        # create subsample k
        if type(waven_method) is str:
            self.k, self.dk = calc_freq_LS(x, L, method=waven_method)
        elif type(waven_method) is np.ndarray:
            self.k, self.dk = waven_method, np.diff(waven_method).mean()
        else:
            raise ValueError("waven_method is neither string nor an array")

        self.k, self.dk = self.k[::kjumps], self.dk * kjumps
        self.win = None

    def cal_spectrogram(self, x=None, data=None, name="power_spec", dx=1):
        """
        defines apply function and calculated all sub-sample sprectra using map
        dx      nominal resolution of the data resolutionif not set, dx= 1
        """

        X = self.x if x is None else x  # all x positions
        DATA = self.data if data is None else data  # all data points
        L, dk = self.L, self.dk
        win = self.win
        self.dx = dx
        # init Lomb scargle object with noise as nummy data ()
        self.LS = LombScargle(X[0:L], np.random.randn(L) * 0.001, fit_mean=True)

        def calc_spectrum_apply(stancil):

            "returns spectrum per stencil, detrends and windows the data"

            x = X[stancil[0] : stancil[-1]]
            idata = DATA[stancil[0] : stancil[-1]]
            y = detrend(idata)

            LS_PSD, LS_object = calc_spectrum_LS(x, y, self.k, LS=self.LS, dk=self.dk)
            return stancil[1], LS_PSD

        # % derive L2 stancil
        stancil_iter = create_chunk_boundaries(L, DATA.size, ov=self.ov)
        # apply func to all stancils
        D_specs = dict(map(calc_spectrum_apply, stancil_iter))

        chunk_positions = np.array(list(D_specs.keys()))

        # number of spectal relazations
        self.N_stancils = len(chunk_positions)

        # repack data, create xarray
        self.spec_name = name
        G = dict()
        for xi, I in D_specs.items():
            G[xi] = xr.DataArray(
                I,
                dims=["k"],
                coords={"k": self.k, "x": xi * self.dx},
                name=self.spec_name,
            )

        self.G = xr.concat(G.values(), dim="x").T
        if self.G.k[0] == 0:
            self.G = self.G[1:, :]

        self.G.attrs["ov"] = self.ov
        self.G.attrs["L"] = self.L

        return self.G

    def calc_var(self):
        return WavenumberSpectrogram.calc_var(self)

    def parceval(self, add_attrs=True):
        return WavenumberSpectrogram.parceval(self, add_attrs=add_attrs)

    def mean_spectral_error(self, confidence=0.95):
        return WavenumberSpectrogram.mean_spectral_error(self, confidence=confidence)


class WavenumberSpectrogramLS:
    def __init__(
        self, x, data, L, dx, dy=None, waven_method="fftX2", ov=None, window=None
    ):
        """
        returns a wavenumber spectrogram with the resolution L-ov
        this uses Lombscargle

        inputs:
        data    data
        x       x-positions of where the data is taken
        dy      passed to LS object "error or sequence of observational errors associated with times t"
        waven_method     ('auto' (default), or something else if "waven_method" is an array these wavenumbers are used )

        L       window length in units of x
        ov      (default=None) number of grid points the windows should overlab
        window  (default=np.hanning) numpy window function

        returns:
        xr.Dataset with x, k as cooridates of the spectrogram and the mean error
            other arributes are in the .attr dict.
        """

        self.L = L
        self.ov = (
            int(L / 2) if ov is None else ov
        )  # when not defined in create_chunk_boundaries then L/2

        self.x = x
        self.dx = dx
        self.data = data
        self.error = dy if dy is not None else None
        self.Lpoints = int(self.L / self.dx)

        # create subsample k
        if type(waven_method) is str:
            self.k, self.dk = calc_freq_LS(x, self.Lpoints, method=waven_method)
        elif type(waven_method) is np.ndarray:
            self.k, self.dk = waven_method, np.diff(waven_method).mean()
        else:
            raise ValueError("waven_method is neither string nor an array")
        self.win = None

    def cal_spectrogram(
        self,
        x=None,
        data=None,
        error=None,
        name=None,
        xlims=None,
        weight_data=True,
        max_nfev=None,
        map_func=None,
    ):
        """
        defines apply function and calculated all sub-sample sprectra using map

        input:
        x (None)
        data (None)
        error (None)
        name (None)

        creates:
        all kinds of self. variables:
        self.G is an xr.DataArray with the best guess spectral power.
        self.GG is a xr.Dataset with the best guess conmplex conjugate and the rar spectral power

        returns:
        self.GG, params_dataframe
            params_dataframe is a pd.DataFrame that containes all the parameters of the fitting process (and may contain uncertainties too once they are calculated)
        """

        X = self.x if x is None else x  # all x positions
        DATA = self.data if data is None else data  # all data points
        ERR = self.error if error is None else error  # all error for points
        L, dk = self.L, self.dk

        self.xlims = (np.round(X.min()), X.max()) if xlims is None else xlims

        # define window
        self.win = WINDOWS.tukey(self.Lpoints, alpha=0.1, sym=True)

        def calc_spectrum_and_field_apply(stancil):
            """
            windows the data accoding to stencil and applies LS spectrogram
            returns: stancil center, spectrum for this stencil, number of datapoints in stancil
            """

            x_mask = (stancil[0] <= X) & (X <= stancil[-1])

            x = X[x_mask]
            if x.size < 200:  # if there are not enough photos set results to nan
                return stancil[1], self.k * np.nan, self.k * np.nan, np.nan, x.size

            y = DATA[x_mask]

            # make x positions
            x_pos = (np.round((x - stancil[0]) / 10.0 - 1, 0)).astype("int")

            # weight data
            if weight_data:
                window = self.win[x_pos]
                y = y * window * np.sqrt(np.var(y) / np.var((y * window)))

            # make y gridded
            x_model = np.arange(stancil[0], stancil[-1], self.dx)
            y_gridded = np.copy(x_model) * np.nan
            y_gridded[x_pos] = y
            nan_mask = np.isnan(y_gridded)

            err = ERR[x_mask] if ERR is not None else None
            LS_PSD, LS_object = calc_spectrum_LS(
                x, y, self.k, err=err, LS=None, dk=self.dk
            )

            y_model = reconstruct_data_from_LS(LS_object, x_model, self.k)

            P = ConserveVariance(
                np.fft.rfft(y_model), self.k, y_gridded, nan_mask=nan_mask
            )
            P.set_parameters()

            fitter = P.optimize(max_nfev=max_nfev)

            return stancil[1], LS_PSD, P.best_guess_Z(), fitter.params, x.size

        #  derive L2 stancil
        self.stancil_iter = create_chunk_boundaries_unit_lengths(
            L, self.xlims, ov=self.ov, iter_flag=True
        )

        map_func = map if map_func is None else map_func
        print(map_func)
        Spec_returns = list(
            map_func(calc_spectrum_and_field_apply, copy.copy(self.stancil_iter))
        )

        # unpack resutls of the mapping:
        D_specs = dict()
        Y_model = dict()
        Pars = dict()
        N_per_stancil = list()
        for I in Spec_returns:
            D_specs[I[0]] = I[1]
            Y_model[I[0]] = I[2]
            Pars[I[0]] = I[3]
            N_per_stancil.append(I[4])

        self.N_per_stancil = N_per_stancil
        chunk_positions = np.array(list(D_specs.keys()))
        self.N_stancils = len(chunk_positions)  # number of spectral realizatiobs

        # repack data, create xarray
        # 1st LS spectal estimates
        self.spec_name = "LS_spectal_power" if name is None else name
        G_LS_power = dict()
        for xi, I in D_specs.items():
            G_LS_power[xi] = xr.DataArray(
                I, dims=["k"], coords={"k": self.k, "x": xi}, name=self.spec_name
            )

        G_LS_power = xr.concat(G_LS_power.values(), dim="x").T

        # 2nd FFT(Y_model)
        G_fft = dict()
        Y_model_k_fft = np.fft.rfftfreq(int(self.Lpoints), d=self.dx)
        for xi, I in Y_model.items():
            if I.size < Y_model_k_fft.size:
                I = np.insert(I, -1, I[-1])

            G_fft[xi] = xr.DataArray(
                I, dims=["k"], coords={"k": Y_model_k_fft, "x": xi}, name="Y_model_hat"
            )

        G_fft = xr.concat(G_fft.values(), dim="x").T

        # generate power spec as well
        self.G = Z_to_power(G_fft, self.dk, self.Lpoints)
        self.G.name = "spectral_power_optm"

        # merge both datasets
        self.GG = xr.merge([G_LS_power, G_fft, self.G])
        self.GG.attrs["ov"] = self.ov
        self.GG.attrs["L"] = self.L
        self.GG.attrs["Lpoints"] = self.Lpoints
        self.GG.coords["N_per_stancil"] = (("x"), N_per_stancil)

        self.GG.expand_dims(dim="eta")
        self.GG.coords["eta"] = (
            ("eta"),
            np.arange(0, self.L + self.dx, self.dx) - self.L / 2,
        )
        self.GG["win"] = (("eta"), np.insert(self.win, -1, self.win[-1]))

        # create dataframe with fitted parameters
        PP2 = dict()
        for k, I in Pars.items():
            if I is not np.nan:
                PP2[k] = I

        keys = PP2[next(iter(PP2))].keys()
        params_dataframe = pd.DataFrame(index=keys)

        for k, I in PP2.items():
            I.values()
            params_dataframe[k] = list(I.valuesdict().values())

        return self.GG, params_dataframe

    def calc_var(self):

        Gmean = np.nanmean(self.G, 1)
        infmask = np.isinf(Gmean)

        return self.dk * Gmean[~infmask].sum().data

    def parceval(self, add_attrs=True, weight_data=False):
        "test Parceval theorem"

        DATA = self.data
        X = self.x

        def get_stancil_var_apply(stancil):

            "returns the variance of yy for stancil"
            x_mask = (stancil[0] < X) & (X <= stancil[-1])
            idata = DATA[x_mask]
            if len(idata) < 1:
                return stancil[1], np.nan, len(idata)
            idata = detrend(idata)
            # weight data
            x_pos = (np.round((X[x_mask] - stancil[0]) / 10.0, 0)).astype("int")
            if weight_data:
                window = self.win[x_pos]
                idata = (
                    idata * window * np.sqrt(np.var(idata) / np.var((idata * window)))
                )
            return stancil[1], idata.var(), len(idata)

        D_vars = list(map(get_stancil_var_apply, copy.copy(self.stancil_iter)))

        stancil_vars, Nphotons = list(), 0
        for I in D_vars:
            stancil_vars.append(I[1] * I[2])
            Nphotons += I[2]

        stancil_weighted_variance = np.nansum(np.array(stancil_vars)) / Nphotons

        print("Parcevals Theorem:")
        print("variance of timeseries: ", DATA.var())
        print("mean variance of stancils: ", stancil_weighted_variance)
        print("variance of the optimzed windowed LS Spectrum: ", self.calc_var())

        if add_attrs:
            self.G.attrs["variance_unweighted_data"] = DATA.var()
            self.G.attrs["mean_variance_stancils"] = np.nanmean(np.array(stancil_vars))
            self.G.attrs["mean_variance_LS_pwelch_spectrum"] = self.calc_var()

    def mean_spectral_error(self, mask=None, confidence=0.95):
        return WavenumberSpectrogram.mean_spectral_error(
            self, mask=mask, confidence=confidence
        )


# class for getting standard Pwelch spectrum. old version, deprechiate
class WavenumberPwelch:
    def __init__(
        self, data, x, L, ov=None, window=None, save_chunks=False, plot_chunks=False
    ):
        """
        returns a wavenumber spectrum using the pwelch method

        inputs:
        data    data
        x       grid the fft is taken on
        L       window length in number of grid points
        ov      (default=None) number of grid points the windows should overlab
        window  (default=np.hanning) numpy window function
        save_chunks     if True, self.chunks contains all compute chunks
        plot_chunks     if True, it plots all chunks

        returns:
        self.spec_est   mean power spectrum
        self.n_spec
        self.n
        self.dx
        self.n_spec
        """
        # field to be analyzed
        self.data = data
        # sampling interval
        self.dx = np.diff(x)[0]
        self.save_chunks = save_chunks
        dsize = data.size

        ov = int(np.round(L / 2)) if ov is None else ov

        self.n = L
        if window is None:
            win = np.hanning(self.n)
        else:
            win = window

        factor = np.sqrt(self.n / (win**2).sum())
        win *= factor

        # test if n is even
        if self.n % 2:
            self.neven = False
        else:
            self.neven = True
        # calculate freq
        self.k = self.calc_freq()

        nbin = int(np.floor(dsize / (L - ov)))

        if save_chunks:
            chunks = np.empty([int(nbin), int(L)])

        self.specs = np.empty([int(nbin), self.k.size])

        last_k = 0
        k = 0

        for i in np.arange(0, dsize - int(L - ov) + 1, int(L - ov)):

            if (plot_chunks) and (i >= dsize - 6 * int(L - ov)):
                M.figure_axis_xy()

            self.phi = data[int(i) : int(i + L)]

            if int(i + L) <= data.size - 1:
                if save_chunks:
                    chunks[k, :] = self.phi

                self.phi = signal.detrend(self.phi) * win
                if plot_chunks:
                    plt.plot(self.phi)

                self.specs[k, :] = self.calc_spectrum()
                last_k = k
                last_used_TS = int(i + L)

            else:
                if plot_chunks:
                    print("end of TS is reached")
                    print("last spec No: " + str(last_k))
                    print("spec container: " + str(specs.shape))
                    print("last used Timestep: " + str(last_used_TS))
                    print("length of TS " + str(dsize) + "ms")

            k += 1

        if save_chunks:
            self.chunks = chunks

        self.spec_est = self.specs.mean(axis=0)

        self.n_spec, _ = self.specs.shape
        self.calc_var()

    def calc_freq(self):
        """calculate array of spectral variable (frequency or
        wavenumber) in cycles per unit of L"""

        self.df = 1.0 / ((self.n - 1) * self.dx)

        if self.neven:
            f = self.df * np.arange(self.n / 2 + 1)
        else:
            f = self.df * np.arange((self.n - 1) / 2.0 + 1)
        return f

    def calc_spectrum(self):
        """compute the 1d spectrum of a field phi"""

        self.phih = np.fft.rfft(self.phi)

        # the factor of 2 comes from the symmetry of the Fourier coeffs
        spec = 2.0 * (self.phih * self.phih.conj()).real / self.df / self.n**2

        # the zeroth frequency should be counted only once
        spec[0] = spec[0] / 2.0
        if self.neven:
            spec[-1] = spec[-1] / 2.0

        return spec

    def error(self, ci=0.95):
        self.El, self.Eu = spec_error(self.spec_est, self.n_spec, ci=ci)

    def parceval(self):
        print("Parcevals Theorem:")
        print("variance of unweighted timeseries: ", self.data.var())
        print(
            "mean variance of timeseries chunks: ",
            (
                self.chunks.var(axis=1).mean()
                if self.save_chunks is True
                else "data not saved"
            ),
        )
        print("variance of the pwelch Spectrum: ", self.var)

    def calc_var(self):
        """Compute total variance from spectrum"""
        self.var = (
            self.df * np.nanmean(self.specs[1:], 0).sum()
        )  # do not consider zeroth frequency


# optimze spectral variance
class ConserveVariance:
    def __init__(self, Z, freq, data, nan_mask=None):
        """ """

        self.LM = LM
        self.data = data
        self.Z = Z
        self.freq = freq
        self.nan_mask = nan_mask

    def set_parameters(self):

        params = self.LM.Parameters()

        p_smothed = self.runningmean(np.abs(self.Z), 20, tailcopy=True)
        f_max = self.freq[p_smothed[~np.isnan(p_smothed)].argmax()]

        lambda_max = 9.81 * 5**2 / (2 * np.pi)
        params.add("x_cutoff", 1 / lambda_max, min=0, max=1, vary=False)
        params.add(
            "x_max_pos", f_max, min=f_max * 0.75, max=f_max * 5 + 0.001, vary=False
        )
        params.add("LF_amp", 1, min=0.5, max=1.2, vary=True)
        params.add("HF_amp", 0.5, min=0, max=1.5, vary=True)
        params.add("sigma_g", 0.002, min=0.001, max=0.05, vary=False)
        params.add("Gauss_amp", 0.5, min=0.01, max=2, vary=True)

        self.params = params
        return params

    def test_ojective_func(self, weight_func, plot_flag=True):
        self.objective_func(
            self.params,
            self.data,
            self.Z,
            weight_func,
            self.freq,
            self.nan_mask,
            plot_flag=plot_flag,
        )

    def tanh_weight_function(self, ff, params):
        return self.tanh_weight(
            ff,
            params["x_cutoff"].value,
            params["x_max_pos"].value,
            params["LF_amp"].value,
            params["HF_amp"].value,
            params["Gauss_amp"].value,
            params["sigma_g"].value,
        )

    def tanh_weight(self, x, x_cutoff, x_max_pos, LF_amp, HF_amp, Gauss_amp, sigma_g):
        """
        zdgfsg
        """
        HF_amp1 = LF_amp - HF_amp
        decay = 0.5 - np.tanh((x - x_cutoff) / sigma_g) / 2
        y = decay * HF_amp1 + (1 - HF_amp1)
        y = y - y[0] + LF_amp

        def gaus(x, x_0, amp, sigma_g):
            return amp * np.exp(-0.5 * ((x - x_0) / sigma_g) ** 2)

        y += gaus(x, x_max_pos, Gauss_amp, sigma_g)

        return y

    def objective_func(
        self,
        params,
        data_x,
        Z_results,
        weight_func,
        freq,
        nan_mask=None,
        plot_flag=False,
    ):

        alpha = 1e7

        def model_real_space(Z, weights, n=None):
            """
            Both inputs must have the same length
            """
            return np.fft.irfft(Z * weights, n=n)

        weights = weight_func(freq, params)

        if Z_results.size > weights.size:
            weights = np.insert(weights, -1, weights[-1])

        if nan_mask is not None:
            model = model_real_space(Z_results, weights, n=data_x.size)[~nan_mask]
            dd = data_x[~nan_mask][:]
        else:
            model = model_real_space(Z_results, weights, n=data_x.size)[:]
            dd = data_x[:]

        if model.size > dd.size:
            model = model[:-1]
        elif model.size < dd.size:
            dd = dd[:-1]

        if plot_flag:

            F = M.figure_axis_xy(10, 4.1 * 2.5, view_scale=0.5, container=True)

            gs = GridSpec(5, 1, wspace=0.1, hspace=0.4)
            pos0, pos1, pos2 = gs[0:3, 0], gs[3, 0], gs[4, 0]
            ax1 = F.fig.add_subplot(pos0)
            plt.title("Stacked Timeseries", loc="left")

            chunk_l = 400
            chunk_iter = create_chunk_boundaries(
                chunk_l, data_x.size, ov=0, iter_flag=True
            )

            ofsett0 = 6
            ofsett = np.copy(ofsett0)
            for chi in chunk_iter:

                v1 = np.round(np.nanvar(dd), 4)
                plt.plot(
                    ofsett + data_x[chi[0] : chi[-1]],
                    linewidth=3,
                    alpha=0.5,
                    c="black",
                    label=" org. data (var:" + str(v1) + ")",
                )

                v1 = np.round(
                    model_real_space(Z_results, weights * 0 + 1)[~nan_mask[1:]].var(), 4
                )
                plt.plot(
                    ofsett
                    + model_real_space(Z_results, weights * 0 + 1)[chi[0] : chi[-1]],
                    linewidth=0.8,
                    c="red",
                    label="LS model init (var:" + str(v1) + ")",
                )

                v1 = np.round(model.var(), 4)
                plt.plot(
                    ofsett + model_real_space(Z_results, weights)[chi[0] : chi[-1]],
                    linewidth=0.8,
                    c="blue",
                    label="LS model weighted (var:" + str(v1) + ")",
                )

                if ofsett == ofsett0:
                    plt.legend()
                ofsett -= 1

            plt.ylim(ofsett, ofsett0 + 1)
            plt.xlim(0, chunk_l * 2)

            ax2 = F.fig.add_subplot(pos1)
            plt.title("Amplitude Weight Function", loc="left")
            plt.plot(weights, c="black")
            ax2.set_xscale("log")

            ax3 = F.fig.add_subplot(pos2)
            plt.title("Initial and tuned |Z|", loc="left")

            v2 = np.round(
                (4.0 * (Z_results * Z_results.conj()).real / data_x.size**2).sum(), 4
            )
            plt.plot(
                abs(Z_results), linewidth=0.8, c="red", label="Z (var: " + str(v2) + ")"
            )
            plt.plot(
                M.runningmean(abs(Z_results), 20, tailcopy=True),
                linewidth=1.5,
                c="red",
                zorder=12,
            )

            Z2 = Z_results * weights
            v2 = np.round((4.0 * (Z2 * Z2.conj()).real / data_x.size**2).sum(), 4)
            plt.plot(
                abs(Z2),
                linewidth=0.8,
                c="blue",
                label="weighted Z(var: " + str(v2) + ")",
            )
            plt.plot(
                M.runningmean(abs(Z2), 20, tailcopy=True),
                linewidth=1.5,
                c="blue",
                zorder=12,
            )
            plt.legend()

            plt.ylim(np.percentile(abs(Z_results), 0.5), abs(Z_results).max() * 1.3)
            plt.xlabel("wavenumber k")
            ax3.set_xscale("log")
            ax3.set_yscale("log")

        fitting_cost = (abs(dd - model) / dd.std()) ** 2
        variance_cost = (abs(dd.var() - model.var()) / dd.std()) ** 2

        return fitting_cost.sum(), alpha * variance_cost

    def optimize(self, fitting_args=None, method="dual_annealing", max_nfev=None):

        if fitting_args is None:
            fitting_args = (self.data, self.Z, self.tanh_weight_function, self.freq)

        self.weight_func = fitting_args[2]
        self.fitter = self.LM.minimize(
            self.objective_func,
            self.params,
            args=fitting_args,
            kws={"nan_mask": self.nan_mask},
            method=method,
            max_nfev=max_nfev,
        )
        return self.fitter

    def plot_result(self):
        self.objective_func(
            self.fitter.params,
            self.data,
            self.Z,
            self.weight_func,
            self.freq,
            self.nan_mask,
            plot_flag=True,
        )

    def best_guess_Z(self):
        return self.Z * self.weight_func(self.freq, self.fitter.params)

    def runningmean(self, var, m, tailcopy=False):
        m = int(m)
        s = var.shape
        if s[0] <= 2 * m:
            print("0 Dimension is smaller then averaging length")
            return
        rr = np.asarray(var) * np.nan
        var_range = np.arange(m, int(s[0]) - m - 1, 1)
        for i in var_range[np.isfinite(var[m : int(s[0]) - m - 1])]:
            rr[int(i)] = np.nanmean(var[i - m : i + m])
        if tailcopy:
            rr[0:m] = rr[m + 1]
            rr[-m - 1 : -1] = rr[-m - 2]

        return rr
