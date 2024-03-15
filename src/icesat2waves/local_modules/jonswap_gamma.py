import logging
from scipy.constants import g

from lmfit import minimize, Parameters
import copy
import matplotlib.pyplot as plt
import numpy as np

_logger = logging.getLogger(__name__)


#
def normalize_time(time):
    """
    returns a time vector from 0 to 1 of the length of time
        and the time step dt in dimentional units
    """
    time = np.copy(time)
    dt = np.diff(time).mean()
    time = (time - time[0]) / dt
    return (time) / (time[-1]), dt


def gaussian(x, x0, sigma):
    return np.exp(-np.power((x - x0) / sigma, 2.0) / 2.0)


#  generate some data_ano
if __name__ == "__main__":
    from scipy.stats import gamma

    f = np.arange(1 / 1000.0, 0.2, 0.001)
    time = np.arange(0, 1, 0.001)

    tt, ff = np.meshgrid(time, f)
    fake_data = np.sin(tt * np.pi) ** 4 * gaussian(ff, 0.05, 0.03)

    plt.contour(tt, ff, fake_data, colors="k")


#  basic functions
def gamma_time_normlized_amp(time, gammapar=2, loc=0.2, scale=0.1):
    from scipy.stats import gamma

    np.seterr(divide="ignore", invalid="ignore")
    """
    configured for normalized time scale (0, 1)
    gammapar > 1
    time > 0
    loc is position 0 ..1
    scale  >0 .. \approx 0.3
    """
    gamma_mod = gamma(gammapar, loc=loc, scale=scale).pdf(time)
    return gamma_mod / gamma_mod.max()


def gamma_time_normlized_amp_shifted(time, gammapar=2, loc=0.2, scale=0.1):
    from scipy.stats import gamma

    np.seterr(divide="ignore", invalid="ignore")
    """

    configured for normalized time scale (0, 1)
    returns a gamam distribution with a maximum of 1 at loc.
    gammapar > 1
    time > 0
    loc is position 0 ..1
    scale  >0 .. \approx 0.3
    """
    gamma_mod = gamma(gammapar, loc=loc[0, :], scale=scale).pdf(time[0, :])
    dloc = loc[0, :] - time[0, :][gamma_mod.argmax()]
    gamma_mod = gamma(gammapar, loc=loc + dloc, scale=scale).pdf(time)
    return gamma_mod / gamma_mod.max()


if __name__ == "__main__":
    plt.plot(time, gamma_time_normlized_amp(time))
    plt.plot(
        time, gamma_time_normlized_amp_shifted(time, gammapar=3, loc=0.2, scale=0.1)
    )


# JONSWAP


def JONSWAP_bulk(f, floc=0.04, famp=1e-2, gamma=3.3, peak_std=1e-1):
    """
    see Ocean Surface waves - S. R. Massel eq.3.69 and eq.3.81

    """
    B = 0.74

    w = f * 2 * np.pi
    wp = floc * 2 * np.pi
    stretch = 5 / 4

    alpha = famp

    delta = np.exp(-((w - wp) ** 2) / (2 * peak_std**2 * wp**2))
    peak_factor = gamma**delta

    # units of m^2 / Hz
    return alpha * w ** (-5) * np.exp(-stretch * (w / wp) ** -4) * peak_factor


def pierson_moskowitz_default(f, U):
    """
    see Ocean Surface waves - S. R. Massel eq.3.79 and eq.3.80

    """

    wp = 0.879 * g / U
    w = 2.0 * np.pi * f
    sigma = 0.04 * g / wp**2.0
    alpha = 5.0 * (wp**2.0 * sigma / g) ** 2.0

    return alpha * w ** (-5.0) * g**2.0 * np.exp(-5.0 / 4.0 * (w / wp) ** -4)  #


def pierson_moskowitz_fetch_limit(f, X, U):
    """
    see Ocean Surface waves - S. R. Massel eq.3.81 - eq.3.84

    """

    w = 2.0 * np.pi * f  # rad/sec

    # dimensionless
    alpha = 0.076 * (g * X / U**2) ** (-0.22)

    wp = 7.0 * np.pi * (g / U) * (g * X / U**2) ** (-0.33)
    _logger.debug("wp=%s", wp)

    sigma_p = 0.07
    sigma_pp = 0.09
    sigma = np.array([[sigma_p if i else sigma_pp][0] for i in list(w <= wp)])

    gamma = 3.3
    delta = np.exp(-((w - wp) ** 2) / (2.0 * sigma**2.0 * wp**2.0))
    peak_factor = gamma**delta

    # Hz**-5 m**2 /s**4 = m**2 sec
    return alpha * g**2.0 * w ** (-5.0) * np.exp(-5.0 / 4.0 * (w / wp) ** -4)


def JONSWAP_default_alt(f, f_max, U, gamma=3.3):
    return JONSWAP_default(f, X(f_max, U), U, gamma=gamma)


def JONSWAP_default(f, X, U, gamma=3.3):
    """
    see Ocean Surface waves - S. R. Massel eq.3.81 - eq.3.84
    inputs:
    X         fetch lenght in meters about 5e5,
    U          wind speed in m/s about 15 m/s
    gamma is defined in the JONSWAP spectrum as 3.3. However, The JONSWAP spectrum is a psectrum for a fully deleoped sea, and likely not observed in a remote location.
    varying gamma to less then its JONSWAP value will mimic the attenuation of the peak with travel time to the limit of gamma=1, which is the PM spectrums
    """

    w = 2.0 * np.pi * f  # rad/sec

    # dimensionless
    alpha = 0.076 * (g * X / U**2) ** (-0.22)
    wp = 7.0 * np.pi * (g / U) * (g * X / U**2) ** (-0.33)

    sigma_p = 0.07
    sigma_pp = 0.09
    sigma = np.array([[sigma_p if i else sigma_pp][0] for i in list(w <= wp)])

    delta = np.exp(-((w - wp) ** 2) / (2.0 * sigma**2.0 * wp**2.0))
    peak_factor = gamma**delta

    return (
        alpha * g**2.0 * w ** (-5.0) * np.exp(-5.0 / 4.0 * (w / wp) ** -4) * peak_factor
    )  # Hz**-5 m**2 /s**4 = m**2 sec


""" add function for X_tilde(X, U10), alpha(f_max, U10) and f_max(U10, X_tilde) or f_max(U, X)"""


def X_tilde(X, U10):
    return g * X / U10**2


def alpha(f_max, U10):
    return 0.033 * (f_max * U10 / g) ** 0.67


def f_max(U10, X):
    return 3.5 * g / U10 * X_tilde(X, U10) ** (-0.33)


def X(f_max, U10):
    return 3.5**3 * g**2 / (U10 * f_max**3)


if __name__ == "__main__":
    M.figure_axis_xy()
    plt.plot(f, JONSWAP_bulk(f, gamma=3.3), label="JONSWAP bulk (RIS version)")
    plt.plot(f, JONSWAP_bulk(f, gamma=1), label="JONSWAP bulk --> PM (RIS version)")

    plt.plot(f, pierson_moskowitz_default(f, 15), label="PM default")
    plt.plot(f, pierson_moskowitz_fetch_limit(f, 5e5, 15), label="PM fetch limited")
    plt.plot(f, JONSWAP_default(f, 5e5, 15, 3.3), label="JONSWAP default")
    plt.legend()


def gamma_time_JONSWAP_default(
    time, f, slope_t, intersectT, tgammapar, tscale, f_max, U10, gamma_peak, plot=False
):
    """
    This method calculated a 2D shaped function given the parameters:
    inputs:

    time        normalized time [0, 1] np.array
    f           frequency in the swell band, np.array,

    slope_t     slope of the "dispersed peak frequencies" df/dt [Hz/ normalized time]
    intersectT  intersect of that line in units of normalized time

    tgammapar   gamma parameter of the gamma function in time
    tscale      scaling parameter of the gamma function

    f_max       =0.04, location of the peak frequency on the JONSWAP spectrum
    U10           Wind speed that generates the waves
    gamma_peak  =3.3,  gamma peak parameter of the JONSWAP spectrum

    ##amplitude   amplitude of the whole function.  if =1 , peak amplitude corresponds to JONSWAPs values

    plot        True, False. Simple plot of the output function

    return:
                2d function with the shape of [time,freq]
    """

    intersectF = -intersectT * slope_t
    pfreq = time * slope_t + intersectF

    slopeF = 1 / slope_t
    pfreq_forgamma = f * slopeF + intersectT

    tt, line = np.meshgrid(time, pfreq_forgamma)

    func_t = gamma_time_normlized_amp_shifted(
        tt, gammapar=tgammapar, loc=line, scale=tscale
    )

    """ Define X(f_max and U) here """

    def X(f_max, U10):
        return 3.5**3 * g**2 / (U10 * f_max**3)

    func_freq_temp = JONSWAP_default(f, X=X(f_max, U10), U=U10, gamma=gamma_peak)

    tt, func_freq = np.meshgrid(time, func_freq_temp)
    tt3, ff3 = np.meshgrid(time, func_freq_temp)

    if plot:
        plt.subplot(3, 1, 1)
        plt.contourf(func_t)

        plt.subplot(3, 1, 2)

        plt.contourf(func_freq)

        plt.subplot(3, 1, 3)

        plt.plot(time, pfreq)
        plt.contourf((func_t * func_freq).reshape(tt.shape))

    return (func_t * func_freq).T


def gamma_time_JONSWAP_nondim(
    time, f, slope_t, intersectT, tgammapar, tscale, f_max, U10, gamma_peak, plot=False
):
    """
    This method calculated a 2D shaped function given the parameters:
    the JONSWAP function is non-dimensionalized to work with normalized data
    inputs:

    time        normalized time [0, 1] np.array
    f           frequency in the swell band, np.array,

    slope_t     slope of the "dispersed peak frequencies" df/dt [Hz/ normalized time]
    intersectT  intersect of that line in units of normalized time

    tgammapar   gamma parameter of the gamma function in time
    tscale      scaling parameter of the gamma function

    f_max       =0.04, location of the peak frequency on the JONSWAP spectrum
    U10           Wind speed that generates the waves
    gamma_peak  =3.3,  gamma peak parameter of the JONSWAP spectrum

    ##amplitude   amplitude of the whole function.  if =1 , peak amplitude corresponds to JONSWAPs values

    plot        True, False. Simple plot of the output function

    return:
                2d nondim function with the shape of [time,freq]
    """

    intersectF = -intersectT * slope_t
    pfreq = time * slope_t + intersectF

    slopeF = 1 / slope_t
    pfreq_forgamma = f * slopeF + intersectT

    tt, line = np.meshgrid(time, pfreq_forgamma)

    func_t = gamma_time_normlized_amp_shifted(
        tt, gammapar=tgammapar, loc=line, scale=tscale
    )

    def X(f_max, U10):
        return 3.5**3.0 * g**2.0 / (U10 * f_max**3.0)

    func_freq_temp = JONSWAP_default(f, X=X(f_max, U10), U=U10, gamma=gamma_peak) * (
        g**2.0 / U10**4.0
    )

    tt, func_freq = np.meshgrid(time, func_freq_temp)
    tt3, ff3 = np.meshgrid(time, func_freq_temp)

    if plot:
        plt.subplot(3, 1, 1)
        plt.contourf(func_t)

        plt.subplot(3, 1, 2)

        plt.contourf(func_freq)

        plt.subplot(3, 1, 3)

        plt.plot(time, pfreq)
        plt.contourf((func_t * func_freq).reshape(tt.shape))

    return (func_t * func_freq).T


#  build residual
def residual_JONSWAP_default_gamma(
    value_dict, time, f, data=None, weight=None, eps=None, plot_flag=False
):
    """
    derived the residual between model and data given params_local.

    inputs:
    value_dict         dictionary with parameters, or lmfit.parameters instance.
                       contains all parameters that are needed for creating the model.
    time               normalized time axis
    f                  frequency axis
    data               data, same shape as time and f, if None function returns model as a 1d vector.
    weight             weigthing for each of the data points, can be a 1d or 2d Vector, must have the same size as data, if None, no weighting is applied
    eps                is just a dummy
    """
    from lmfit import Parameters
    from collections import OrderedDict

    if type(value_dict) is Parameters:
        vd = value_dict.valuesdict()
    elif (type(value_dict) is dict) | (type(value_dict) is OrderedDict):
        vd = value_dict
    else:
        raise ValueError("value_dict is eiher a dicitionary or a Params instance")

    r0 = vd["DT"] * g / (4 * np.pi * vd["slope"])
    nue = vd["attenuation_nue"]
    attenuation = np.exp(-r0 * nue)
    model = attenuation * gamma_time_JONSWAP_default(
        time,
        f,
        vd["slope"],
        vd["intersect"],
        vd["tgammapar"],
        vd["tscale"],
        vd["f_max"],
        vd["U10"],
        vd["gamma_peak"],
        plot=plot_flag,
    )

    model1d = model.reshape(model.shape[0] * model.shape[1])
    model1d[np.isnan(model1d)] = 0

    if data is not None:
        if np.size(data.shape) != 1:
            if model.shape == data.shape:
                data1d = data.reshape(data.shape[0] * data.shape[1])
                nan_track = np.isnan(data1d)
            elif model.shape == data.T.shape:
                data1d = data.T.reshape(data.T.shape[0] * data.T.shape[1])
                nan_track = np.isnan(data1d)
            else:
                raise TypeError("data shape does not match")

    if weight is not None:
        if (len(weight.shape) == 1) & (model1d.size == weight.size):
            weight1d = weight
        elif (len(weight.shape) == 2) & (model1d.size == weight.size):
            weight1d = weight.reshape(weight.shape[0] * weight.shape[1]).T
        else:
            raise ValueError(
                "weight has not the same dimensions as model. \n"
                + "data "
                + str(model.shape)
                + "\n weight "
                + str(weight.shape)
            )

    if data is None:
        return model1d
    if weight is not None:

        d = (model1d - data1d) * weight1d
        d[nan_track] = np.nan
        return d
    if (weight is None) and (data is not None):
        d = model1d - data1d
        d[nan_track] = np.nan
        return d


def residual_JONSWAP_nondim_gamma(
    value_dict, time, f, data=None, weight=None, eps=None, plot_flag=False
):
    """
    derived the residual between model and data given params_local.

    inputs:
    value_dict         dictionary with parameters, or lmfit.parameters instance.
                       contains all parameters that are needed for creating the model.
    time               normalized time axis
    f                  frequency axis
    data               data, same shape as time and f, if None function returns model as a 1d vector.
    weight             weigthing for each of the data points, can be a 1d or 2d Vector, must have the same size as data, if None, no weighting is applied
    eps                is just a dummy
    """
    from lmfit import Parameters
    from collections import OrderedDict

    if type(value_dict) is Parameters:
        vd = value_dict.valuesdict()
    elif (type(value_dict) is dict) | (type(value_dict) is OrderedDict):
        vd = value_dict
    else:
        raise ValueError("value_dict is eiher a dicitionary or a Params instance")

    r0 = vd["DT"] * g / (4 * np.pi * vd["slope"])
    nue = vd["attenuation_nue"]
    attenuation = np.exp(-r0 * nue)
    model = attenuation * gamma_time_JONSWAP_nondim(
        time,
        f,
        vd["slope"],
        vd["intersect"],
        vd["tgammapar"],
        vd["tscale"],
        vd["f_max"],
        vd["U10"],
        vd["gamma_peak"],
        plot=plot_flag,
    )

    model1d = model.reshape(model.shape[0] * model.shape[1])

    if data is not None:
        if np.size(data.shape) != 1:
            if model.shape == data.shape:
                data1d = data.reshape(data.shape[0] * data.shape[1])
                nan_track = np.isnan(data1d)
            elif model.shape == data.T.shape:
                data1d = data.T.reshape(data.T.shape[0] * data.T.shape[1])
                nan_track = np.isnan(data1d)
            else:
                raise TypeError("data shape does not match")

    if weight is not None:
        if (len(weight.shape) == 1) & (model1d.size == weight.size):
            weight1d = weight
        elif (len(weight.shape) == 2) & (model1d.size == weight.size):
            weight1d = weight.reshape(weight.shape[0] * weight.shape[1]).T
        else:
            raise ValueError(
                "weight has not the same dimensions as model. \n"
                + "data "
                + str(model.shape)
                + "\n weight "
                + str(weight.shape)
            )

    if data is None:
        return model1d
    if weight is not None:

        d = (model1d - data1d) * weight1d
        d[nan_track] = np.nan
        return d
    if (weight is None) and (data is not None):
        d = model1d - data1d
        d[nan_track] = np.nan
        return d


def gamma_from_params(value_dict, time, plot_flag=False):
    from lmfit import Parameters
    from collections import OrderedDict

    if type(value_dict) is Parameters:
        vd = value_dict.valuesdict()
    elif (type(value_dict) is dict) | (type(value_dict) is OrderedDict):
        vd = value_dict
    else:
        raise ValueError("value_dict is eiher a dicitionary or a Params instance")

    spec = JONSWAP_default(f, X(vd["f_max"], vd["U10"]), vd["U10"], vd["gamma_peak"])
    func_t = gamma_time_normlized_amp_shifted(
        time, gammapar=vd["tgammapar"], loc=0.5, scale=vd["tgammapar"]
    )

    if plot_flag:
        plt.plot(f, spec, label="JONSWAP default")

    return spec


def JONSWAP_default_from_params(value_dict, f, plot_flag=False):
    from lmfit import Parameters
    from collections import OrderedDict

    if type(value_dict) is Parameters:
        vd = value_dict.valuesdict()
    elif (type(value_dict) is dict) | (type(value_dict) is OrderedDict):
        vd = value_dict
    else:
        raise ValueError("value_dict is eiher a dicitionary or a Params instance")

    spec = JONSWAP_default(f, X(vd["f_max"], vd["U10"]), vd["U10"], vd["gamma_peak"])
    if plot_flag:
        plt.plot(f, spec, label="JONSWAP default")

    return spec


def JONSWAP_nondim_from_params(value_dict, f, plot_flag=False):
    from lmfit import Parameters
    from collections import OrderedDict

    if type(value_dict) is Parameters:
        vd = value_dict.valuesdict()
    elif (type(value_dict) is dict) | (type(value_dict) is OrderedDict):
        vd = value_dict
    else:
        raise ValueError("value_dict is eiher a dicitionary or a Params instance")

    spec = JONSWAP_default(
        f, X(vd["f_max"], vd["U10"]), vd["U10"], vd["gamma_peak"]
    ) * (g**2 / vd["U10"] ** 4)
    if plot_flag:
        plt.plot(f, spec, label="JONSWAP default non-dim")

    return spec


if __name__ == "__main__":
    # http://cars9.uchicago.edu/software/python/lmfit/fitting.html
    params = Parameters()

    params.add("slope", value=slope0, min=slope0 * 0.1, max=slope0 * 10)
    params.add("intersect", value=intersect0, min=-0.5, max=0.5)

    params.add("tgammapar", value=tgammapar0, min=0.0001, max=4)
    params.add("tscale", value=tscale0, min=0, max=0.1)

    params.add("f_max", value=0.01, min=0.0, max=0.1)
    params.add("U10", value=10, min=2, max=25)
    params.add("gamma_peak", value=3.3, min=0.1, max=4, vary=False)

    params.add("amp", value=1, min=1e-4, max=1e2)

if __name__ == "__main__":
    # should return model:
    model1d = residual_JONSWAP_default_gamma(params, time, f)
    M.figure_axis_xy(3, 6)

    plt.subplot(2, 1, 1)
    plt.plot(model1d)

    plt.subplot(2, 1, 2)
    plt.contourf(time, f, model1d.reshape(time.size, f.size).T)

    #  should return the residual
    resid1d = residual_JONSWAP_default_gamma(params, time, f, data=fake_data)

    M.figure_axis_xy(3, 6)

    plt.subplot(2, 1, 1)
    plt.plot(resid1d)

    plt.subplot(2, 1, 2)
    plt.contour(time, f, fake_data, colors="k")
    plt.contourf(time, f, resid1d.reshape(time.size, f.size).T)

    # should return the residual weighted residual
    weightdummy = fake_data * 0 + 2
    len(weightdummy.shape)
    weightdummy.size
    len(fake_data.shape)
    fake_data.size
    weightdummy.shape

    weightdummy = 100.0 * np.random.random(fake_data.shape)
    resid1d_weight = residual_JONSWAP_default_gamma(
        params, time, f, data=fake_data, weight=weightdummy
    )

    M.figure_axis_xy(3, 6)

    plt.subplot(2, 1, 1)
    plt.plot(resid1d)
    plt.plot(resid1d_weight, alpha=0.4)

    plt.subplot(2, 1, 2)
    plt.contour(time, f, fake_data, colors="k")
    plt.contourf(time, f, resid1d_weight.reshape(time.size, f.size).T)


#


def Jm_regulizer(value_dict, prior):
    """
    returns a Model cost function as list. each item is the cost for each prior given the parameter value in value_dict

    value_dict  is a dict with all values that need to be regulized, can be standard dict or Parameter instance
    prior       is a dict with all priors for this dict

    """

    from lmfit import Parameters
    from collections import OrderedDict

    if type(value_dict) is Parameters:
        vd = value_dict.valuesdict()
    elif (type(value_dict) is dict) | (type(value_dict) is OrderedDict):
        vd = value_dict
    else:
        raise ValueError("value_dict is eiher a dicitionary or a Params instance")

    Jm = list()
    for k, I in prior.items():
        if type(I["m_err"]) is float:
            Jm.append((I["m0"] - vd[k]) / I["m_err"])
        else:
            if value_dict[k] >= I["m0"]:
                Jm.append((I["m0"] - vd[k]) / I["m_err"][1])
            else:
                Jm.append((I["m0"] - vd[k]) / I["m_err"][0])
    return Jm


def Jm_dimensional_regulizer(value_dict, prior, prior_percentage=True):
    """
    returns a Model cost function as list. each item is the cost for each prior given the parameter value in value_dict

    value_dict  is a dict with all values that need to be regulized, can be standard dict or Parameter instance
    prior       is a dict with all priors for this dict. priors should contain m_err.

    prior_percentage = True
                m_err is the deviation of the value (value dict) from the inital value (prior) as
                a fraction at which the regularization should return a cost of 1.
    prior_percentage = False
                m_err is the deviation of the value (value dict) from the inital value (prior)
                in units of of the parameter at which the regularization should return a cost of 1.

    """

    from lmfit import Parameters
    from collections import OrderedDict

    if type(value_dict) is Parameters:
        vd = value_dict.valuesdict()
    elif (type(value_dict) is dict) | (type(value_dict) is OrderedDict):
        vd = value_dict
    else:
        raise ValueError("value_dict is eiher a dicitionary or a Params instance")

    Jm = list()

    if prior_percentage:

        def reg_func(p0, pi, p_err):
            return (pi / p0 - 1) / p_err

    else:

        def reg_func(p0, pi, p_err_unit):
            ppercent = p_err_unit / p0
            return (pi / p0 - 1) / ppercent

    for k, I in prior.items():
        if type(I["m_err"]) is float:
            Jm.append(reg_func(I["m0"], vd[k], I["m_err"]))

        else:
            if value_dict[k] >= I["m0"]:
                Jm.append(reg_func(I["m0"], vd[k], I["m_err"]))
            else:
                Jm.append(reg_func(I["m0"], vd[k], I["m_err"]))
    return Jm


if __name__ == "__main__":
    # test Jm_regulizer with priors
    # create face priors
    prior_errors = {
        "slope": 1.0,
        "intersect": 0.2,
        "tgammapar": 0.4,
        "tscale": 0.2,
        "f_max": 0.01,
        "amp": 10.0,
    }

    priors = dict()
    for k, I in prior_errors.items():
        priors[k] = {"m_err": I, "m0": params[k].value}

    # fake parameters
    vd = copy.copy(params.valuesdict())
    for k, I in vd.items():
        vd[k] = I * np.random.rand()

    Jm = Jm_regulizer(vd, priors)
    _logger.debug("Jm from regulizer: %s", Jm)


def cost(value_dict, time, f, data=None, weight=None, prior=False, eps=None):
    """
    Wrapper around residual and regulizer.

    returns 1d cost vector

    eps is just a dummy
    """
    from lmfit import Parameters

    Jd = residual_JONSWAP_default_gamma(value_dict, time, f, data=data, weight=weight)

    if prior is not False:

        Jm = Jm_regulizer(value_dict, prior)
        return np.concatenate((Jd, Jm))
    else:
        return Jd


#  test cost
if __name__ == "__main__":
    cost1d_weight = cost(params, time, f, data=fake_data, weight=None, prior=False)

    M.figure_axis_xy(3, 6)

    plt.subplot(2, 1, 1)
    plt.plot(resid1d)
    plt.plot(resid1d_weight, alpha=0.4)

    plt.subplot(2, 1, 2)
    plt.contour(time, f, fake_data, colors="k")
    plt.contourf(time, f, cost1d_weight.reshape(time.size, f.size).T)

    cost1d_weight = cost(vd, time, f, data=fake_data, weight=None, prior=priors)
