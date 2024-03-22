import numpy as np
import scipy.signal as signal
from scipy.ndimage import convolve
import scipy.ndimage.filters as filters


def lanczos_1d(width, dx, a=2):
    """
    This is a 1D lanczos Filter for time series analysis.
    it generates the Filter to be convolved with the timeseries
    https://en.wikipedia.org/wiki/Lanczos_resampling
    inputs:
    width  width of the filter in units of the timeseries
    a      Lanczos parameter (default =2). the length of the filter is a*width
    dx     delta x of the to be filtered timeseries

    returns:
    L       Lanczos Filter with the length a*width and dx.

    """
    # width= 2   # width of the filter in units of the timeseries
    # a=     1   # Lanczos parameter. the length of the filter is a*width
    # dx=   .1   # deltax of the to be filtered timeseries

    r = width / 2.0
    xl = a * r
    x = np.arange(-xl, xl, dx)
    xprime = x / r

    # define the filter
    L = np.sinc(xprime) * np.sinc(xprime / a)
    L = np.where((xprime > -a) & (xprime < a), L, 0)

    return x, L / L.sum()


def lanczos_2d(width, dx, a=2):
    x, L = lanczos_1d(width, dx, a=a)
    L2d = np.outer(L, L.T)

    return x, L2d


def lanczos_filter_1d(x, data, width, a=2, mode="same", method="direct"):
    """
    colvolves the lanzcos filter with data.
    inputs
    x       independent variaable, dimension for data
    data    to be smoothed data, same dimensions a x
    width   width of the lanzos filter in dimensions of x
    a       lanzcos parameters. default 2. Integer.

    mode    passed to signal.convolve() 'full', 'valid','same'
    method  'direct', 'fft', 'auto'

    returns
    data_lp low-passed data, same size as before.
    """

    dx = np.diff(x).mean()
    x, L = lanczos_1d(width, dx, a=a)
    _method = method
    data_lp = signal.convolve(data, L, mode=mode, method=_method)

    return data_lp


def lanczos_filter_1d_wrapping(x, data, width, a=2, mode="wrap"):
    """
    colvolves the lanzcos filter with data.
    same as lanczos_filter_1d but can wrap around

    inputs
    x       independent variaable, dimension for data
    data    to be smoothed data, same dimensions a x
    width   width of the lanzos filter in dimensions of x
    a       lanzcos parameters. default 2. Integer.

    mode    passed to signal.convolve() 'full', 'valid','same'
    method  'direct', 'fft', 'auto'

    returns
    data_lp low-passed data, same size as before.
    """

    dx = np.diff(x).mean()
    x, L = lanczos_1d(width, dx, a=a)

    data_lp = convolve(data, L, mode=mode)  # *

    return data_lp


def lanczos_filter_2d(x, data, width, a=2, mode="same"):
    """
    colvolves the lanzcos filter with data in 3 dimensions.
    inputs
    x       independent variable, dimension for data. the resolution of data varies in direction, choose a convinient x
    data    to be smoothed data, same dimensions a x
    width   width of the lanzos filter in dimensions of x
    a       lanzcos parameters. default 2. Integer.

    mode    passed to signal.convolve() 'full', 'valid','same'
    method  'direct', 'fft', 'auto'

    returns
    data_lp low-passed data, same size as before.
    """

    dx = abs(np.diff(x).mean())
    x, L2d = lanczos_2d(width, dx, a=a)

    data_lp = filters.convolve(data, L2d, mode=mode)

    return data_lp


def lanczos_filter_2d_apply(data, x, width, a=2, mode="same"):
    return lanczos_filter_2d(x, data, width, a=a, mode=mode)


def lanczos_3d(width, dx, a=2):
    x, L = lanczos_1d(width, dx, a=a)
    L2d = np.outer(L, L.T)
    L3d = np.multiply.outer(L2d, L.T)

    return x, L3d


def lanczos_filter_3d(x, data, width, a=2, mode="same"):
    """
    colvolves the lanzcos filter with data in 3 dimensions.
    inputs
    x       independent variable, dimension for data. the resolution of data varies in direction, choose a convinient x
    data    to be smoothed data, same dimensions a x
    width   width of the lanzos filter in dimensions of x
    a       lanzcos parameters. default 2. Integer.

    mode    passed to signal.convolve() 'full', 'valid','same'
    method  'direct', 'fft', 'auto'

    returns
    data_lp low-passed data, same size as before.
    """

    dx = abs(np.diff(x).mean())
    x, L3d = lanczos_3d(width, dx, a=a)

    data_lp = filters.convolve(data, L3d, mode=mode)

    return data_lp
