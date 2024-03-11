import numpy as np
import matplotlib.pyplot as plt
from scipy import signal


def spicke_remover(data, nstd=20.0, spreed=500.0, max_loops=10.0, verbose=False):
    """
    This function removes spikes from timeseries based on its std and maximum values
    inputs:
    data     timeseries as1d arrays
    nstd     number of standard deviations that have to be exceeded by the maximum data value
    spreed   width of data between wich will be linear interpolated around the spike. width is in units of data points (dt)
    max_loops number of maximum possible loobs until the spike removal is stopped, even it the first creteria is not true
    """
    datastd = np.nanstd(data)
    data2 = np.copy(data)

    peak_remove = True
    looper_count = 0
    act_flag = False
    while peak_remove is True:

        if nstd * datastd < np.nanmax(np.abs(data2)):
            act_flag = True
            if verbose:
                print(
                    "true: "
                    + str(nstd * datastd)
                    + " < "
                    + str(np.nanmax(np.abs(data)))
                )
            data2 = spickes_to_mean(data2, nloop=0, spreed=spreed, gaussian=False)
            looper_count += 1
        else:
            if verbose:
                print(
                    "False: "
                    + str(nstd * datastd)
                    + " > "
                    + str(np.nanmax(np.abs(data)))
                )
            peak_remove = False

        if looper_count > max_loops:
            peak_remove = False
            if verbose:
                print("stoped by max#")

    if verbose:

        plt.plot(data, "r")
        plt.plot(data2, "b")

    return data2, act_flag


def spickes_to_mean(ts, nloop=None, spreed=1, gaussian=True):

    nloop = 0 if nloop is None else nloop
    i = 0
    tsmean = ts.mean()
    b = 2 * spreed
    gaus = signal.gaussian(b, std=b / 10)
    while i <= nloop:
        tsabs = np.abs(ts)
        tmax = np.nanmax(tsabs)
        pa = np.where(tsabs == tmax)[0][0]

        if gaussian:
            tsm = np.mean([ts[pa - spreed], ts[pa + spreed]])
            le = int(pa - spreed)
            ue = int(pa + spreed)

            ts[le:ue] = ts[le:ue] - gaus * (tmax - tsm)
        else:
            if pa + spreed > len(ts):
                le = int(pa - spreed)
                ts[le:-1] = np.linspace(ts[le], ts[-1], len(ts[le:-1]))
            else:
                le = int(pa - spreed)
                ue = int(pa + spreed)
                ts[le:ue] = np.linspace(ts[le], ts[ue], len(ts[le:ue]))

        i = i + 1
    return ts
