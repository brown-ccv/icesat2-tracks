import logging

import numpy as np
import pandas as pd
import icesat2waves.tools.spectral_estimates as spec
import icesat2waves.tools.iotools as io_local

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import h5py

_logger = logging.getLogger(__name__)


def derive_beam_statistics(Gd, all_beams, Lmeter=10e3, dx=10):
    """
    this method returns a dict of dataframes with the beam statistics
    Gd          is a dict of beam tables or a hdf5 file
    all_beams   is a list of all beams
    Lemter      is the length of the segment in meters for the statistics
    dx          is the nominal resolution of the ATL06 data in meters
    """

    D = dict()
    for k in all_beams:
        if isinstance(Gd, dict):
            Gi = Gd[k]
        elif isinstance(Gd, h5py.File):
            Gi = io_local.get_beam_hdf_store(Gd[k])
        else:
            _logger.debug("Gd is neither dict nor hdf5 file")
            break

        dd = Gi["h_mean"]
        xx = Gi["dist"]

        def get_var(sti):
            mask = (sti[0] < xx) & (xx <= sti[1])
            return np.nanvar(dd[mask])

        def get_N(sti):
            mask = (sti[0] < xx) & (xx <= sti[1])
            return dd[mask].size

        def get_lat(sti):
            mask = (sti[0] < xx) & (xx <= sti[1])
            return np.nanmean(Gi["lats"][mask])

        iter_x = spec.create_chunk_boundaries_unit_lengths(
            Lmeter, [xx.min(), xx.max()], ov=0, iter_flag=False
        )[1, :]

        stencil_iter = spec.create_chunk_boundaries_unit_lengths(
            Lmeter, [xx.min(), xx.max()], ov=0, iter_flag=True
        )
        var_list = np.array(list(map(get_var, stencil_iter)))

        stencil_iter = spec.create_chunk_boundaries_unit_lengths(
            Lmeter, [xx.min(), xx.max()], ov=0, iter_flag=True
        )
        N_list = np.array(list(map(get_N, stencil_iter)))

        stencil_iter = spec.create_chunk_boundaries_unit_lengths(
            Lmeter, [xx.min(), xx.max()], ov=0, iter_flag=True
        )
        lat_list = np.array(list(map(get_lat, stencil_iter)))

        # make Dataframe
        df = pd.DataFrame()
        df["x"] = iter_x
        df["lat"] = lat_list
        df["var"] = var_list
        df["N"] = N_list * 2 * dx / Lmeter

        D[k] = df

    return D


def plot_beam_statistics(D, high_beams, low_beams, col_dict, track_name=None):
    """
    Plots the beam statistics in a 2 x 2 plot
    D is a dict of dataframes with the beam statistics
    high_beams is a list of high beams
    low_beams is a list of low beams
    col_dict is a dict with the colors for the beams
    track_name is the name of the track
    """

    if track_name is not None:
        plt.suptitle(track_name, fontsize=10)

    gs = gridspec.GridSpec(2, 3)

    # make 2 x 2 plot
    ax1 = plt.subplot(gs[0, 0])
    for k in high_beams:
        plt.plot(
            D[k]["x"] / 1e3,
            np.sqrt(D[k]["var"]),
            ".",
            color=col_dict[k],
            markersize=4,
            label=k,
        )

    plt.title("high beams std", loc="left")
    plt.ylabel("segment std log(m)")

    ax1.set_yscale("log")

    ax2 = plt.subplot(gs[1, 0])
    for k in high_beams:
        Di = D[k]["N"]
        Di[Di == 0] = np.nan
        plt.plot(
            D[k]["x"] / 1e3, D[k]["N"], ".", color=col_dict[k], markersize=4, label=k
        )

    plt.title("high beams N", loc="left")
    plt.xlabel("along track distance (km)")
    plt.ylabel("Point Density (m)")

    ax3 = plt.subplot(gs[0, 1])
    for k in low_beams:
        plt.plot(
            D[k]["x"] / 1e3,
            np.sqrt(D[k]["var"]),
            ".",
            color=col_dict[k],
            markersize=4,
            label=k,
        )

    plt.title("low beams std", loc="left")

    ax3.set_yscale("log")

    ax4 = plt.subplot(gs[1, 1])
    for k in low_beams:
        Di = D[k]["N"]
        Di[Di == 0] = np.nan
        plt.plot(
            D[k]["x"] / 1e3, D[k]["N"], ".", color=col_dict[k], markersize=4, label=k
        )

    plt.title("low beams N", loc="left")
    plt.xlabel("along track distance (km)")

    ax5 = plt.subplot(gs[0:2, 2])

    lat_shift = 0
    for k in low_beams:
        Di = D[k]
        ax5.scatter(
            Di["x"] / 1e3,
            Di["lat"] + lat_shift,
            s=np.exp(Di["N"] * 5),
            marker=".",
            color=col_dict[k],
            label=k,
            alpha=0.3,
        )
        lat_shift = lat_shift + 2

    for k in high_beams:
        Di = D[k]
        ax5.scatter(
            Di["x"] / 1e3,
            Di["lat"] + lat_shift,
            s=np.exp(Di["N"] * 5),
            marker=".",
            color=col_dict[k],
            label=k,
            alpha=0.3,
        )
        lat_shift = lat_shift + 2

    ax5.set_title("Density in space", loc="left")
    ax5.set_ylabel("Latitude (deg)")
    ax5.set_xlabel("along track distance (km)")
    ax5.legend()
    plt.show()


## plot track stats basics for sliderules ATL06 output
def plot_ATL06_track_data(G2, cdict):
    """
    Plots the beam statistics in a 3 x 3 plot
    G2      is a GeoDataFrame from SL (ATL06)
    title   is the title of the plot
    cdict   is a dict with the colors for each 'spot'
    returns a figure object
    """
    gs = gridspec.GridSpec(3, 3)

    ax1 = plt.subplot(gs[0, 0:2])
    ax2 = plt.subplot(gs[1, 0:2])
    ax3 = plt.subplot(gs[2, 0:2])
    ax4 = plt.subplot(gs[0, 2])
    ax5 = plt.subplot(gs[1, 2])
    ax6 = plt.subplot(gs[2, 2])

    for sp in G2["spot"].unique():
        Gc = G2[G2["spot"] == 1]

        Gc["h_mean_gradient"] = np.gradient(Gc["h_mean"])
        ts_config = {
            "marker": ".",
            "markersize": 0.2,
            "linestyle": "none",
            "color": cdict[sp],
            "alpha": 0.3,
        }
        hist_confit = {"density": True, "color": cdict[sp], "alpha": 0.3}

        ax1.plot(Gc.geometry.y, Gc["h_mean"], **ts_config)
        ax2.plot(Gc.geometry.y, Gc["h_mean_gradient"], **ts_config)
        ax3.plot(Gc.geometry.y, Gc["n_fit_photons"], **ts_config)

        Gc["h_mean"].plot.hist(ax=ax4, bins=30, **hist_confit)
        Gc["h_mean_gradient"].plot.hist(
            ax=ax5, bins=np.linspace(-5, 5, 30), **hist_confit
        )
        Gc["rms_misfit"].plot.hist(ax=ax6, bins=30, **hist_confit)

    ax1.set_ylabel("h_mean (m)")
    ax2.set_ylabel("slope (m/m)")
    ax3.set_ylabel("N Photons")
    ax3.set_xlabel("Latitude (degree)")
    ax1.set_xticklabels([])
    ax2.set_xticklabels([])

    ax1.axhline(0, color="k", linestyle="-", linewidth=0.8)
    ax2.axhline(0, color="k", linestyle="-", linewidth=0.8)

    ax1.set_title("Height", loc="left")
    ax2.set_title("Slope", loc="left")
    ax3.set_title("Photons per extend", loc="left")

    ax4.set_title("Histograms", loc="left")
    ax5.set_title("Histograms", loc="left")

    ax6.set_title("Error Hist.", loc="left")
    ax6.set_xlabel("rms_misfit (m)")

    for axi in [ax4, ax5, ax6]:
        axi.set_ylabel("")

    return [ax1, ax2, ax3, ax4, ax5, ax6]
