#!/usr/bin/env python3
"""
This file open a ICEsat2 track applied filters and corrections and returns smoothed photon heights on a regular grid in an .nc file.
This is python 3
"""
import logging
from ast import comprehension
from pathlib import Path
import matplotlib
import numpy as np
import xarray as xr
from matplotlib.gridspec import GridSpec
import typer

import icesat2waves.tools.iotools as io
import icesat2waves.tools.generalized_FT as gFT
import icesat2waves.local_modules.m_tools_ph3 as MT
from icesat2waves.local_modules import m_general_ph3 as M
from icesat2waves.config.startup import (
    mconfig,
    color_schemes,
    plt,
    font_for_print,
)

from icesat2waves.clitools import (
    validate_batch_key,
    validate_output_dir,
    update_paths_mconfig,
    report_input_parameters,
    validate_track_name_steps_gt_1,
    makeapp,
)

_logger = logging.getLogger(__name__)


def plot_wavenumber_spectrogram(ax, Gi, clev, title=None, plot_photon_density=True):
    if Gi.k[0] == 0:
        Gi = Gi.sel(k=Gi.k[1:])
    x_lambda = 2 * np.pi / Gi.k
    plt.pcolormesh(
        Gi.x / 1e3, x_lambda, Gi, cmap=plt.cm.ocean_r, vmin=clev[0], vmax=clev[-1]
    )

    ax.set_yscale("log")

    if plot_photon_density:
        plt.plot(
            Gi.x / 1e3,
            x_lambda[-1] + (Gi.N_per_stancil / Gi.N_per_stancil.max()) * 10,
            c="black",
            linewidth=0.8,
            label="NAN-density",
        )
        plt.fill_between(
            Gi.x / 1e3,
            x_lambda[-1] + (Gi.N_per_stancil / Gi.N_per_stancil.max()) * 10,
            0,
            color="gray",
            alpha=0.3,
        )
        ax.axhline(30, color="black", linewidth=0.3)

    plt.ylim(x_lambda[-1], x_lambda[0])
    plt.title(title, loc="left")


def plot_data_eta(D, offset=0, **kargs):
    eta_1 = D.eta  # + D.x
    y_data = D.y_model + offset
    plt.plot(eta_1, y_data, **kargs)
    return eta_1


def plot_model_eta(D, ax, offset=0, **kargs):
    eta = D.eta  # + D.x
    y_data = D.y_model + offset
    plt.plot(eta, y_data, **kargs)

    ax.axvline(eta[0].data, linewidth=0.1, color=kargs["color"], alpha=0.5)
    ax.axvline(eta[-1].data, linewidth=0.1, color=kargs["color"], alpha=0.5)


matplotlib.use("Agg")  # prevent plot windows from opening


def run_B03_plot_spectra_ov(
    track_name: str = typer.Option(..., callback=validate_track_name_steps_gt_1),
    batch_key: str = typer.Option(..., callback=validate_batch_key),
    ID_flag: bool = True,
    output_dir: str = typer.Option(None, callback=validate_output_dir),
):
    """
    TODO: add docstring
    """

    track_name, batch_key, _ = io.init_from_input(
        [
            None,
            track_name,
            batch_key,
            ID_flag,
        ]  # init_from_input expects sys.argv with 4 elements
    )

    kargs = {
        "track_name": track_name,
        "batch_key": batch_key,
        "ID_flag": ID_flag,
        "output_dir": output_dir,
    }
    report_input_parameters(**kargs)

    hemis, _ = batch_key.split("_")

    workdir, plotsdir = update_paths_mconfig(output_dir, mconfig)

    load_path = Path(workdir, batch_key, "B02_spectra")
    load_file = str(load_path / ("B02_" + track_name))
    plot_path = Path(plotsdir, hemis, batch_key, track_name)
    plot_path.mkdir(parents=True, exist_ok=True)

    # TODO: use list comprehension to load all the files
    Gk = xr.open_dataset(load_file + "_gFT_k.nc")
    Gx = xr.open_dataset(load_file + "_gFT_x.nc")
    Gfft = xr.open_dataset(load_file + "_FFT.nc")

    all_beams = mconfig["beams"]["all_beams"]
    high_beams = mconfig["beams"]["high_beams"]
    low_beams = mconfig["beams"]["low_beams"]
    color_schemes.colormaps2(21)

    col_dict = color_schemes.rels
    F = M.FigureAxisXY(9, 3, view_scale=0.5)

    plt.subplot(1, 3, 1)
    plt.title(track_name, loc="left")
    for k in all_beams:
        I = Gk.sel(beam=k)
        I2 = Gx.sel(beam=k)
        plt.plot(I["lon"], I["lat"], ".", c=col_dict[k], markersize=0.7, linewidth=0.3)
        plt.plot(I2["lon"], I2["lat"], "|", c=col_dict[k], markersize=0.7)

    plt.xlabel("lon")
    plt.ylabel("lat")

    plt.subplot(1, 3, 2)

    xscale = 1e3
    for k in all_beams:
        I = Gk.sel(beam=k)
        plt.plot(
            I["x_coord"] / xscale,
            I["y_coord"] / xscale,
            ".",
            c=col_dict[k],
            linewidth=0.8,
            markersize=0.8,
        )

    plt.xlabel("x_coord (km)")
    plt.ylabel("y_coord (km)")

    plt.subplot(1, 3, 3)

    xscale = 1e3
    for k in all_beams:
        I = Gk.sel(beam=k)
        plt.plot(
            I["x_coord"] / xscale,
            (I["y_coord"] - I["y_coord"][0]),
            ".",
            c=col_dict[k],
            linewidth=0.8,
            markersize=0.8,
        )

    plt.xlabel("x_coord (km)")
    plt.ylabel("y_coord deviation (m)")

    F.save_light(path=plot_path, name="B03_specs_coord_check")

    # TODO: refactor to make more readable. CP
    G_gFT_wmean = (
        Gk["gFT_PSD_data"].where(~np.isnan(Gk["gFT_PSD_data"]), 0) * Gk["N_per_stancil"]
    ).sum("beam") / Gk["N_per_stancil"].sum("beam")
    G_gFT_wmean["N_per_stancil"] = Gk["N_per_stancil"].sum("beam")

    G_fft_wmean = (Gfft.where(~np.isnan(Gfft), 0) * Gfft["N_per_stancil"]).sum(
        "beam"
    ) / Gfft["N_per_stancil"].sum("beam")
    G_fft_wmean["N_per_stancil"] = Gfft["N_per_stancil"].sum("beam")
    Gmean = G_gFT_wmean.rolling(k=5, center=True).mean()

    # TODO: make function to compute k_max. CP
    try:
        k_max = Gmean.k[Gmean.isel(x=slice(0, 5)).mean("x").argmax().data].data
    except Exception:
        k_max = Gmean.k[Gmean.isel(x=slice(0, 20)).mean("x").argmax().data].data

    k_max_range = (k_max * 0.75, k_max, k_max * 1.25)
    font_for_print()
    F = M.FigureAxisXY(6.5, 5.6, container=True, view_scale=1)
    Lmeters = Gk.L.data[0]

    plt.suptitle("gFT Slope Spectrograms\n" + track_name, y=0.98)
    gs = GridSpec(3, 3, wspace=0.2, hspace=0.5)

    Gplot = (
        G_gFT_wmean.squeeze()
        .rolling(k=10, min_periods=1, center=True)
        .median()
        .rolling(x=3, min_periods=1, center=True)
        .median()
    )
    dd = 10 * np.log10(Gplot)
    dd = dd.where(~np.isinf(dd), np.nan)
    clev_log = M.clevels([dd.quantile(0.01).data, dd.quantile(0.98).data * 1.2], 31) * 1

    xlims = Gmean.x[0] / 1e3, Gmean.x[-1] / 1e3

    k = high_beams[0]
    for pos, k, pflag in zip(
        [gs[0, 0], gs[0, 1], gs[0, 2]], high_beams, [True, False, False]
    ):
        ax0 = F.fig.add_subplot(pos)
        Gplot = Gk.sel(beam=k).gFT_PSD_data.squeeze()
        dd2 = 10 * np.log10(Gplot)
        dd2 = dd2.where(~np.isinf(dd2), np.nan)
        plot_wavenumber_spectrogram(
            ax0, dd2, clev_log, title=k + " unsmoothed", plot_photon_density=True
        )
        plt.xlim(xlims)
        if pflag:
            plt.ylabel("Wave length\n(meters)")
            plt.legend()

    for pos, k, pflag in zip(
        [gs[1, 0], gs[1, 1], gs[1, 2]], low_beams, [True, False, False]
    ):
        ax0 = F.fig.add_subplot(pos)
        Gplot = Gk.sel(beam=k).gFT_PSD_data.squeeze()
        dd2 = 10 * np.log10(Gplot)
        dd2 = dd2.where(~np.isinf(dd2), np.nan)
        plot_wavenumber_spectrogram(
            ax0, dd2, clev_log, title=k + " unsmoothed", plot_photon_density=True
        )
        plt.xlim(xlims)
        if pflag:
            plt.ylabel("Wave length\n(meters)")
            plt.legend()

    ax0 = F.fig.add_subplot(gs[2, 0])

    plot_wavenumber_spectrogram(
        ax0,
        dd,
        clev_log,
        title="smoothed weighted mean \n10 $\log_{10}( (m/m)^2 m )$",
        plot_photon_density=True,
    )
    plt.xlim(xlims)

    line_styles = ["--", "-", "--"]
    for k_max, style in zip(k_max_range, line_styles):
        ax0.axhline(2 * np.pi / k_max, color="red", linestyle=style, linewidth=0.5)

    if pflag:
        plt.ylabel("Wave length\n(meters)")
        plt.legend()

    pos = gs[2, 1]
    ax0 = F.fig.add_subplot(pos)
    plt.title("Photons density ($m^{-1}$)", loc="left")

    for k in all_beams:
        I = Gk.sel(beam=k)["gFT_PSD_data"]
        plt.plot(Gplot.x / 1e3, I.N_photons / I.L.data, label=k, linewidth=0.8)
    plt.plot(
        Gplot.x / 1e3,
        G_gFT_wmean.N_per_stancil / 3 / I.L.data,
        c="black",
        label="ave Photons",
        linewidth=0.8,
    )
    plt.xlim(xlims)
    plt.xlabel("Distance from the Ice Edge (km)")

    pos = gs[2, 2]

    ax0 = F.fig.add_subplot(pos)
    ax0.set_yscale("log")

    plt.title("Peak Spectral Power", loc="left")

    x0 = Gk.x[0].data
    for k in all_beams:
        I = Gk.sel(beam=k)["gFT_PSD_data"]
        plt.scatter(
            I.x.data / 1e3,
            I.sel(k=slice(k_max_range[0], k_max_range[2])).integrate("k").data,
            s=0.5,
            marker=".",
            color="red",
            alpha=0.3,
        )
        I = Gfft.sel(beam=k)
        plt.scatter(
            (x0 + I.x.data) / 1e3,
            I.power_spec.sel(k=slice(k_max_range[0], k_max_range[2])).integrate("k"),
            s=0.5,
            marker=".",
            c="blue",
            alpha=0.3,
        )

    Gplot = G_fft_wmean.squeeze()
    Gplot = Gplot.power_spec[
        :, Gplot.N_per_stancil >= Gplot.N_per_stancil.max().data * 0.9
    ]
    plt.plot(
        (x0 + Gplot.x) / 1e3,
        Gplot.sel(k=slice(k_max_range[0], k_max_range[2])).integrate("k"),
        ".",
        markersize=1.5,
        c="blue",
        label="FFT",
    )

    Gplot = G_gFT_wmean.squeeze()
    plt.plot(
        Gplot.x / 1e3,
        Gplot.sel(k=slice(k_max_range[0], k_max_range[2])).integrate("k"),
        ".",
        markersize=1.5,
        c="red",
        label="gFT",
    )

    plt.ylabel("1e-3 $(m)^2~m$")
    plt.legend()

    F.save_light(path=plot_path, name="B03_specs_L" + str(Lmeters))

    Gk.sel(beam=k).gFT_PSD_data.plot()

    if "y_data" in Gx.sel(beam="gt3r").keys():
        _logger.debug("ydata is %s", ("y_data" in Gx.sel(beam="gt3r").keys()))
    else:
        _logger.warning("ydata is %s", ("y_data" in Gx.sel(beam="gt3r").keys()))
        MT.json_save("B03_fail", plot_path, {"reason": "no y_data"})
        _logger.warning("failed, exit")
        exit()

    fltostr, _ = MT.float_to_str, MT.num_to_str

    font_for_print()

    (plot_path / "B03_spectra").mkdir(parents=True, exist_ok=True)

    x_pos_sel = np.arange(Gk.x.size)[
        ~np.isnan(Gk.mean("beam").mean("k").gFT_PSD_data.data)
    ]
    x_pos_max = (
        Gk.mean("beam")
        .mean("k")
        .gFT_PSD_data[~np.isnan(Gk.mean("beam").mean("k").gFT_PSD_data)]
        .argmax()
        .data
    )
    xpp = x_pos_sel[[int(i) for i in np.round(np.linspace(0, x_pos_sel.size - 1, 4))]]
    xpp = np.insert(xpp, 0, x_pos_max)

    for i in xpp:
        F = M.FigureAxisXY(6, 8, container=True, view_scale=0.8)

        plt.suptitle(
            "gFT Model and Spectrograms | x=" + str(Gk.x[i].data) + " \n" + track_name,
            y=0.95,
        )
        gs = GridSpec(5, 6, wspace=0.2, hspace=0.7)

        ax0 = F.fig.add_subplot(gs[0:2, :])
        col_d = color_schemes.__dict__["rels"]

        neven = True
        offs = 0
        for k in all_beams:
            Gx_1 = Gx.isel(x=i).sel(beam=k)
            Gk_1 = Gk.isel(x=i).sel(beam=k)

            plot_model_eta(
                Gx_1,
                ax0,
                offset=offs,
                linestyle="-",
                color=col_d[k],
                linewidth=0.4,
                alpha=1,
                zorder=12,
            )

            # original data
            eta_1 = plot_data_eta(
                Gx_1,
                offset=offs,
                linestyle="-",
                c="k",
                linewidth=1,
                alpha=0.5,
                zorder=11,
            )

            # reconstruct in  gaps
            FT = gFT.generalized_Fourier(Gx_1.eta + Gx_1.x, None, Gk_1.k)
            _ = FT.get_H()
            FT.p_hat = np.concatenate([Gk_1.gFT_cos_coeff, Gk_1.gFT_sin_coeff])
            plt.plot(
                Gx_1.eta,
                FT.model() + offs,
                "-",
                c="orange",
                linewidth=0.3,
                alpha=1,
                zorder=2,
            )

            if neven:
                neven = False
                offs += 0.3
            else:
                neven = True
                offs += 0.6

        dx = eta_1.diff("eta").mean().data

        eta_ticks = np.linspace(Gx_1.eta.data[0], Gx_1.eta.data[-1], 11)

        ax0.set_xticks(eta_ticks)
        ax0.set_xticklabels(eta_ticks / 1e3)
        plt.xlim(eta_1[0].data - 40 * dx, eta_1[-1].data + 40 * dx)
        plt.title("Model reconst.", loc="left")

        plt.ylabel("relative slope (m/m)")
        # TODO: compute xlabel as fstring. CP
        plt.xlabel(
            "segment distance $\eta$ (km) @ x=" + fltostr(Gx_1.x.data / 1e3, 2) + "km"
        )

        # spectra
        # define threshold
        k_thresh = 0.085
        ax1_list = list()
        dd_max = list()
        for pos, kgroup, lflag in zip(
            [gs[2, 0:2], gs[2, 2:4], gs[2, 4:]],
            [["gt1l", "gt1r"], ["gt2l", "gt2r"], ["gt3l", "gt3r"]],
            [True, False, False],
        ):
            ax11 = F.fig.add_subplot(pos)
            ax11.tick_params(labelleft=lflag)
            ax1_list.append(ax11)
            for k in kgroup:
                Gx_1 = Gx.isel(x=i).sel(beam=k)
                Gk_1 = Gk.isel(x=i).sel(beam=k)

                klim = Gk_1.k[0], Gk_1.k[-1]

                if "l" in k:
                    dd = Gk_1.gFT_PSD_data
                    plt.plot(Gk_1.k, dd, color="gray", linewidth=0.5, alpha=0.5)

                dd = Gk_1.gFT_PSD_data.rolling(k=10, min_periods=1, center=True).mean()
                plt.plot(Gk_1.k, dd, color=col_d[k], linewidth=0.8)
                # handle the 'All-NaN slice encountered' warning
                if np.all(np.isnan(dd.data)):
                    dd_max.append(np.nan)
                else:
                    dd_max.append(np.nanmax(dd.data))

                plt.xlim(klim)
                if lflag:
                    plt.ylabel("$(m/m)^2/k$")
                    plt.title("Energy Spectra", loc="left")

                plt.xlabel("wavenumber k (2$\pi$ m$^{-1}$)")

            ax11.axvline(k_thresh, linewidth=1, color="gray", alpha=1)
            ax11.axvspan(k_thresh, klim[-1], color="gray", alpha=0.5, zorder=12)

        if not np.all(np.isnan(dd_max)):
            max_vale = np.nanmax(dd_max)
            for ax in ax1_list:
                ax.set_ylim(0, max_vale * 1.1)

        ax0 = F.fig.add_subplot(gs[-2:, :])

        neven = True
        offs = 0
        for k in all_beams:
            Gx_1 = Gx.isel(x=i).sel(beam=k)
            Gk_1 = Gk.isel(x=i).sel(beam=k)

            # original data
            eta_1 = plot_data_eta(
                Gx_1,
                offset=offs,
                linestyle="-",
                c="k",
                linewidth=1.5,
                alpha=0.5,
                zorder=11,
            )

            # reconstruct in  gaps
            FT = gFT.generalized_Fourier(Gx_1.eta + Gx_1.x, None, Gk_1.k)
            _ = FT.get_H()
            FT.p_hat = np.concatenate([Gk_1.gFT_cos_coeff, Gk_1.gFT_sin_coeff])

            p_hat_k = np.concatenate([Gk_1.k, Gk_1.k])
            k_mask = p_hat_k < k_thresh
            FT.p_hat[~k_mask] = 0

            plt.plot(
                Gx_1.eta,
                FT.model() + offs,
                "-",
                c=col_d[k],
                linewidth=0.8,
                alpha=1,
                zorder=12,
            )

            if neven:
                neven = False
                offs += 0.3
            else:
                neven = True
                offs += 0.6

        dx = eta_1.diff("eta").mean().data

        eta_ticks = np.linspace(Gx_1.eta.data[0], Gx_1.eta.data[-1], 11)

        ax0.set_xticks(eta_ticks)
        ax0.set_xticklabels(eta_ticks / 1e3)
        plt.xlim(eta_1[1000].data - 40 * dx, eta_1[-1000].data + 40 * dx)
        plt.title("Low-Wavenumber Model reconst.", loc="left")

        plt.ylabel("relative slope (m/m)")
        # TODO: compute xlabel as fstring. CP
        plt.xlabel(
            "segment distance $\eta$ (km) @ x=" + fltostr(Gx_1.x.data / 1e3, 2) + "km"
        )

        F.save_pup(path=str(plot_path / "B03_spectra"), name=f"B03_freq_reconst_x{i}")

    MT.json_save(
        "B03_success",
        plot_path,
        {"time": "time.asctime( time.localtime(time.time()) )"},
    )

    _logger.info("success")


plot_spectra = makeapp(run_B03_plot_spectra_ov, name="plotspectra")

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    plot_spectra()
