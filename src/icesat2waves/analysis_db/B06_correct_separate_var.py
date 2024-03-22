"""
This file open a ICEsat2 track applied filters and corections and returns smoothed photon heights on a regular grid in an .nc file.
This is python 3
"""

import logging
import os


import h5py
from pathlib import Path
import icesat2waves.tools.iotools as io
import icesat2waves.local_modules.m_tools_ph3 as MT
from icesat2waves.local_modules import m_general_ph3 as M
import time
import copy
import icesat2waves.tools.generalized_FT as gFT
import pandas as pd
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import piecewise_regression
import typer

from icesat2waves.config.startup import (
    mconfig,
    color_schemes,
    font_for_pres,
    font_for_print,
    lstrings,
    fig_sizes,
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


def get_correct_breakpoint(pw_results):
    br_points = [i for i in pw_results.keys() if "breakpoint" in i]

    br_points_df = pw_results[br_points]
    br_points_sorted = br_points_df.sort_values()

    alphas_sorted = []
    betas_sorted = []
    for point in br_points_sorted.index:
        alphas_sorted.append(point.replace("breakpoint", "alpha"))
        betas_sorted.append(point.replace("breakpoint", "beta"))

    alphas_sorted.append(f"alpha{len(alphas_sorted) + 1}")

    ## TODO: Camilo decided to leave this piece of code here in case the output data is not
    ## the right one

    # alphas_sorted = [
    #     point.replace("breakpoint", "alpha") for point in br_points_df.sort_values().index
    # ]
    # alphas_sorted.append(f"alpha{len(alphas_sorted) + 1}")

    # betas_sorted = [
    #     point.replace("breakpoint", "beta") for point in br_points_df.sort_values().index
    # ]

    # betas_sorted
    alphas_v2 = list()
    alpha_i = pw_results["alpha1"]
    alphas_v2 = [alpha_i := alpha_i + i for i in [0] + list(pw_results[betas_sorted])]

    alphas_v2_sorted = pd.Series(index=alphas_sorted, data=alphas_v2)
    br_points_sorted[f"breakpoint{br_points_sorted.size + 1}"] = "end"

    _logger.debug("all alphas")
    _logger.debug("%s", alphas_v2_sorted)
    slope_mask = alphas_v2_sorted < 0

    if sum(slope_mask) == 0:
        _logger.debug("no negative slope found, set to lowest")
        breakpoint = "start"
    else:
        # take steepest slope
        alpah_v2_sub = alphas_v2_sorted[slope_mask]
        _logger.debug("alpah_v2_sub: %s", alpah_v2_sub)
        _logger.debug("alpah_v2_sub.argmin: %s", alpah_v2_sub.argmin())
        break_point_name = alpah_v2_sub.index[alpah_v2_sub.argmin()].replace(
            "alpha", "breakpoint"
        )

        # take first slope
        breakpoint = br_points_sorted[break_point_name]

    return breakpoint


def get_breakingpoints(xx, dd):

    convergence_flag = True
    n_breakpoints = 3
    while convergence_flag:
        pw_fit = piecewise_regression.Fit(xx, dd, n_breakpoints=n_breakpoints)
        _logger.debug(
            "n_breakpoints %s\n%s", n_breakpoints, pw_fit.get_results()["converged"]
        )
        convergence_flag = not pw_fit.get_results()["converged"]
        n_breakpoints += 1
        if n_breakpoints >= 4:
            convergence_flag = False

    pw_results = pw_fit.get_results()

    if pw_results["converged"]:
        pw_results_df = pd.DataFrame(pw_results["estimates"]).loc["estimate"]

        breakpoint = get_correct_breakpoint(pw_results_df)

        return pw_fit, breakpoint

    else:
        return pw_fit, False


def define_noise_wavenumber_piecewise(data_xr, plot_flag=False):

    data_log = np.log(data_xr)

    k = data_log.k.data
    k_log = np.log(k)

    pw_fit, breakpoint_log = get_breakingpoints(k_log, data_log.data)

    if breakpoint_log == "start":
        _logger.debug("no decay, set to lowerst wavenumber")
        breakpoint_log = k_log[0]
    if (breakpoint_log == "end") | (breakpoint_log is False):
        _logger.debug("higest wavenumner")
        breakpoint_log = k_log[-1]

    breakpoint_pos = abs(k_log - breakpoint_log).argmin()
    breakpoint_k = k[breakpoint_pos]

    if plot_flag:
        pw_fit.plot()
        plt.plot(k_log, data_log)

    return breakpoint_k, pw_fit


def weighted_mean(data, weights, additional_data=None):
    # Where data is not NaN, replace NaN with 0 and multiply by weights
    weighted_data = data.where(~np.isnan(data), 0) * weights

    # Calculate the sum of weighted data along the specified dimension
    mean_data = weighted_data.sum("beam") / weights.sum("beam")

    # Optionally, add additional data to the resulting array
    if additional_data is not None:
        mean_data[additional_data] = data[additional_data].sum("beam")

    return mean_data


def calculate_k_end(x, k, k_end_previous, G_gFT_smth):
    _logger.debug("From calculate_k_end -- x: %s", x)
    k_end, _ = define_noise_wavenumber_piecewise(
        G_gFT_smth.sel(x=x) / k, plot_flag=False
    )
    k_save = k_end_previous if k_end == k[0] else k_end
    return k_save


def tanh_fitler(x, x_cutoff, sigma_g=0.01):
    """
    zdgfsg
    """

    decay = 0.5 - np.tanh((x - x_cutoff) / sigma_g) / 2
    return decay


def get_k_x_corrected(Gk, theta=0, theta_flag=False):

    if not theta_flag:
        return np.nan, np.nan

    lam_p = 2 * np.pi / Gk.k
    lam = lam_p * np.cos(theta)
    k_corrected = 2 * np.pi / lam
    x_corrected = Gk.x * np.cos(theta)

    return k_corrected, x_corrected


### TODO: Fix the variables in this function.
##
# dx = Gx.eta.diff("eta").mean().dat
###
# def reconstruct_displacement(Gx_1, Gk_1, T3, k_thresh):
#             """
#             reconstructs photon displacement heights for each stancil given the model parameters in Gk_1
#             A low-pass frequeny filter can be applied using k-thresh

#             inputs:
#             Gk_1    model data per stencil from _gFT_k file with sin and cos coefficients
#             Gx_1    real data per stencil from _gFT_x file with mean photon heights and coordindate systems
#             T3
#             k_thresh (None) threshold for low-pass filter

#             returns:
#             height_model  reconstucted displements heights of the stancil
#             poly_offset   fitted staight line to the residual between observations and model to account for low-pass variability
#             nan_mask      mask where is observed data in
#             """

#             dist_stencil = Gx_1.eta + Gx_1.x

#             gFT_cos_coeff_sel = np.copy(Gk_1.gFT_cos_coeff)
#             gFT_sin_coeff_sel = np.copy(Gk_1.gFT_sin_coeff)

#             gFT_cos_coeff_sel = gFT_cos_coeff_sel * tanh_fitler(
#                 Gk_1.k, k_thresh, sigma_g=0.003
#             )
#             gFT_sin_coeff_sel = gFT_sin_coeff_sel * tanh_fitler(
#                 Gk_1.k, k_thresh, sigma_g=0.003
#             )

#             FT_int = gFT.generalized_Fourier(Gx_1.eta + Gx_1.x, None, Gk_1.k)
#             _ = FT_int.get_H()
#             FT_int.p_hat = np.concatenate(
#                 [-gFT_sin_coeff_sel / Gk_1.k, gFT_cos_coeff_sel / Gk_1.k]
#             )

#             dx = Gx.eta.diff("eta").mean().data
#             height_model = FT_int.model() / dx
#             dist_nanmask = np.isnan(Gx_1.y_data)
#             height_data = np.interp(
#                 dist_stencil, T3_sel["dist"], T3_sel["heights_c_weighted_mean"]
#             )
#             return height_model, np.nan, dist_nanmask


def save_table(data, tablename, save_path):
    try:
        io.save_pandas_table(data, tablename, save_path)
    except Exception as e:
        tabletoremove = save_path + tablename + ".h5"
        _logger.warning(
            "Failed to save table with error %s. Removing %s and re-trying..",
            e,
            tabletoremove,
        )
        os.remove(tabletoremove)
        io.save_pandas_table(data, tablename, save_path)


def buil_G_error(Gk_sel, PSD_list, list_name):

    return xr.DataArray(
        data=PSD_list, coords=Gk_sel.drop("N_per_stancil").coords, name=list_name
    ).expand_dims("beam")


def run_B06_correct_separate_var(
    track_name: str = typer.Option(..., callback=validate_track_name_steps_gt_1),
    batch_key: str = typer.Option(..., callback=validate_batch_key),
    ID_flag: bool = True,
    output_dir: str = typer.Option(..., callback=validate_output_dir),
):
    """
    TODO: Add docstring.
    """

    color_schemes.colormaps2(31, gamma=1)
    col_dict = color_schemes.rels

    track_name, batch_key, test_flag = io.init_from_input(
        [
            None,
            track_name,
            batch_key,
            ID_flag,
        ]  # init_from_input expects sys.argv with 4 elements
    )

    kwargs = {
        "track_name": track_name,
        "batch_key": batch_key,
        "ID_flag": ID_flag,
        "output_dir": output_dir,
    }

    report_input_parameters(**kwargs)

    hemis, _ = batch_key.split("_")

    workdir, plotsdir = update_paths_mconfig(output_dir, mconfig)

    xr.set_options(display_style="text")

    all_beams = mconfig["beams"]["all_beams"]
    high_beams = mconfig["beams"]["high_beams"]
    low_beams = mconfig["beams"]["low_beams"]

    load_path_work = workdir / batch_key

    h5_file_path = load_path_work / "B01_regrid" / (track_name + "_B01_binned.h5")
    with h5py.File(h5_file_path, "r") as B3_hdf5:
        load_path_angle = load_path_work / "B04_angle"
        B3 = {b: io.get_beam_hdf_store(B3_hdf5[b]) for b in all_beams}

    file_suffixes = ["_gFT_k.nc", "_gFT_x.nc"]
    Gk, Gx = [
        xr.open_dataset(load_path_work / "B02_spectra" / f"B02_{track_name}{suffix}")
        for suffix in file_suffixes
    ]

    plot_path = Path(plotsdir, hemis, batch_key, track_name, "B06_correction")
    plot_path.mkdir(parents=True, exist_ok=True)

    save_path = Path(workdir, batch_key, "B06_corrected_separated")
    save_path.mkdir(parents=True, exist_ok=True)

    G_gFT_wmean = (
        Gk.where(~np.isnan(Gk["gFT_PSD_data"]), 0) * Gk["N_per_stancil"]
    ).sum("beam") / Gk["N_per_stancil"].sum("beam")
    G_gFT_wmean["N_photons"] = Gk["N_photons"].sum("beam")

    # plot
    # derive spectral errors:
    Lpoints = Gk.Lpoints.mean("beam").data
    N_per_stancil = Gk.N_per_stancil.mean("beam").data  # [0:-2]

    G_error_data = dict()

    for bb in Gk.beam.data:
        I = Gk.sel(beam=bb)
        b_bat_error = np.concatenate(
            [I.model_error_k_cos.data, I.model_error_k_sin.data]
        )
        Z_error = gFT.complex_represenation(b_bat_error, Gk.k.size, Lpoints)
        PSD_error_data, PSD_error_model = gFT.Z_to_power_gFT(
            Z_error, np.diff(Gk.k)[0], N_per_stancil, Lpoints
        )

        G_error_data[bb] = xr.DataArray(
            data=PSD_error_data,
            coords=I.drop("N_per_stancil").coords,
            name="gFT_PSD_data_error",
        ).expand_dims("beam")

    gFT_PSD_data_error_mean = xr.concat(G_error_data.values(), dim="beam")

    gFT_PSD_data_error_mean = (
        gFT_PSD_data_error_mean.where(~np.isnan(gFT_PSD_data_error_mean), 0)
        * Gk["N_per_stancil"]
    ).sum("beam") / Gk["N_per_stancil"].sum("beam")

    G_gFT_wmean["gFT_PSD_data_err"] = gFT_PSD_data_error_mean

    Gk["gFT_PSD_data_err"] = xr.concat(G_error_data.values(), dim="beam")

    G_gFT_smth = (
        G_gFT_wmean["gFT_PSD_data"].rolling(k=30, center=True, min_periods=1).mean()
    )
    G_gFT_smth["N_photons"] = G_gFT_wmean.N_photons
    G_gFT_smth["N_per_stancil_fraction"] = Gk["N_per_stancil"].T.mean(
        "beam"
    ) / Gk.Lpoints.mean("beam")

    k = G_gFT_smth.k

    F = M.FigureAxisXY()

    plt.loglog(k, G_gFT_smth / k)

    plt.title("displacement power Spectra", loc="left")

    # new version
    k_lim_list = list()
    k_end_previous = np.nan
    x = G_gFT_smth.x.data[0]
    k = G_gFT_smth.k.data
    k_end_previous = np.nan

    k_lim_list = [
        calculate_k_end(x, k, k_end_previous, G_gFT_smth) for x in G_gFT_smth.x.data
    ]

    font_for_pres()
    G_gFT_smth.coords["k_lim"] = ("x", k_lim_list)
    G_gFT_smth.k_lim.plot()
    k_lim_smth = G_gFT_smth.k_lim.rolling(x=3, center=True, min_periods=1).mean()
    k_lim_smth.plot(c="r")

    plt.title("k_c filter", loc="left")
    F.save_light(path=plot_path, name=str(track_name) + "_B06_atten_ov")

    G_gFT_smth["k_lim"] = k_lim_smth
    G_gFT_wmean.coords["k_lim"] = k_lim_smth

    font_for_print()

    fn = copy.copy(lstrings)
    F = M.FigureAxisXY(
        fig_sizes["two_column"][0],
        fig_sizes["two_column"][0] * 0.9,
        container=True,
        view_scale=1,
    )

    plt.suptitle(
        "Cut-off Frequency for Displacement Spectral\n" + io.ID_to_str(track_name),
        y=0.97,
    )
    gs = GridSpec(8, 3, wspace=0.1, hspace=1.5)

    k_lims = G_gFT_wmean.k_lim
    xlims = G_gFT_wmean.k[0], G_gFT_wmean.k[-1]

    k = high_beams[0]

    for pos, k, pflag in zip(
        [gs[0:2, 0], gs[0:2, 1], gs[0:2, 2]], high_beams, [True, False, False]
    ):
        ax0 = F.fig.add_subplot(pos)
        Gplot = (
            Gk.sel(beam=k)
            .isel(x=slice(0, -1))
            .gFT_PSD_data.squeeze()
            .rolling(k=20, x=2, min_periods=1, center=True)
            .mean()
        )
        Gplot = Gplot.where(Gplot["N_per_stancil"] / Gplot["Lpoints"] >= 0.1)

        alpha_range = np.linspace(1, 0, Gplot.x.data.size)

        for x, ialpha in zip(Gplot.x.data, alpha_range):
            plt.loglog(
                Gplot.k,
                Gplot.sel(x=x) / Gplot.k,
                linewidth=0.5,
                color=color_schemes.rels[k],
                alpha=ialpha,
            )
            ax0.axvline(
                k_lims.sel(x=x),
                linewidth=0.4,
                color="black",
                zorder=0,
                alpha=ialpha,
            )

        plt.title(next(fn) + k, color=col_dict[k], loc="left")
        plt.xlim(xlims)

        if pflag:
            ax0.tick_params(labelbottom=False, bottom=True)
            plt.ylabel("Power (m$^2$/k')")
            plt.legend()
        else:
            ax0.tick_params(labelbottom=False, bottom=True, labelleft=False)

    for pos, k, pflag in zip(
        [gs[2:4, 0], gs[2:4, 1], gs[2:4, 2]], low_beams, [True, False, False]
    ):
        ax0 = F.fig.add_subplot(pos)
        Gplot = (
            Gk.sel(beam=k)
            .isel(x=slice(0, -1))
            .gFT_PSD_data.squeeze()
            .rolling(k=20, x=2, min_periods=1, center=True)
            .mean()
        )

        Gplot = Gplot.where(Gplot["N_per_stancil"] / Gplot["Lpoints"] >= 0.1)

        alpha_range = np.linspace(1, 0, Gplot.x.data.size)
        for x, ialpha in zip(Gplot.x.data, alpha_range):
            plt.loglog(
                Gplot.k,
                Gplot.sel(x=x) / Gplot.k,
                linewidth=0.5,
                color=color_schemes.rels[k],
                alpha=ialpha,
            )
            ax0.axvline(
                k_lims.sel(x=x),
                linewidth=0.4,
                color="black",
                zorder=0,
                alpha=ialpha,
            )

        plt.title(next(fn) + k, color=col_dict[k], loc="left")
        plt.xlim(xlims)
        plt.xlabel("observed wavenumber k' ")

        if pflag:
            ax0.tick_params(bottom=True)
            plt.ylabel("Power (m$^2$/k')")
            plt.legend()
        else:
            ax0.tick_params(bottom=True, labelleft=False)

    F.save_light(path=plot_path, name=str(track_name) + "_B06_atten_ov_simple")
    F.save_pup(path=plot_path, name=str(track_name) + "_B06_atten_ov_simple")

    pos = gs[5:, 0:2]
    ax0 = F.fig.add_subplot(pos)

    lat_str = (
        str(np.round(Gx.isel(x=0).lat.mean().data, 2))
        + " to "
        + str(np.round(Gx.isel(x=-1).lat.mean().data, 2))
    )
    plt.title(next(fn) + "Mean Displacement Spectra\n(lat=" + lat_str + ")", loc="left")

    dd = 10 * np.log((G_gFT_smth / G_gFT_smth.k).isel(x=slice(0, -1)))
    dd = dd.where(~np.isinf(dd), np.nan)

    ## filter out segments with less then 10% of data points
    dd = dd.where(G_gFT_smth["N_per_stancil_fraction"] >= 0.1)

    dd_lims = np.round(dd.quantile(0.01).data * 0.95, 0), np.round(
        dd.quantile(0.95).data * 1.05, 0
    )
    plt.pcolor(
        dd.x / 1e3,
        dd.k,
        dd,
        vmin=dd_lims[0],
        vmax=dd_lims[-1],
        cmap=color_schemes.white_base_blgror,
    )
    cb = plt.colorbar(orientation="vertical")

    cb.set_label("Power (m$^2$/k)")
    plt.plot(
        G_gFT_smth.isel(x=slice(0, -1)).x / 1e3,
        G_gFT_smth.isel(x=slice(0, -1)).k_lim,
        color=color_schemes.black,
        linewidth=1,
    )
    plt.ylabel("wavenumber k")
    plt.xlabel("X (km)")

    pos = gs[6:, -1]
    ax9 = F.fig.add_subplot(pos)

    plt.title("Data Coverage (%)", loc="left")
    plt.plot(
        G_gFT_smth.x / 1e3,
        G_gFT_smth["N_per_stancil_fraction"] * 100,
        linewidth=0.8,
        color="black",
    )
    ax9.spines["left"].set_visible(False)
    ax9.spines["right"].set_visible(True)
    ax9.tick_params(labelright=True, right=True, labelleft=False, left=False)
    ax9.axhline(10, linewidth=0.8, linestyle="--", color="black")
    plt.xlabel("X (km)")

    F.save_light(path=plot_path, name=str(track_name) + "_B06_atten_ov")
    F.save_pup(path=plot_path, name=str(track_name) + "_B06_atten_ov")

    def reconstruct_displacement(Gx_1, Gk_1, T3, k_thresh):
        """
        reconstructs photon displacement heights for each stancil given the model parameters in Gk_1
        A low-pass frequeny filter can be applied using k-thresh

        inputs:
        Gk_1    model data per stencil from _gFT_k file with sin and cos coefficients
        Gx_1    real data per stencil from _gFT_x file with mean photon heights and coordindate systems
        T3
        k_thresh (None) threshold for low-pass filter

        returns:
        height_model  reconstucted displements heights of the stancil
        poly_offset   fitted staight line to the residual between observations and model to account for low-pass variability
        nan_mask      mask where is observed data in
        """

        dist_stencil = Gx_1.eta + Gx_1.x

        gFT_cos_coeff_sel = np.copy(Gk_1.gFT_cos_coeff)
        gFT_sin_coeff_sel = np.copy(Gk_1.gFT_sin_coeff)

        gFT_cos_coeff_sel = gFT_cos_coeff_sel * tanh_fitler(
            Gk_1.k, k_thresh, sigma_g=0.003
        )
        gFT_sin_coeff_sel = gFT_sin_coeff_sel * tanh_fitler(
            Gk_1.k, k_thresh, sigma_g=0.003
        )

        FT_int = gFT.generalized_Fourier(Gx_1.eta + Gx_1.x, None, Gk_1.k)
        _ = FT_int.get_H()
        FT_int.p_hat = np.concatenate(
            [-gFT_sin_coeff_sel / Gk_1.k, gFT_cos_coeff_sel / Gk_1.k]
        )

        dx = Gx.eta.diff("eta").mean().data
        height_model = FT_int.model() / dx
        dist_nanmask = np.isnan(Gx_1.y_data)

        return height_model, np.nan, dist_nanmask

    # cutting Table data
    G_height_model = dict()
    k = "gt2l"
    for bb in Gx.beam.data:
        G_height_model_temp = dict()
        for i in np.arange(Gx.x.size):
            Gx_1 = Gx.isel(x=i).sel(beam=bb)
            Gk_1 = Gk.isel(x=i).sel(beam=bb)
            k_thresh = G_gFT_smth.k_lim.isel(x=0).data

            dist_stencil = Gx_1.eta + Gx_1.x
            dist_stencil_lims = dist_stencil[0].data, dist_stencil[-1].data

            T3_sel = B3[k].loc[
                (
                    (B3[k]["dist"] >= dist_stencil_lims[0])
                    & (B3[k]["dist"] <= dist_stencil_lims[1])
                )
            ]

            if T3_sel.shape[0] != 0:
                height_model, poly_offset, _ = reconstruct_displacement(
                    Gx_1, Gk_1, T3_sel, k_thresh=k_thresh
                )
                poly_offset = poly_offset * 0
                G_height_model_temp[str(i) + bb] = xr.DataArray(
                    height_model,
                    coords=Gx_1.coords,
                    dims=Gx_1.dims,
                    name="height_model",
                )
            else:
                G_height_model_temp[str(i) + bb] = xr.DataArray(
                    Gx_1.y_model.data,
                    coords=Gx_1.coords,
                    dims=Gx_1.dims,
                    name="height_model",
                )

        G_height_model[bb] = xr.concat(G_height_model_temp.values(), dim="x").T

    Gx["height_model"] = xr.concat(G_height_model.values(), dim="beam").transpose(
        "eta", "beam", "x"
    )

    Gx_v2, B2_v2, B3_v2 = dict(), dict(), dict()

    for bb in Gx.beam.data:
        _logger.debug("bb: %s", bb)
        Gx_k = Gx.sel(beam=bb)
        Gh = Gx["height_model"].sel(beam=bb).T
        Gh_err = Gx_k["model_error_x"].T
        Gnans = np.isnan(Gx_k.y_model)

        concented_heights = Gh.data.reshape(Gh.data.size)
        concented_err = Gh_err.data.reshape(Gh.data.size)
        concented_x = (Gh.x + Gh.eta).data.reshape(Gh.data.size)

        dx = Gh.eta.diff("eta")[0].data
        continous_x_grid = np.arange(concented_x.min(), concented_x.max(), dx)
        continous_height_model = np.interp(
            continous_x_grid, concented_x, concented_heights
        )
        concented_err = np.interp(continous_x_grid, concented_x, concented_err)

        T3 = B3[bb]
        T3 = T3.sort_values("x")
        T3 = T3.sort_values("dist")

        T3["heights_c_model"] = np.interp(
            T3["dist"], continous_x_grid, continous_height_model
        )
        T3["heights_c_model_err"] = np.interp(
            T3["dist"], continous_x_grid, concented_err
        )
        T3["heights_c_residual"] = T3["heights_c_weighted_mean"] - T3["heights_c_model"]

        B3_v2[bb] = T3
        Gx_v2[bb] = Gx_k

    try:
        G_angle = xr.open_dataset(load_path_angle / (f"B05_{track_name}_angle_pdf.nc"))

    except ValueError as e:
        _logger.warning("%s no angle data found, skip angle corretion", e)
        theta = 0
        theta_flag = False
    else:
        font_for_pres()
        Ga_abs = (
            G_angle.weighted_angle_PDF_smth.isel(angle=G_angle.angle > 0).data
            + G_angle.weighted_angle_PDF_smth.isel(angle=G_angle.angle < 0).data[
                :, ::-1
            ]
        ) / 2
        Ga_abs = xr.DataArray(
            data=Ga_abs.T,
            dims=G_angle.dims,
            coords=G_angle.isel(angle=G_angle.angle > 0).coords,
        )

        Ga_abs_front = Ga_abs.isel(x=slice(0, 3))
        Ga_best = (Ga_abs_front * Ga_abs_front.N_data).sum(
            "x"
        ) / Ga_abs_front.N_data.sum("x")

        theta = Ga_best.angle[Ga_best.argmax()].data
        theta_flag = True

        font_for_print()
        F = M.FigureAxisXY(3, 5, view_scale=0.7)

        plt.subplot(2, 1, 1)
        plt.pcolor(Ga_abs)
        plt.xlabel("abs angle")
        plt.ylabel("x")

        ax = plt.subplot(2, 1, 2)
        Ga_best.plot()
        plt.title("angle front " + str(theta * 180 / np.pi), loc="left")
        ax.axvline(theta, color="red")
        F.save_light(path=plot_path, name="B06_angle_def")

    k_corrected, x_corrected = get_k_x_corrected(Gk, theta, theta_flag)

    # spectral save
    G5 = G_gFT_wmean.expand_dims(dim="beam", axis=1)
    G5.coords["beam"] = ["weighted_mean"]
    G5 = G5.assign_coords(N_photons=G5.N_photons)
    G5["N_photons"] = G5["N_photons"].expand_dims("beam")
    G5["N_per_stancil_fraction"] = G5["N_per_stancil_fraction"].expand_dims("beam")

    Gk_v2 = xr.merge([Gk, G5])

    Gk_v2 = Gk_v2.assign_coords(x_corrected=("x", x_corrected.data)).assign_coords(
        k_corrected=("k", k_corrected.data)
    )

    ## TODO: abstract to a function Github Iusse #117
    Gk_v2.attrs["best_guess_incident_angle"] = theta

    # save collected spectral data
    Gk_v2.to_netcdf(save_path / ("B06_" + track_name + "_gFT_k_corrected.nc"))

    # save real space data
    Gx.to_netcdf(save_path / ("B06_" + track_name + "_gFT_x_corrected.nc"))

    B06_ID_name = "B06_" + track_name
    table_names = [
        B06_ID_name + suffix for suffix in ["_B06_corrected_resid", "_binned_resid"]
    ]
    data = [B2_v2, B3_v2]
    for tablename, data in zip(table_names, data):
        save_table(data, tablename, save_path)

    MT.json_save(
        "B06_success",
        (plot_path / "../"),
        {"time": time.asctime(time.localtime(time.time()))},
    )
    _logger.info("done. saved target at %s../B06_success", plot_path)
    _logger.info("Done B06_correct_separate_var")


correct_separate_app = makeapp(run_B06_correct_separate_var, name="correct-separate")

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    correct_separate_app()
