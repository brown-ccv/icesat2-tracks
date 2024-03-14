#!/usr/bin/env python
"""
This file open a ICEsat2 track applied filters and corrections and returns smoothed photon heights on a regular grid in an .nc file.
This is python 3
"""

import itertools
import logging

from icesat2waves.config.startup import (
    mconfig,
    color_schemes,
    font_for_print,
    font_for_pres,
)


import h5py
import icesat2waves.tools.iotools as io
import xarray as xr
import numpy as np
from scipy.constants import g


from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
from numba import jit  # maybe for later optimizations?  # noqa: F401

from icesat2waves.tools import angle_optimizer

import icesat2waves.local_modules.m_tools_ph3 as MT
import icesat2waves.local_modules.m_general_ph3 as M
from icesat2waves.tools import angle_optimizer

import pandas as pd


import time

from typer import Option

from icesat2waves.clitools import (
    validate_batch_key,
    validate_output_dir,
    update_paths_mconfig,
    report_input_parameters,
    validate_track_name_steps_gt_1,
    makeapp,
)

_logger = logging.getLogger(__name__)


def run_B04_angle(
    track_name: str = Option(..., callback=validate_track_name_steps_gt_1),
    batch_key: str = Option(..., callback=validate_batch_key),
    ID_flag: bool = True,
    output_dir: str = Option(..., callback=validate_output_dir),
):
    """
    TODO: add docstring
    """

    color_schemes.colormaps2(21)

    col_dict = color_schemes.rels

    track_name, batch_key, test_flag = io.init_from_input(
        [
            None,
            track_name,
            batch_key,
            ID_flag,
        ]
    )

    kargs = {
        "track_name": track_name,
        "batch_key": batch_key,
        "ID_flag": ID_flag,
        "output_dir": output_dir,
    }
    report_input_parameters(**kargs)

    hemis, batch = batch_key.split("_")

    workdir, plotsdir = update_paths_mconfig(output_dir, mconfig)

    save_path = workdir / batch_key / "B04_angle"
    plot_path = plotsdir / hemis / batch_key / track_name
    save_path.mkdir(parents=True, exist_ok=True)
    plot_path.mkdir(parents=True, exist_ok=True)

    all_beams = mconfig["beams"]["all_beams"]
    beam_groups = mconfig["beams"]["groups"]

    load_path = workdir / batch_key / "B01_regrid"
    G_binned_store = h5py.File(load_path / (track_name + "_B01_binned.h5"), "r")
    G_binned = dict()
    for b in all_beams:
        G_binned[b] = io.get_beam_hdf_store(G_binned_store[b])
    G_binned_store.close()

    load_path = workdir / batch_key / "B02_spectra"
    xtrack, ktrack = [
        load_path / ("B02_" + track_name + f"_gFT_{var}.nc") for var in ["x", "k"]
    ]

    Gx = xr.load_dataset(xtrack)
    Gk = xr.load_dataset(ktrack)

    # load prior information
    load_path = workdir / batch_key / "A02_prior"

    try:
        Prior = MT.load_pandas_table_dict("/A02_" + track_name, str(load_path))[
            "priors_hindcast"
        ]
    except FileNotFoundError:
        _logger.critical("Prior not found. exit")
        MT.json_save(
            "B04_fail",
            plot_path,
            {
                "time": time.asctime(time.localtime(time.time())),
                "reason": "Prior not found",
            },
        )
        exit()

    if np.isnan(Prior["mean"]["dir"]):
        _logger.critical("Prior failed, entries are nan. exit.")
        MT.json_save(
            "B04_fail",
            plot_path,
            {
                "time": time.asctime(time.localtime(time.time())),
                "reason": "Prior not found",
            },
        )
        exit()

    #### Define Prior
    Pperiod = Prior.loc[["ptp0", "ptp1", "ptp2", "ptp3", "ptp4", "ptp5"]]["mean"]
    Pdir = Prior.loc[["pdp0", "pdp1", "pdp2", "pdp3", "pdp4", "pdp5"]]["mean"].astype(
        "float"
    )
    Pspread = Prior.loc[["pspr0", "pspr1", "pspr2", "pspr3", "pspr4", "pspr5"]]["mean"]

    Pperiod = Pperiod[~np.isnan(list(Pspread))]
    Pdir = Pdir[~np.isnan(list(Pspread))]
    Pspread = Pspread[~np.isnan(list(Pspread))]

    # this is a hack since the current data does not have a spread
    Pspread[Pspread == 0] = 20

    # reset dirs:
    Pdir[Pdir > 180] = Pdir[Pdir > 180] - 360
    Pdir[Pdir < -180] = Pdir[Pdir < -180] + 360

    # reorder dirs
    dir_best = [0]
    for dir in Pdir:
        ip = np.argmin(
            [
                abs(dir_best[-1] - dir),
                abs(dir_best[-1] - (dir - 360)),
                abs(dir_best[-1] - (dir + 360)),
            ]
        )
        new_dir = np.array([dir, (dir - 360), (dir + 360)])[ip]
        dir_best.append(new_dir)
    dir_best = np.array(dir_best[1:])

    if len(Pperiod) == 0:
        _logger.debug("constant peak wave number")
        kk = Gk.k
        Pwavenumber = kk * 0 + (2 * np.pi / (1 / Prior.loc["fp"]["mean"])) ** 2 / g
        dir_best = kk * 0 + Prior.loc["dp"]["mean"]
        dir_interp_smth = dir_interp = kk * 0 + Prior.loc["dp"]["mean"]
        spread_smth = spread_interp = kk * 0 + Prior.loc["spr"]["mean"]

    else:
        Pwavenumber = (2 * np.pi / Pperiod) ** 2 / g
        kk = Gk.k
        dir_interp = np.interp(
            kk, Pwavenumber[Pwavenumber.argsort()], dir_best[Pwavenumber.argsort()]
        )
        dir_interp_smth = M.runningmean(dir_interp, 30, tailcopy=True)
        dir_interp_smth[-1] = dir_interp_smth[-2]

        spread_interp = np.interp(
            kk,
            Pwavenumber[Pwavenumber.argsort()],
            Pspread[Pwavenumber.argsort()].astype("float"),
        )
        spread_smth = M.runningmean(spread_interp, 30, tailcopy=True)
        spread_smth[-1] = spread_smth[-2]

    font_for_pres()

    F = M.FigureAxisXY(5, 4.5, view_scale=0.5)
    plt.subplot(2, 1, 1)
    plt.title("Prior angle smoothed\n" + track_name, loc="left")

    plt.plot(Pwavenumber, dir_best, ".r", markersize=8)
    plt.plot(kk, dir_interp, "-", color="red", linewidth=0.8, zorder=11)
    plt.plot(kk, dir_interp_smth, color=color_schemes.green1)

    plt.fill_between(
        kk,
        dir_interp_smth - spread_smth,
        dir_interp_smth + spread_smth,
        zorder=1,
        color=color_schemes.green1,
        alpha=0.2,
    )
    plt.ylabel("Angle (deg)")

    ax2 = plt.subplot(2, 1, 2)
    plt.title("Prior angle adjusted ", loc="left")

    # adjust angle def:
    dir_interp_smth[dir_interp_smth > 180] = (
        dir_interp_smth[dir_interp_smth > 180] - 360
    )
    dir_interp_smth[dir_interp_smth < -180] = (
        dir_interp_smth[dir_interp_smth < -180] + 360
    )

    plt.fill_between(
        kk,
        dir_interp_smth - spread_smth,
        dir_interp_smth + spread_smth,
        zorder=1,
        color=color_schemes.green1,
        alpha=0.2,
    )
    plt.plot(kk, dir_interp_smth, ".", markersize=1, color=color_schemes.green1)

    ax2.axhline(85, color="gray", linewidth=2)
    ax2.axhline(-85, color="gray", linewidth=2)

    plt.ylabel("Angle (deg)")
    plt.xlabel("wavenumber ($2 \pi/\lambda$)")

    F.save_light(path=plot_path, name="B04_prior_angle")

    # save
    dir_interp_smth = xr.DataArray(
        data=dir_interp_smth * np.pi / 180,
        dims="k",
        coords={"k": kk},
        name="Prior_direction",
    )
    spread_smth = xr.DataArray(
        data=spread_smth * np.pi / 180,
        dims="k",
        coords={"k": kk},
        name="Prior_spread",
    )
    Prior_smth = xr.merge([dir_interp_smth, spread_smth])

    prior_angle = Prior_smth.Prior_direction * 180 / np.pi
    if (abs(prior_angle) > 80).all():
        _logger.critical(
            "Prior angle is %s %s. Quit.",
            prior_angle.min().data,
            prior_angle.max().data,
        )
        dd_save = {
            "time": time.asctime(time.localtime(time.time())),
            "angle": list(
                [
                    float(prior_angle.min().data),
                    float(prior_angle.max().data),
                    float(prior_angle.median()),
                ]
            ),
        }
        MT.json_save("B04_fail", plot_path, dd_save)
        _logger.critical("terminating")
        exit()

    # define parameter range
    params_dict = {
        "alpha": [-0.85 * np.pi / 2, 0.85 * np.pi / 2, 5],
        "phase": [0, 2 * np.pi, 10],
    }

    alpha_dx = 0.02
    max_wavenumbers = 25

    sample_flag = True
    optimize_flag = False
    brute_flag = False

    plot_flag = False

    # Nworkers = 6 for later parallelization?
    N_sample_chain = 300
    N_sample_chain_burn = 30

    max_x_pos = 8
    x_pos_jump = 2

    def make_fake_data(xi, group):
        ki = Gk.k[0:2]

        bins = np.arange(
            params_dict["alpha"][0], params_dict["alpha"][1] + alpha_dx, alpha_dx
        )
        bins_pos = bins[0:-1] + np.diff(bins) / 2
        marginal_stack = xr.DataArray(
            np.nan * np.vstack([bins_pos, bins_pos]).T,
            dims=("angle", "k"),
            coords={"angle": bins_pos, "k": ki.data},
        )

        group_name = str("group" + group[0].split("gt")[1].split("l")[0])
        marginal_stack.coords["beam_group"] = group_name
        marginal_stack.coords["x"] = xi
        marginal_stack.name = "marginals"
        marginal_stack.expand_dims(dim="x", axis=2).expand_dims(
            dim="beam_group", axis=3
        )
        return marginal_stack

    def define_wavenumber_weights_tot_var(
        dd, m=3, variance_frac=0.33, k_upper_lim=None, verbose=False
    ):
        """
        return peaks of a power spectrum dd that in the format such that they can be used as weights for the frequencies based fitting

        inputs:
        dd             xarray with PSD as data amd coordinate wavenumber k
        m               running mean half-width in gridpoints
        variance_frac  (0 to 1) How much variance should be explained by the returned peaks
        verbose        if true it plots some stuff


        return:
        mask           size of dd. where True the data is identified as having significant amplitude
        k              wanumbers where mask is true
        dd_rm          smoothed version of dd
        positions      positions where of significant data in array
        """

        if len(dd.shape) == 2:
            dd_use = dd.mean("beam")

        if m is None:
            dd_rm = dd_use.data
        else:
            dd_rm = M.runningmean(dd_use, m, tailcopy=True)

        k = dd_use.k[~np.isnan(dd_rm)].data
        dd_rm = dd_rm[~np.isnan(dd_rm)]

        orders = dd_rm.argsort()[::-1]
        var_mask = dd_rm[orders].cumsum() / dd_rm.sum() < variance_frac
        pos_cumsum = orders[var_mask]
        mask = var_mask[orders.argsort()]
        if k_upper_lim is not None:
            mask = (k < k_upper_lim) & mask

        if verbose:
            plt.plot(
                dd.k,
                dd,
                "-",
                color=col_dict[str(amp_data.beam[0].data)],
                markersize=20,
                alpha=0.6,
            )
            plt.plot(k, dd_rm, "-k", markersize=20)

            plt.plot(k[mask], dd_rm[mask], ".r", markersize=10, zorder=12)
            if k_upper_lim is not None:
                plt.gca().axvline(k_upper_lim, color="black")

        return mask, k, dd_rm, pos_cumsum

    def define_wavenumber_weights_threshold(dd, m=3, Nstd=2, verbose=False):
        if m is None:
            dd_rm = dd
        else:
            dd_rm = M.runningmean(dd, m, tailcopy=True)

        k = dd.k[~np.isnan(dd_rm)]
        dd_rm = dd_rm[~np.isnan(dd_rm)]

        treshold = np.nanmean(dd_rm) + np.nanstd(dd_rm) * Nstd
        mask = dd_rm > treshold

        if verbose:
            plt.plot(dd.k, dd, "-k", markersize=20)
            plt.plot(k, dd_rm, "-b", markersize=20)

            k_list = k[mask]
            dd_list = dd_rm[mask]

            plt.plot(k_list, dd_list, ".r", markersize=10, zorder=12)

        return mask, k, dd_rm, np.arange(0, mask.size)[mask]

    def plot_instance(
        z_model,
        fargs,
        key,
        SM,
        non_dim=False,
        title_str=None,
        brute=False,
        optimze=False,
        sample=False,
        view_scale=0.3,
    ):
        x_concat, y_concat, z_concat = fargs

        F = M.FigureAxisXY(5, 6, view_scale=view_scale, container=True)
        plt.suptitle(title_str)
        gs = GridSpec(4, 3, wspace=0.4, hspace=1.2)
        F.gs = gs

        col_list = itertools.cycle(
            [
                color_schemes.cascade2,
                color_schemes.rascade2,
                color_schemes.cascade1,
                color_schemes.rascade1,
                color_schemes.cascade3,
                color_schemes.rascade3,
            ]
        )

        beam_list = list(set(y_concat))
        for y_pos, pos in zip(beam_list, [gs[0, :], gs[1, :]]):
            F.ax2 = F.fig.add_subplot(pos)

            plt.title(str(y_pos))

            plt.plot(
                x_concat[y_concat == y_pos],
                z_concat[y_concat == y_pos],
                c=color_schemes.gray,
                linewidth=1,
            )
            plt.plot(
                x_concat[y_concat == y_pos],
                z_model[y_concat == y_pos],
                "-",
                c=next(col_list),
            )
            plt.xlim(x_concat[y_concat == y_pos][0], x_concat[y_concat == y_pos][-1])

        plt.xlabel("meter")
        F.ax3 = F.fig.add_subplot(gs[2:, 0:-1])
        if brute is True:
            plt.title("Brute-force costs", loc="left")
            SM.plot_brute(
                marker=".", color="blue", markersize=15, label="Brute", zorder=10
            )

        if optimze is True:
            SM.plot_optimze(color="r", markersize=10, zorder=12, label="Dual Annealing")

        if sample is True:
            SM.plot_sample(
                markersize=2, linewidth=0.8, alpha=0.2, color="black", zorder=8
            )

        F.ax4 = F.fig.add_subplot(gs[2:, -1])
        return F

    # isolate x positions with data
    data_mask = Gk.gFT_PSD_data.mean("k")
    data_mask.coords["beam_group"] = (
        "beam",
        ["beam_group" + g_[2] for g_ in data_mask.beam.data],
    )
    data_mask_group = data_mask.groupby("beam_group").mean(skipna=False)
    # these stencils are actually used
    data_sel_mask = data_mask_group.sum("beam_group") != 0

    x_list = data_sel_mask.x[data_sel_mask]  # iterate over these x positions
    x_list_flag = ~np.isnan(
        data_mask_group.sel(x=x_list)
    )  # flag that is False if there is no data

    #### limit number of x coordinates

    x_list = x_list[::x_pos_jump]
    if len(x_list) > max_x_pos:
        x_list = x_list[0:max_x_pos]
    x_list_flag = x_list_flag.sel(x=x_list)

    # plot
    font_for_print()
    F = M.FigureAxisXY(5.5, 3, view_scale=0.8)
    plt.suptitle(track_name)
    ax1 = plt.subplot(2, 1, 1)
    plt.title("Data in Beam", loc="left")
    plt.pcolormesh(data_mask.x / 1e3, data_mask.beam, data_mask, cmap=plt.cm.OrRd)
    for i in np.arange(1.5, 6, 2):
        ax1.axhline(i, color="black", linewidth=0.5)
    plt.xlabel("Distance from Ice Edge")

    ax2 = plt.subplot(2, 1, 2)
    plt.title("Data in Group", loc="left")
    plt.pcolormesh(
        data_mask.x / 1e3,
        data_mask_group.beam_group,
        data_mask_group,
        cmap=plt.cm.OrRd,
    )

    for i in np.arange(0.5, 3, 1):
        ax2.axhline(i, color="black", linewidth=0.5)

    plt.plot(
        x_list / 1e3,
        x_list * 0 + 0,
        ".",
        markersize=2,
        color=color_schemes.cascade1,
    )
    plt.plot(
        x_list / 1e3,
        x_list * 0 + 1,
        ".",
        markersize=2,
        color=color_schemes.cascade1,
    )
    plt.plot(
        x_list / 1e3,
        x_list * 0 + 2,
        ".",
        markersize=2,
        color=color_schemes.cascade1,
    )

    plt.xlabel("Distance from Ice Edge")

    F.save_pup(path=plot_path, name="B04_data_avail")

    Marginals = dict()
    L_collect = dict()

    group_number = np.arange(len(beam_groups))
    ggg, xxx = np.meshgrid(group_number, x_list.data)

    for gi in zip(ggg.flatten(), xxx.flatten()):
        _logger.debug("gi = %s", gi)

        group, xi = beam_groups[gi[0]], gi[1]

        if bool(x_list_flag.sel(x=xi).isel(beam_group=gi[0]).data) is False:
            _logger.debug("no data, fill with dummy")
            ikey = str(xi) + "_" + "_".join(group)
            Marginals[ikey] = make_fake_data(xi, group)
            continue

        GGx = Gx.sel(beam=group).sel(x=xi)
        GGk = Gk.sel(beam=group).sel(x=xi)

        ### define data
        # normalize data
        key = "y_data"
        amp_Z = (GGx[key] - GGx[key].mean(["eta"])) / GGx[key].std(["eta"])

        # define x,y positions
        eta_2d = GGx.eta + GGx.x_coord - GGx.x_coord.mean()
        nu_2d = GGx.eta * 0 + GGx.y_coord - GGx.y_coord.mean()

        # repack as np arrays
        x_concat = eta_2d.data.T.flatten()
        y_concat = nu_2d.data.T.flatten()
        z_concat = amp_Z.data.flatten()

        x_concat = x_concat[~np.isnan(z_concat)]
        y_concat = y_concat[~np.isnan(z_concat)]
        z_concat = z_concat[~np.isnan(z_concat)]
        N_data = x_concat.size

        if np.isnan(z_concat).sum() != 0:
            raise ValueError("There are still nans")

        mean_dist = (nu_2d.isel(beam=0) - nu_2d.isel(beam=1)).mean().data
        k_upper_lim = 2 * np.pi / (mean_dist * 1)

        _logger.debug("k_upper_lim %s", k_upper_lim)

        # variance method
        amp_data = np.sqrt(GGk.gFT_cos_coeff**2 + GGk.gFT_sin_coeff**2)
        mask, k, weights, positions = define_wavenumber_weights_tot_var(
            amp_data,
            m=1,
            k_upper_lim=k_upper_lim,
            variance_frac=0.20,
            verbose=False,
        )

        if len(k[mask]) == 0:
            _logger.debug("no good k found, fill with dummy")
            ikey = str(xi) + "_" + "_".join(group)
            Marginals[ikey] = make_fake_data(xi, group)
            continue

        SM = angle_optimizer.SampleWithMcmc(params_dict)
        SM.set_objective_func(angle_optimizer.objective_func)
        nan_list = np.isnan(x_concat) | np.isnan(y_concat) | np.isnan(y_concat)
        x_concat[nan_list] = []
        y_concat[nan_list] = []
        z_concat[nan_list] = []
        SM.fitting_args = fitting_args = (x_concat, y_concat, z_concat)

        # test:
        k_prime_max = 0.02
        amp_Z = 1
        prior_sel = {
            "alpha": (
                Prior_smth.sel(k=k_prime_max, method="nearest").Prior_direction.data,
                Prior_smth.sel(k=k_prime_max, method="nearest").Prior_spread.data,
            )
        }
        SM.fitting_kargs = {
            "prior": prior_sel,
            "prior_weight": 3,
        }  # not sure this might be used somewhere. CP

        # test if it works
        SM.params.add(
            "K_prime",
            k_prime_max,
            vary=False,
            min=k_prime_max * 0.5,
            max=k_prime_max * 1.5,
        )
        SM.params.add("K_amp", amp_Z, vary=False, min=amp_Z * 0.0, max=amp_Z * 5)
        try:
            SM.test_objective_func()
        except ValueError:
            raise ValueError("Objective function test fails")

        def get_instance(k_pair):
            k_prime_max, Z_max = k_pair

            prior_sel = {
                "alpha": (
                    Prior_smth.sel(
                        k=k_prime_max, method="nearest"
                    ).Prior_direction.data,
                    Prior_smth.sel(k=k_prime_max, method="nearest").Prior_spread.data,
                )
            }

            SM.fitting_kargs = {
                "prior": prior_sel,
                "prior_weight": 2,
            }  # not sure this might be used somewhere. CP

            amp_Z = 1
            SM.params.add(
                "K_prime",
                k_prime_max,
                vary=False,
                min=k_prime_max * 0.5,
                max=k_prime_max * 1.5,
            )
            SM.params.add("K_amp", amp_Z, vary=False, min=amp_Z * 0.0, max=amp_Z * 5)

            L_sample_i = None
            L_optimize_i = None
            L_brute_i = None
            if sample_flag:
                SM.sample(
                    verbose=False,
                    steps=N_sample_chain,
                    progress=False,
                    workers=None,
                )
                L_sample_i = list(
                    SM.fitter.params.valuesdict().values()
                )  # mcmc results

            elif optimize_flag:
                SM.optimize(verbose=False)
                L_optimize_i = list(
                    SM.fitter_optimize.params.valuesdict().values()
                )  # mcmc results

            elif brute_flag:
                SM.brute(verbose=False)
                L_brute_i = list(
                    SM.fitter_brute.params.valuesdict().values()
                )  # mcmc results
            else:
                raise ValueError(
                    "non of sample_flag,optimize_flag, or brute_flag  are True"
                )

            y_hist, bins, bins_pos = SM.get_marginal_dist(
                "alpha", alpha_dx, burn=N_sample_chain_burn, plot_flag=False
            )
            fitter = SM.fitter  # MCMC results
            z_model = SM.objective_func(fitter.params, *fitting_args, test_flag=True)
            cost = (fitter.residual**2).sum() / (z_concat**2).sum()

            if plot_flag:
                F = plot_instance(
                    z_model,
                    fitting_args,
                    "y_data_normed",
                    SM,
                    brute=brute_flag,
                    optimze=optimize_flag,
                    sample=sample_flag,
                    title_str="k=" + str(np.round(k_prime_max, 4)),
                    view_scale=0.6,
                )

                if not prior_sel:  # check if prior is empty
                    F.ax3.axhline(
                        prior_sel["alpha"][0],
                        color="green",
                        linewidth=2,
                        label="Prior",
                    )
                    F.ax3.axhline(
                        prior_sel["alpha"][0] - prior_sel["alpha"][1],
                        color="green",
                        linewidth=0.7,
                    )
                    F.ax3.axhline(
                        prior_sel["alpha"][0] + prior_sel["alpha"][1],
                        color="green",
                        linewidth=0.7,
                    )

                F.ax3.axhline(
                    fitter.params["alpha"].min, color="gray", linewidth=2, alpha=0.6
                )
                F.ax3.axhline(
                    fitter.params["alpha"].max, color="gray", linewidth=2, alpha=0.6
                )

                plt.sca(F.ax3)
                plt.legend()
                plt.xlabel("Phase")
                plt.ylabel("Angle")
                plt.xlim(0, np.pi * 2)

                plt.sca(F.ax4)
                plt.xlabel("Density")
                plt.stairs(y_hist, bins, orientation="horizontal", color="k")

                F.ax4.axhline(
                    fitter.params["alpha"].min, color="gray", linewidth=2, alpha=0.6
                )
                F.ax4.axhline(
                    fitter.params["alpha"].max, color="gray", linewidth=2, alpha=0.6
                )

                F.ax3.set_ylim(
                    min(-np.pi / 2, prior_sel["alpha"][0] - 0.2),
                    max(np.pi / 2, prior_sel["alpha"][0] + 0.2),
                )
                F.ax4.set_ylim(
                    min(-np.pi / 2, prior_sel["alpha"][0] - 0.2),
                    max(np.pi / 2, prior_sel["alpha"][0] + 0.2),
                )

                plt.show()
                F.save_light(
                    path=plot_path, name=track_name + "_fit_k" + str(k_prime_max)
                )

            marginal_stack_i = xr.DataArray(
                y_hist, dims=("angle"), coords={"angle": bins_pos}
            )
            marginal_stack_i.coords["k"] = np.array(k_prime_max)

            rdict = {
                "marginal_stack_i": marginal_stack_i,
                "L_sample_i": L_sample_i,
                "L_optimize_i": L_optimize_i,
                "L_brute_i": L_brute_i,
                "cost": cost,
            }
            return k_prime_max, rdict

        k_list, weight_list = k[mask], weights[mask]
        _logger.debug("# of wavenumber: %s", len(k_list))
        if len(k_list) > max_wavenumbers:
            _logger.debug("cut wavenumber list to 20")
            k_list = k_list[0:max_wavenumbers]
            weight_list = weight_list[0:max_wavenumbers]

        # # parallel version tends to fail...
        # Let's keep this in case we decided to work on parallelize it
        # with futures.ProcessPoolExecutor(max_workers=Nworkers) as executor:
        #     A = dict( executor.map(get_instance, zip(k_list, weight_list)   ))

        A = dict()
        for k_pair in zip(k_list, weight_list):
            kk, I = get_instance(k_pair)
            A[kk] = I

        cost_stack = dict()
        marginal_stack = dict()
        L_sample = pd.DataFrame(index=["alpha", "group_phase", "K_prime", "K_amp"])
        L_optimize = pd.DataFrame(index=["alpha", "group_phase", "K_prime", "K_amp"])
        L_brute = pd.DataFrame(index=["alpha", "group_phase", "K_prime", "K_amp"])

        for kk, I in A.items():
            L_sample[kk] = I["L_sample_i"]
            L_optimize[kk] = I["L_optimize_i"]
            L_brute[kk] = I["L_brute_i"]

            marginal_stack[kk] = I["marginal_stack_i"]
            cost_stack[kk] = I["cost"]

        # ## add beam_group dimension
        marginal_stack = xr.concat(marginal_stack.values(), dim="k").sortby("k")
        L_sample = L_sample.T.sort_values("K_prime")
        L_optimize = L_optimize.T.sort_values("K_prime")
        L_brute = L_brute.T.sort_values("K_prime")

        _logger.info("done with group %s,  xi = %s", group, xi / 1e3)

        # collect
        ikey = str(xi) + "_" + "_".join(group)

        marginal_stack.name = "marginals"
        marginal_stack = marginal_stack.to_dataset()
        marginal_stack["cost"] = (("k"), list(cost_stack.values()))
        marginal_stack["weight"] = (("k"), weight_list)

        group_name = str("group" + group[0].split("gt")[1].split("l")[0])
        marginal_stack.coords["beam_group"] = group_name
        marginal_stack.coords["x"] = xi

        Marginals[ikey] = marginal_stack.expand_dims(dim="x", axis=0).expand_dims(
            dim="beam_group", axis=1
        )
        Marginals[ikey].coords["N_data"] = (
            ("x", "beam_group"),
            np.expand_dims(np.expand_dims(N_data, 0), 1),
        )

        L_sample["cost"] = cost_stack
        L_sample["weight"] = weight_list
        L_collect[group_name, str(int(xi))] = L_sample

    MM = xr.merge(Marginals.values())
    MM = xr.merge([MM, Prior_smth])

    save_name = "B04_" + track_name
    MM.to_netcdf(save_path / (save_name + "_marginals.nc"))

    try:
        LL = pd.concat(L_collect)
        MT.save_pandas_table(
            {"L_sample": LL}, save_name + "_res_table", str(save_path)
        )  # TODO: clean up save_pandas_table to use pathlib
    except Exception as e:
        _logger.warning("This is a warning: %s", e)
    else:
        # plotting with LL
        font_for_print()
        F = M.FigureAxisXY(6, 5.5, view_scale=0.7, container=True)

        gs = GridSpec(4, 6, wspace=0.2, hspace=0.8)

        ax0 = F.fig.add_subplot(gs[0:2, -1])
        ax0.tick_params(labelleft=False)

        klims = 0, LL["K_prime"].max() * 1.2

        for g_ in MM.beam_group:
            MMi = MM.sel(beam_group=g_)
            plt.plot(
                MMi.weight.T,
                MMi.k,
                ".",
                color=col_dict[str(g_.data)],
                markersize=3,
                linewidth=0.8,
            )

        plt.xlabel("Power")
        plt.ylim(klims)

        ax1 = F.fig.add_subplot(gs[0:2, 0:-1])

        for g_ in MM.beam_group:
            Li = LL.loc[str(g_.data)]

            angle_list = np.array(Li["alpha"]) * 180 / np.pi
            kk_list = np.array(Li["K_prime"])
            weight_list_i = np.array(Li["weight"])

            plt.scatter(
                angle_list,
                kk_list,
                s=(weight_list_i * 8e1) ** 2,
                c=col_dict[str(g_.data)],
                label="mode " + str(g_.data),
            )

        dir_best[dir_best > 180] = dir_best[dir_best > 180] - 360
        plt.plot(dir_best, Pwavenumber, ".r", markersize=6)

        dir_interp[dir_interp > 180] = dir_interp[dir_interp > 180] - 360
        plt.plot(dir_interp, Gk.k, "-", color="red", linewidth=0.3, zorder=11)

        plt.fill_betweenx(
            Gk.k,
            (dir_interp_smth - spread_smth) * 180 / np.pi,
            (dir_interp_smth + spread_smth) * 180 / np.pi,
            zorder=1,
            color=color_schemes.green1,
            alpha=0.2,
        )
        plt.plot(
            dir_interp_smth * 180 / np.pi,
            Gk.k,
            ".",
            markersize=1,
            color=color_schemes.green1,
        )

        ax1.axvline(85, color="gray", linewidth=2)
        ax1.axvline(-85, color="gray", linewidth=2)

        plt.legend()
        plt.ylabel("wavenumber (deg)")
        plt.xlabel("Angle (deg)")

        plt.ylim(klims)

        prior_angle_str = str(np.round((prior_sel["alpha"][0]) * 180 / np.pi))
        plt.title(track_name + "\nprior=" + prior_angle_str + "deg", loc="left")

        plt.xlim(min([-90, np.nanmin(dir_best)]), max([np.nanmax(dir_best), 90]))

        ax3 = F.fig.add_subplot(gs[2, 0:-1])  # can the assignment be removed? CP

        for g_ in MM.beam_group:
            MMi = MM.sel(beam_group=g_)
            weighted_margins = (MMi.marginals * MMi.weight).sum(
                ["x", "k"]
            ) / MMi.weight.sum(["x", "k"])
            plt.plot(
                MMi.angle * 180 / np.pi,
                weighted_margins,
                ".",
                color=col_dict[str(g_.data)],
                markersize=2,
                linewidth=0.8,
            )

        plt.ylabel("Density")
        plt.title("weight margins", loc="left")

        plt.xlim(-90, 90)

        ax3 = F.fig.add_subplot(
            gs[-1, 0:-1]
        )  # can the assignment be removed? Not used later. CP

        for g_ in MM.beam_group:
            MMi = MM.sel(beam_group=g_)
            weighted_margins = MMi.marginals.mean(["x", "k"])
            plt.plot(
                MMi.angle * 180 / np.pi,
                weighted_margins,
                ".",
                color=col_dict[str(g_.data)],
                markersize=2,
                linewidth=0.8,
            )

        plt.ylabel("Density")
        plt.xlabel("Angle (deg)")
        plt.title("unweighted margins", loc="left")

        plt.xlim(-90, 90)

        F.save_pup(path=plot_path, name="B04_marginal_distributions")

        MT.json_save(
            "B04_success",
            plot_path,
            {"time": "time.asctime( time.localtime(time.time()) )"},
        )


make_b04_angle_app = makeapp(run_B04_angle, name="B04_angle")

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    make_b04_angle_app()
