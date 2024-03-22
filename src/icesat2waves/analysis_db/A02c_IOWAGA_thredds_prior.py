#!/usr/bin/env python3

import datetime
import logging
from pathlib import Path

import h5py
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import xarray as xr
from siphon.catalog import TDSCatalog
import typer

from icesat2waves.config.startup import mconfig
import icesat2waves.tools.iotools as io
import icesat2waves.tools.wave_tools as waves
import icesat2waves.local_modules.m_tools_ph3 as MT
import icesat2waves.local_modules.m_general_ph3 as M
from icesat2waves.config.startup import color_schemes
from icesat2waves.config.startup import font_for_print

from icesat2waves.clitools import (
    validate_batch_key,
    validate_output_dir,
    update_paths_mconfig,
    report_input_parameters,
    validate_track_name_steps_gt_1,
    makeapp,
)

_logger = logging.getLogger(__name__)


def get_iowaga(data_url, dataset_key):
    ## load WW3 data
    # ECMWF hindcast
    # data_url = 'https://tds3.ifremer.fr/thredds/IOWAGA-WW3-HINDCAST/IOWAGA-GLOBAL_ECMWF-WW3-HINDCAST_FULL_TIME_SERIE.xml'

    # CFSR hindcast
    # data_url = 'https://tds3.ifremer.fr/thredds/IOWAGA-WW3-HINDCAST/IOWAGA-GLOBAL_CFSR-WW3-HINDCAST_FULL_TIME_SERIE.xml'
    cat = TDSCatalog(data_url)
    ncss = cat.datasets[dataset_key].remote_access(use_xarray=True)

    var_list = [
        "dir",
        "dp",
        "fp",
        "hs",
        "ice",
        "spr",
        "t01",
        "t02",
        "plp0",
        "pdir0",
        "pdir1",
        "pdir2",
        "pdir3",
        "pdir4",
        "pdir5",
        "pspr0",
        "pspr1",
        "pspr2",
        "pspr3",
        "pspr4",
        "pspr5",
        "ptp0",
        "ptp1",
        "ptp2",
        "ptp3",
        "ptp4",
        "ptp5",
        "phs0",
        "phs1",
        "phs2",
        "phs3",
        "phs4",
        "phs5",
    ]
    names_map = {
        "pdir0": "pdp0",
        "pdir1": "pdp1",
        "pdir2": "pdp2",
        "pdir3": "pdp3",
        "pdir4": "pdp4",
        "pdir5": "pdp5",
    }
    IOWAGA = ncss[var_list]
    IOWAGA["time"] = np.array([np.datetime64(k0) for k0 in IOWAGA.time.data]).astype(
        "M8[h]"
    )
    IOWAGA = IOWAGA.rename(name_dict=names_map)

    return IOWAGA


def sel_data(Ibeam, lon_range, lat_range, time_range, timestamp=None):
    """
    this method returns the selected data in the lon-lat box at an interpolated timestamp
    """
    # TODO: refactor to avoid code duplication
    lon_flag = (lon_range[0] < Ibeam.longitude.data) & (
        Ibeam.longitude.data < lon_range[1]
    )
    lat_flag = (lat_range[0] < Ibeam.latitude.data) & (
        Ibeam.latitude.data < lat_range[1]
    )
    time_flag = (time_range[0] < Ibeam.time.data) & (Ibeam.time.data < time_range[1])

    if timestamp is None:
        Ibeam = Ibeam.isel(latitude=lat_flag, longitude=lon_flag)
    else:
        Ibeam = (
            Ibeam.isel(latitude=lat_flag, longitude=lon_flag, time=time_flag)
            .sortby("time")
            .interp(time=np.datetime64(timestamp))
        )
    return Ibeam


def build_prior(Tend: pd.DataFrame):
    """
    Build a prior dictionary from the Tend DataFrame

    Args:
        Tend (pd.DataFrame): DataFrame containing the mean and standard deviation of the wave parameters

    Returns:
        dict: A dictionary containing the prior wave parameters
    """
    Prior = dict()
    key_mapping = {
        "incident_angle": "dp",
        "spread": "spr",
        "Hs": "hs",
        "center_lon": "lon",
        "center_lat": "lat",
    }

    # populate Prior
    for key, tend_key in key_mapping.items():
        Prior[key] = {
            "value": Tend["mean"][tend_key].astype("float"),
            "name": Tend["name"][tend_key],
        }

    # Handle "peak_period" separately
    Prior["peak_period"] = {
        "value": 1 / Tend["mean"]["fp"].astype("float"),
        "name": "1/" + Tend["name"]["fp"],
    }

    return Prior


def calculate_ranges(hemis, G1, dlon_deg, dlat_deg, dlat_deg_prior):
    lon_range = G1["lons"].min() - dlon_deg, G1["lons"].max() + dlon_deg

    if hemis == "SH":
        lat_range = np.sign(G1["lats"].min()) * 78, G1["lats"].max() + dlat_deg[1]
    else:
        lat_range = G1["lats"].min() - dlat_deg[0], G1["lats"].max() + dlat_deg[1]

    lat_range_prior = (
        G1["lats"].min() - dlat_deg_prior[0],
        G1["lats"].max() + dlat_deg_prior[1],
    )

    return lon_range, lat_range, lat_range_prior


def define_timestamp_and_time_range(ID, dtime):
    timestamp = pd.to_datetime(ID["pars"]["start"]["delta_time"], unit="s")
    time_range = np.datetime64(timestamp) - np.timedelta64(dtime, "h"), np.datetime64(
        timestamp
    ) + np.timedelta64(dtime, "h")
    return timestamp, time_range


def run_A02c_IOWAGA_thredds_prior(
    track_name: str = typer.Option(..., callback=validate_track_name_steps_gt_1),
    batch_key: str = typer.Option(..., callback=validate_batch_key),
    ID_flag: bool = True,
    output_dir: str = typer.Option(..., callback=validate_output_dir),
):
    """
    TODO: add docstring
    """
    color_schemes.colormaps2(21)
    matplotlib.use("Agg")  # prevent plot windows from opening

    dtime = 4  # in hours

    # IOWAGA constants
    data_url = "https://tds3.ifremer.fr/thredds/IOWAGA-WW3-FORECAST/IOWAGA-WW3-FORECAST_GLOBMULTI_GLOB-30M.xml"
    dataset_key = "IOWAGA-WW3-FORECAST_GLOBMULTI_GLOB-30M_FIELD_NC_MARC_WW3-GLOB-30M"

    track_name, batch_key, _ = io.init_from_input(
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

    track_name_short = track_name[0:-16]

    workdir, plotsdir = update_paths_mconfig(output_dir, mconfig)

    ID, track_names, hemis, batch = io.init_data(
        str(track_name), str(batch_key), str(ID_flag), str(workdir)
    )  # TODO: clean up application of str() to all arguments

    kwargs = {
        "ID": ID,
        "track_names": track_names,
        "hemis": hemis,
        "batch": batch,
        "mconfig": workdir,
        "heading": "** Revised input parameters:",
    }
    report_input_parameters(**kwargs)

    hemis, batch = batch_key.split("_")

    save_path = Path(workdir, batch_key, "A02_prior")
    plot_path = Path(plotsdir, hemis, batch_key, track_name)
    save_name = "A02_" + track_name
    plot_name = "A02_" + track_name_short
    plot_path.mkdir(parents=True, exist_ok=True)
    save_path.mkdir(parents=True, exist_ok=True)

    all_beams = mconfig["beams"]["all_beams"]

    load_path = Path(workdir, batch_key, "B01_regrid")
    with h5py.File(load_path / (track_name + "_B01_binned.h5"), "r") as Gd:

        # Select the beam with the minimum absolute latitude for each beam in the dataset Gd
        # and store the corresponding row as a dictionary in G1? CP
        G1 = {
            b: io.get_beam_hdf_store(Gd[b]).iloc[
                abs(io.get_beam_hdf_store(Gd[b])["lats"]).argmin()
            ]
            for b in all_beams
        }

    G1 = pd.DataFrame.from_dict(G1).T

    if hemis == "SH":
        lon_range, lat_range, lat_range_prior = calculate_ranges(
            hemis=hemis, G1=G1, dlon_deg=1, dlat_deg=(30, 5), dlat_deg_prior=(2, 1)
        )
    else:
        lon_range, lat_range, lat_range_prior = calculate_ranges(
            hemis=hemis, G1=G1, dlon_deg=2, dlat_deg=(20, 20), dlat_deg_prior=(2, 1)
        )

    IOWAGA = get_iowaga(data_url=data_url, dataset_key=dataset_key)

    timestamp, time_range = define_timestamp_and_time_range(ID, dtime)

    # TODO: refactor this try-except block -- too much complexity within. CP
    try:
        G_beam = sel_data(
            Ibeam=IOWAGA,
            lon_range=lon_range,
            lat_range=lat_range,
            time_range=time_range,
            timestamp=timestamp,
        ).load()
        G_prior = sel_data(
            Ibeam=G_beam,
            lon_range=lon_range,
            lat_range=lat_range_prior,
            time_range=time_range,
        )

        if hemis == "SH":
            # create Ice mask
            ice_mask = (G_beam.ice > 0) | np.isnan(G_beam.ice)

            lats = list(ice_mask.latitude.data)
            lats.sort(reverse=True)

            # find 1st latitude that is completely full with sea ice.
            ice_lat_pos = next(
                (
                    i
                    for i, j in enumerate(
                        (ice_mask.sum("longitude") == ice_mask.longitude.size).sel(
                            latitude=lats
                        )
                    )
                    if j
                ),
                None,
            )
            # recreate lat mask based on this criteria
            lat_mask = lats < lats[ice_lat_pos]
            lat_mask = xr.DataArray(
                lat_mask.repeat(ice_mask.longitude.size).reshape(ice_mask.shape),
                dims=ice_mask.dims,
                coords=ice_mask.coords,
            )
            lat_mask["latitude"] = lats

            # combine ice mask and new lat mask
            ice_mask = ice_mask + lat_mask

        else:
            ice_mask = np.isnan(G_beam.ice)
            lats = ice_mask.latitude

            # find closed latitude with with non-nan data
            ice_lat_pos = (
                abs(
                    lats.where(ice_mask.sum("longitude") > 4, np.nan)
                    - np.array(lat_range).mean()
                )
                .argmin()
                .data
            )

            # redefine lat-range
            lat_range = lats[ice_lat_pos].data - 2, lats[ice_lat_pos].data + 2
            lat_flag2 = (lat_range[0] < lats.data) & (lats.data < lat_range[1])

            lat_mask = xr.DataArray(
                lat_flag2.repeat(ice_mask.longitude.size).reshape(ice_mask.shape),
                dims=ice_mask.dims,
                coords=ice_mask.coords,
            )
            lat_mask["latitude"] = lats

        # plot 1st figure
        def draw_range(lon_range, lat_range, *args, **kwargs):
            plt.plot(
                [
                    lon_range[0],
                    lon_range[1],
                    lon_range[1],
                    lon_range[0],
                    lon_range[0],
                ],
                [
                    lat_range[0],
                    lat_range[0],
                    lat_range[1],
                    lat_range[1],
                    lat_range[0],
                ],
                *args,
                **kwargs,
            )

        dir_clev = np.arange(0, 380, 20)
        f_clev = np.arange(1 / 40, 1 / 5, 0.01)
        fvar = ["ice", "dir", "dp", "spr", "fp", "hs"]
        fcmap = [
            plt.cm.Blues_r,
            color_schemes.circle_medium_triple,
            color_schemes.circle_medium_triple,
            plt.cm.Blues,
            plt.cm.Blues,
            plt.cm.Blues,
        ]
        fpos = [0, 1, 2, 3, 4, 5]
        clevs = [
            np.arange(0, 1, 0.2),
            dir_clev,
            dir_clev,
            np.arange(0, 90, 10),
            f_clev,
            np.arange(0.5, 9, 0.5),
        ]

        font_for_print()

        F = M.FigureAxisXY(4, 3.5, view_scale=0.9, container=True)

        file_name_base = "LOPS_WW3-GLOB-30M_"
        plt.suptitle(
            track_name_short + " | " + file_name_base[0:-1].replace("_", " "), y=1.3
        )
        lon, lat = G_beam.longitude, G_beam.latitude

        gs = GridSpec(9, 6, wspace=0.1, hspace=0.4)

        for fv, fp, fc, cl in zip(fvar, fpos, fcmap, clevs):
            ax1 = F.fig.add_subplot(gs[0:7, fp])
            if fp == 0:
                ax1.spines["bottom"].set_visible(False)
                ax1.spines["left"].set_visible(False)
                ax1.tick_params(labelbottom=True, bottom=True)

            else:
                ax1.axis("off")

            plt.plot(G1["lons"], G1["lats"], ".r", markersize=5)
            draw_range(lon_range, lat_range_prior, c="red", linewidth=1, zorder=12)
            draw_range(lon_range, lat_range, c="blue", linewidth=0.7, zorder=10)

            if fv != "ice":
                cm = plt.pcolor(lon, lat, G_beam[fv], vmin=cl[0], vmax=cl[-1], cmap=fc)
                if G_beam.ice.shape[0] > 1:
                    plt.contour(lon, lat, G_beam.ice, colors="black", linewidths=0.6)
            else:
                cm = plt.pcolor(lon, lat, G_beam[fv], vmin=cl[0], vmax=cl[-1], cmap=fc)

            plt.title(G_beam[fv].long_name.replace(" ", "\n") + "\n" + fv, loc="left")
            ax1.axis("equal")

            ax2 = F.fig.add_subplot(gs[-1, fp])
            cbar = plt.colorbar(
                cm, cax=ax2, orientation="horizontal", aspect=1, fraction=1
            )
            cl_ticks = np.linspace(cl[0], cl[-1], 3)

            cbar.set_ticks(np.round(cl_ticks, 3))
            cbar.set_ticklabels(np.round(cl_ticks, 2))

        F.save_pup(path=plot_path, name=plot_name + "_hindcast_data")

        ice_mask_prior = ice_mask.sel(latitude=G_prior.latitude)
        G_prior_masked = G_prior.where(~ice_mask_prior, np.nan)

        def test_nan_frac(imask):
            "test if False is less then 0.3"
            return ((~imask).sum() / imask.size).data < 0.3

        while test_nan_frac(ice_mask_prior):
            _logger.debug("lat range prior: %s", lat_range_prior)
            lat_range_prior = lat_range_prior[0] + 0.5, lat_range_prior[1] + 0.5
            G_prior = sel_data(G_beam, lon_range, lat_range_prior)
            ice_mask_prior = ice_mask.sel(latitude=G_prior.latitude)

        G_prior_masked = G_prior.where(~ice_mask_prior, np.nan)

        ### make pandas table with obs track end positions

        key_list = list(G_prior_masked.keys())
        # define directional and amplitude pairs
        # pack as  (amp, angle)
        key_list_pairs = {
            "mean": ("hs", "dir"),
            "peak": ("hs", "dp"),
            "partion0": ("phs0", "pdp0"),
            "partion1": ("phs1", "pdp1"),
            "partion2": ("phs2", "pdp2"),
            "partion3": ("phs3", "pdp3"),
            "partion4": ("phs4", "pdp4"),
        }

        # flatten key_list_pairs.values to a list
        key_list_pairs2 = [item for pair in key_list_pairs.values() for item in pair]

        key_list_scaler = set(key_list) - set(key_list_pairs2)

        ### derive angle average
        Tend = pd.DataFrame(index=key_list, columns=["mean", "std", "name"])

        for k, pair in key_list_pairs.items():
            ave_amp, ave_deg, std_amp, std_deg = waves.get_ave_amp_angle(
                G_prior_masked[pair[0]].data, G_prior_masked[pair[1]].data
            )
            Tend.loc[pair[0]] = ave_amp, std_amp, G_prior_masked[pair[0]].long_name
            Tend.loc[pair[1]] = ave_deg, std_deg, G_prior_masked[pair[1]].long_name

        for k in key_list_scaler:
            Tend.loc[k] = (
                G_prior_masked[k].mean().data,
                G_prior_masked[k].std().data,
                G_prior_masked[k].long_name,
            )

        Tend = Tend.T
        Tend["lon"] = [
            ice_mask_prior.longitude.mean().data,
            ice_mask_prior.longitude.std().data,
            "lontigude",  # TODO: fix typo?
        ]
        Tend["lat"] = [
            ice_mask_prior.latitude[ice_mask_prior.sum("longitude") == 0].mean().data,
            ice_mask_prior.latitude[ice_mask_prior.sum("longitude") == 0].std().data,
            "latitude",
        ]
        Tend = Tend.T

        Prior = build_prior(Tend)
        target_name = "A02_" + track_name + "_hindcast_success"
        MT.save_pandas_table(
            {"priors_hindcast": Tend}, save_name, str(save_path)
        )  # TODO: refactor save_pandas_table to use Path objects
    except Exception:
        target_name = "A02_" + track_name + "_hindcast_fail"

    def plot_prior(Prior, axx):
        angle = Prior["incident_angle"][
            "value"
        ]  # incident direction in degrees from North clockwise (Meterological convention)
        # use
        angle_plot = -angle - 90
        axx.quiver(
            Prior["center_lon"]["value"],
            Prior["center_lat"]["value"],
            -np.cos(angle_plot * np.pi / 180),
            -np.sin(angle_plot * np.pi / 180),
            scale=4.5,
            zorder=12,
            width=0.1,
            headlength=4.5,
            minshaft=2,
            alpha=0.6,
            color="black",
        )
        axx.plot(
            Prior["center_lon"]["value"],
            Prior["center_lat"]["value"],
            ".",
            markersize=6,
            zorder=12,
            alpha=1,
            color="black",
        )
        tstring = (
            " "
            + str(np.round(Prior["peak_period"]["value"], 1))
            + "sec \n "
            + str(np.round(Prior["Hs"]["value"], 1))
            + "m\n "
            + str(np.round(angle, 1))
            + "deg"
        )
        plt.text(lon_range[1], Prior["center_lat"]["value"], tstring)

    try:
        # plot 2nd figure

        font_for_print()
        F = M.FigureAxisXY(2, 4.5, view_scale=0.9, container=False)

        ax1 = F.ax
        lon, lat = G_beam.longitude, G_beam.latitude
        ax1.spines["bottom"].set_visible(False)
        ax1.spines["left"].set_visible(False)
        ax1.tick_params(labelbottom=True, bottom=True)

        plot_prior(Prior, ax1)

        # TODO: refactor as comprehension -- it might be less readable and more lines?. CP
        str_list = list()
        for i in np.arange(0, 6):
            str_list.append(
                " "
                + str(np.round(Tend.loc["ptp" + str(i)]["mean"], 1))
                + "sec\n "
                + str(np.round(Tend.loc["phs" + str(i)]["mean"], 1))
                + "m "
                + str(np.round(Tend.loc["pdp" + str(i)]["mean"], 1))
                + "d"
            )

        plt.text(lon_range[1], lat_range[0], "\n ".join(str_list))

        for vv in zip(
            ["pdp0", "pdp1", "pdp2", "pdp3", "pdp4", "pdp5"],
            ["phs0", "phs1", "phs3", "phs4", "phs5"],
        ):
            angle_plot = -Tend.loc[vv[0]]["mean"] - 90
            vsize = (1 / Tend.loc[vv[1]]["mean"]) ** (1 / 2) * 5
            ax1.quiver(
                Prior["center_lon"]["value"],
                Prior["center_lat"]["value"],
                -np.cos(angle_plot * np.pi / 180),
                -np.sin(angle_plot * np.pi / 180),
                scale=vsize,
                zorder=5,
                width=0.1,
                headlength=4.5,
                minshaft=4,
                alpha=0.6,
                color="green",
            )

        plt.plot(G1["lons"], G1["lats"], ".r", markersize=5)
        draw_range(lon_range, lat_range_prior, c="red", linewidth=1, zorder=11)
        draw_range(lon_range, lat_range, c="blue", linewidth=0.7, zorder=10)

        fv = "ice"
        if fv != "ice":
            plt.pcolor(lon, lat, G_beam[fv].where(~ice_mask, np.nan), cmap=fc)
            plt.contour(lon, lat, G_beam.ice, colors="black", linewidths=0.6)
        else:
            plt.pcolor(lon, lat, G_beam[fv], cmap=fc)

        plt.title(
            "Prior\n"
            + file_name_base[0:-1].replace("_", " ")
            + "\n"
            + track_name_short
            + "\nIncident angle",
            loc="left",
        )
        ax1.axis("equal")

        F.save_pup(path=plot_path, name=plot_name + "_hindcast_prior")
    except Exception as e:
        _logger.debug("%s", e)
        _logger.warning("print 2nd figure failed")

    MT.json_save(
        target_name,
        save_path,
        str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M")),
    )

    _logger.info("done")


make_iowaga_threads_prior_app = makeapp(
    run_A02c_IOWAGA_thredds_prior, name="threads-prior"
)

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    make_iowaga_threads_prior_app()
