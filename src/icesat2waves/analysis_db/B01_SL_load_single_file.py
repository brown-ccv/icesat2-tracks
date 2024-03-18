#!/usr/bin/env python
"""
This file open a ICEsat2 tbeam_stats.pyrack applied filters and corrections and returns smoothed photon heights on a regular grid in an .nc file.
This is python 3.11
"""
import datetime
import copy
from pathlib import Path
import warnings
import logging

import xarray as xr
from sliderule import icesat2
import matplotlib
import typer
from pandas.errors import (
    SettingWithCopyWarning,
)  # TODO: remove when warnings are handled

from icesat2waves.config.startup import (
    mconfig,
    color_schemes,
    font_for_pres,
    plt,
)
import icesat2waves.tools.sliderule_converter_tools as sct
import icesat2waves.tools.iotools as io
import icesat2waves.tools.beam_stats as beam_stats
import icesat2waves.local_modules.m_tools_ph3 as MT
from icesat2waves.local_modules import m_general_ph3 as M

from icesat2waves.clitools import (
    validate_track_name,
    validate_batch_key,
    validate_output_dir,
    suppress_stdout,
    report_input_parameters,
    update_paths_mconfig,
    echo,
    echoparam,
    makeapp,
)

_logger = logging.getLogger(__name__)


def make_B01_dict(table_data, split_by_beam=True, to_hdf5=False):
    """
    converts a GeoDataFrame from Sliderule to GeoDataFrames for each beam with the correct columns and names
    inputs:
        table_data: GeoDataFrame with the data
        split_by_beam: True/False. If True the data is split by beam
    returns:
        if split_by_beam:
            table_data: dict of GeoDataFrame with the data for each beam
        else:
            table_data: GeoDataFrame with the data for all beams in one table
    """

    table_data.rename(
        columns={
            "n_fit_photons": "N_photos",
            "w_surface_window_final": "signal_confidence",
            "y_atc": "y",
            "x_atc": "distance",
        },
        inplace=True,
    )

    table_data["lons"] = table_data["geometry"].x
    table_data["lats"] = table_data["geometry"].y

    drop_columns = ["cycle", "gt", "rgt", "pflags"]
    if to_hdf5:
        drop_columns.append("geometry")
    table_data.drop(columns=drop_columns, inplace=True)

    if split_by_beam:
        B01b = dict()
        # this is not tested
        for spot, beam in zip(
            [1, 2, 3, 4, 5, 6], ["gt1l", "gt1r", "gt2l", "gt2r", "gt3l", "gt3r"]
        ):
            ii = table_data.spot == spot
            B01b[beam] = table_data[ii]
        return B01b
    else:
        return table_data


def run_B01_SL_load_single_file(
    track_name: str = typer.Option(..., callback=validate_track_name),
    batch_key: str = typer.Option(..., callback=validate_batch_key),
    ID_flag: bool = True,
    plot_flag: bool = True,
    output_dir: str = typer.Option(..., callback=validate_output_dir),
):
    """
    Open an ICEsat2 tbeam_stats.pyrack, apply filters and corrections, and output smoothed photon heights on a regular grid in an .nc file.
    """
    # report input parameters
    kwargs = {
        "track_name": track_name,
        "batch_key": batch_key,
        "ID_flag": ID_flag,
        "output_dir": output_dir,
    }
    report_input_parameters(**kwargs)

    xr.set_options(display_style="text")
    matplotlib.use("Agg")  # prevent plot windows from opening

    # Select region and retrieve batch of tracks

    track_name, batch_key, ID_flag = io.init_from_input(
        [
            None,
            track_name,
            batch_key,
            ID_flag,
        ]  # init_from_input expects sys.argv with 4 elements
    )  # loads standard experiment

    hemis = batch_key.split("_")[0]

    workdir, plotsdir = update_paths_mconfig(output_dir, mconfig)

    save_path = Path(workdir, batch_key, "B01_regrid")
    save_path.mkdir(parents=True, exist_ok=True)

    save_path_json = Path(workdir, batch_key, "A01b_ID")
    save_path_json.mkdir(parents=True, exist_ok=True)

    ATL03_track_name = f"ATL03_{track_name}.h5"

    # Configure SL Session
    icesat2.init("slideruleearth.io")

    # plot the ground tracks in geographic location
    # Generate ATL06-type segments using the ATL03-native photon classification
    # Use the ocean classification for photons with a confidence parameter to 2 or higher (low confidence or better)

    # YAPC alternative
    params_yapc = {
        "srt": 1,
        "len": 20,
        "ats": 3,
        "res": 10,
        "dist_in_seg": False,  # if False units of len and res are in meters
        "track": 0,
        "pass_invalid": False,
        "cnf": 2,
        "cnt": 20,
        "sigma_r_max": 4,  # maximum standard deviation of photon in extend
        "maxi": 10,
        "yapc": dict(
            knn=0, win_h=6, win_x=11, min_ph=4, score=100
        ),  # use the YAPC photon classifier; these are the recommended parameters, but the results might be more specific with a smaller win_h value, or a higher score cutoff
        #   "yapc": dict(knn=0, win_h=3, win_x=11, min_ph=4, score=50),  # use the YAPC photon classifier; these are the recommended parameters, but the results might be more specific with a smaller win_h value, or a higher score cutoff
        "atl03_geo_fields": ["dem_h"],
    }

    maximum_height = 30  # (meters) maximum height past dem_h correction
    _logger.info("Fetching ATL03 data from sliderule")
    gdf = icesat2.atl06p(params_yapc, resources=[ATL03_track_name])
    _logger.info("Finished fetching ATL03 data from sliderule")
    gdf = sct.correct_and_remove_height(gdf, maximum_height)

    cdict = dict()
    for s, b in zip(
        gdf["spot"].unique(), ["gt1l", "gt1r", "gt2l", "gt2r", "gt3l", "gt3r"]
    ):
        cdict[s] = color_schemes.rels[b]

    font_for_pres()
    F_atl06 = M.FigureAxisXY(6.5, 5, view_scale=0.6)
    F_atl06.fig.suptitle(track_name)

    beam_stats.plot_ATL06_track_data(gdf, cdict)

    # main routine for defining the x coordinate and sacing table data

    # define reference point and then define 'x'
    table_data = copy.copy(gdf)

    # the reference point is defined as the most equatorward point of the polygon.
    # It's distance from the equator is  subtracted from the distance of each photon.
    table_data = sct.define_x_coordinate_from_data(table_data)
    table_time = table_data["time"]
    table_data.drop(columns=["time"], inplace=True)

    # renames columns and splits beams
    Ti = make_B01_dict(table_data, split_by_beam=True, to_hdf5=True)

    with warnings.catch_warnings():
        warnings.simplefilter(
            "ignore", category=SettingWithCopyWarning
        )  # TODO: remove when warnings are handled
        for kk in Ti.keys():
            Ti[kk]["dist"] = Ti[kk]["x"].copy()
            Ti[kk]["heights_c_weighted_mean"] = Ti[kk]["h_mean"].copy()
            Ti[kk]["heights_c_std"] = Ti[kk]["h_sigma"].copy()

    segment = track_name.split("_")[1][-2:]
    ID_name = sct.create_ID_name(gdf.iloc[0], segment=segment)
    echoparam("ID_name", ID_name)
    io.write_track_to_HDF5(Ti, ID_name + "_B01_binned", save_path)  # regridding heights

    #  plot the ground tracks in geographic location

    all_beams = mconfig["beams"]["all_beams"]
    high_beams = mconfig["beams"]["high_beams"]
    low_beams = mconfig["beams"]["low_beams"]

    D = beam_stats.derive_beam_statistics(Ti, all_beams, Lmeter=12.5e3, dx=10)

    # save figure from above:
    plot_path = Path(plotsdir, hemis, batch_key, ID_name)
    plot_path.mkdir(parents=True, exist_ok=True)

    F_atl06.save_light(path=plot_path, name="B01b_ATL06_corrected.png")
    plt.close()

    if plot_flag:
        font_for_pres()
        F = M.FigureAxisXY(8, 4.3, view_scale=0.6)
        beam_stats.plot_beam_statistics(
            D,
            high_beams,
            low_beams,
            color_schemes.rels,
            track_name=track_name
            + "|  ascending ="
            + str(sct.ascending_test_distance(gdf)),
        )

        F.save_light(path=plot_path, name="B01b_beam_statistics.png")
        plt.close()

        # plot the ground tracks in geographic location
        gdf[::100].plot(markersize=0.1, figsize=(4, 6))
        plt.title(
            track_name + "\nascending =" + str(sct.ascending_test_distance(gdf)),
            loc="left",
        )
        M.save_anyfig(plt.gcf(), path=plot_path, name="B01_track.png")
        plt.close()

    _logger.info("write A01b .json")
    DD = {"case_ID": ID_name, "tracks": {}}

    DD["tracks"]["ATL03"] = f"ATL10-{track_name}"

    start_pos = abs(table_data.lats).argmin()
    end_pos = abs(table_data.lats).argmax()

    # add other pars:
    DD["pars"] = {
        "poleward": sct.ascending_test(gdf),
        "region": "0",
        "start": {
            "longitude": table_data.lons[start_pos],
            "latitude": table_data.lats[start_pos],
            "seg_dist_x": float(table_data.x[start_pos]),
            "delta_time": datetime.datetime.timestamp(table_time[start_pos]),
        },
        "end": {
            "longitude": table_data.lons[end_pos],
            "latitude": table_data.lats[end_pos],
            "seg_dist_x": float(table_data.x[end_pos]),
            "delta_time": datetime.datetime.timestamp(table_time[end_pos]),
        },
    }

    MT.json_save2(name="A01b_ID_" + ID_name, path=save_path_json, data=DD)

    _logger.info("done")


load_file_app = makeapp(run_B01_SL_load_single_file, name="load-file")

if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    load_file_app()
