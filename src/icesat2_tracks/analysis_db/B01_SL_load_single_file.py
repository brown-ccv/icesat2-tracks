"""
This file open a ICEsat2 tbeam_stats.pyrack applied filters and corections and returns smoothed photon heights on a regular grid in an .nc file.
This is python 3.11
"""
import sys
import datetime
import copy
from pathlib import Path

import xarray as xr
from sliderule import icesat2

from icesat2_tracks.config.IceSAT2_startup import (
    mconfig,
    color_schemes,
    font_for_pres,
    plt,
)
from icesat2_tracks.ICEsat2_SI_tools import (
    sliderule_converter_tools as sct,
    io,
    beam_stats,
)
from icesat2_tracks.local_modules import m_tools_ph3 as MT, m_general_ph3 as M

xr.set_options(display_style="text")

# Select region and retrive batch of tracks
track_name, batch_key, ID_flag = io.init_from_input(
    sys.argv
)  # loads standard experiment

plot_flag = True
hemis = batch_key.split("_")[0]

# Make target directories
basedir = Path(mconfig["paths"]["work"], batch_key)
save_path, save_path_json = Path(basedir, "B01_regrid"), Path(basedir, "A01b_ID")
for p in [save_path, save_path_json]:
    MT.mkdirs_r(p)

ATL03_track_name = "ATL03_" + track_name + ".h5"

# Configure SL Session
icesat2.init("slideruleearth.io")


# plot the ground tracks in geographic location
# Generate ATL06-type segments using the ATL03-native photon classification
# Use the ocean classification for photons with a confidence parmeter to 2 or higher (low confidence or better)

params = {
    "srt": 1,  # Ocean classification
    "len": 25,  # 10-meter segments
    "ats": 3,  # require that each segment contain photons separated by at least 5 m
    "res": 10,  # return one photon every 10 m
    "dist_in_seg": False,  # if False units of len and res are in meters
    "track": 0,  # return all ground tracks
    "pass_invalid": False,
    "cnt": 20,
    "sigma_r_max": 4,  # maximum standard deviation of photon in extend
    "cnf": 2,  # require classification confidence of 2 or more
    "atl03_geo_fields": ["dem_h"],
}


# YAPC alternatibe
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

gdf = io.get_gdf(ATL03_track_name, params_yapc, maximum_height)


cdict = dict()
for s, b in zip(gdf["spot"].unique(), ["gt1l", "gt1r", "gt2l", "gt2r", "gt3l", "gt3r"]):
    cdict[s] = color_schemes.rels[b]


font_for_pres()
F_atl06 = M.figure_axis_xy(6.5, 5, view_scale=0.6)
F_atl06.fig.suptitle(track_name)

beam_stats.plot_ATL06_track_data(gdf, cdict)


# main routine for defining the x coordinate and sacing table data
def make_B01_dict(table_data, split_by_beam=True, to_hdf5=False):
    """
    converts a GeoDataFrame from Sliderule to GeoDataFrames for each beam witht the correct columns and names
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


# define reference point and then define 'x'
table_data = copy.copy(gdf)

# the reference point is defined as the most equatorward point of the polygon.
# It's distance from the equator is  subtracted from the distance of each photon.
table_data = sct.define_x_coordinate_from_data(table_data)
table_time = table_data["time"]
table_data.drop(columns=["time"], inplace=True)

# renames columns and splits beams
Ti = make_B01_dict(table_data, split_by_beam=True, to_hdf5=True)

for kk in Ti.keys():
    Ti[kk]["dist"] = Ti[kk]["x"].copy()
    Ti[kk]["heights_c_weighted_mean"] = Ti[kk]["h_mean"].copy()
    Ti[kk]["heights_c_std"] = Ti[kk]["h_sigma"].copy()

segment = track_name.split("_")[1][-2:]
ID_name = sct.create_ID_name(gdf.iloc[0], segment=segment)
print(ID_name)
io.write_track_to_HDF5(Ti, ID_name + "_B01_binned", save_path)  # regridding heights

#  plot the ground tracks in geographic location

all_beams = mconfig["beams"]["all_beams"]
high_beams = mconfig["beams"]["high_beams"]
low_beams = mconfig["beams"]["low_beams"]

D = beam_stats.derive_beam_statistics(Ti, all_beams, Lmeter=12.5e3, dx=10)

# save figure from above:
plot_path = (
    mconfig["paths"]["plot"] + "/" + hemis + "/" + batch_key + "/" + ID_name + "/"
)
MT.mkdirs_r(plot_path)
F_atl06.save_light(path=plot_path, name="B01b_ATL06_corrected.png")
plt.close()

if plot_flag:
    font_for_pres()
    F = M.figure_axis_xy(8, 4.3, view_scale=0.6)
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
        track_name + "\nascending =" + str(sct.ascending_test_distance(gdf)), loc="left"
    )
    M.save_anyfig(plt.gcf(), path=plot_path, name="B01_track.png")
    plt.close()


print("write A01b .json")
DD = {"case_ID": ID_name, "tracks": {}}

DD["tracks"]["ATL03"] = "ATL10-" + track_name


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

print("done")
