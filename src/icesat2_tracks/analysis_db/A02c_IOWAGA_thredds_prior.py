import sys
import datetime

import h5py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import xarray as xr
from siphon.catalog import TDSCatalog

from icesat2_tracks.config.IceSAT2_startup import mconfig
import icesat2_tracks.ICEsat2_SI_tools.io as io
import icesat2_tracks.ICEsat2_SI_tools.wave_tools as waves
import icesat2_tracks.local_modules.m_tools_ph3 as MT
import icesat2_tracks.local_modules.m_general_ph3 as M
from icesat2_tracks.config.IceSAT2_startup import color_schemes
from icesat2_tracks.config.IceSAT2_startup import font_for_print

color_schemes.colormaps2(21)

track_name, batch_key, ID_flag = io.init_from_input(sys.argv)

track_name_short = track_name[0:-16]

ID, track_names, hemis, batch = io.init_data(
    track_name, batch_key, ID_flag, mconfig["paths"]["work"]
)

hemis, batch = batch_key.split("_")
ATlevel = "ATL03"

save_path = mconfig["paths"]["work"] + batch_key + "/A02_prior/"
plot_path = (
    mconfig["paths"]["plot"] + "/" + hemis + "/" + batch_key + "/" + track_name + "/"
)
save_name = "A02_" + track_name
plot_name = "A02_" + track_name_short
MT.mkdirs_r(plot_path)
MT.mkdirs_r(save_path)
bad_track_path = mconfig["paths"]["work"] + "bad_tracks/" + batch_key + "/"

all_beams = mconfig["beams"]["all_beams"]
high_beams = mconfig["beams"]["high_beams"]
low_beams = mconfig["beams"]["low_beams"]

load_path_WAVE_GLO = mconfig["paths"]["work"] + "/GLOBMULTI_ERA5_GLOBCUR_01/"
file_name_base = "LOPS_WW3-GLOB-30M_"


load_path = mconfig["paths"]["work"] + batch_key + "/B01_regrid/"
Gd = h5py.File(load_path + "/" + track_name + "_B01_binned.h5", "r")


G1 = dict()
for b in all_beams:
    Gi = io.get_beam_hdf_store(Gd[b])
    G1[b] = Gi.iloc[abs(Gi["lats"]).argmin()]

Gd.close()
G1 = pd.DataFrame.from_dict(G1).T


if hemis == "SH":
    # DEFINE SEARCH REGION AND SIZE OF BOXES FOR AVERAGES
    dlon_deg = 1  # degree range aroud 1st point
    dlat_deg = 30, 5  # degree range aroud 1st point
    dlat_deg_prior = 2, 1  # degree range aroud 1st point

    dtime = 4  # in hours

    lon_range = G1["lons"].min() - dlon_deg, G1["lons"].max() + dlon_deg
    lat_range = np.sign(G1["lats"].min()) * 78, G1["lats"].max() + dlat_deg[1]
    lat_range_prior = (
        G1["lats"].min() - dlat_deg_prior[0],
        G1["lats"].max() + dlat_deg_prior[1],
    )

else:
    # DEFINE SEARCH REGION AND SIZE OF BOXES FOR AVERAGES
    dlon_deg = 2  # lon degree range aroud 1st point
    dlat_deg = 20, 20  # lat degree range aroud 1st point
    dlat_deg_prior = 2, 1  # degree range aroud 1st point

    dtime = 4  # in hours

    lon_range = G1["lons"].min() - dlon_deg, G1["lons"].max() + dlon_deg
    lat_range = G1["lats"].min() - dlat_deg[0], G1["lats"].max() + dlat_deg[1]
    lat_range_prior = (
        G1["lats"].min() - dlat_deg_prior[0],
        G1["lats"].max() + dlat_deg_prior[1],
    )


timestamp = pd.to_datetime(ID["pars"]["start"]["delta_time"], unit="s")
time_range = np.datetime64(timestamp) - np.timedelta64(dtime, "h"), np.datetime64(
    timestamp
) + np.timedelta64(dtime, "h")

## load WW3 data
# ECMWF hindcast
# data_url = 'https://tds3.ifremer.fr/thredds/IOWAGA-WW3-HINDCAST/IOWAGA-GLOBAL_ECMWF-WW3-HINDCAST_FULL_TIME_SERIE.xml'

# CFSR hindcast
# data_url = 'https://tds3.ifremer.fr/thredds/IOWAGA-WW3-HINDCAST/IOWAGA-GLOBAL_CFSR-WW3-HINDCAST_FULL_TIME_SERIE.xml'

# ECMWF forecast
data_url = "https://tds3.ifremer.fr/thredds/IOWAGA-WW3-FORECAST/IOWAGA-WW3-FORECAST_GLOBMULTI_GLOB-30M.xml"

cat = TDSCatalog(data_url)

ncss = cat.datasets[
    "IOWAGA-WW3-FORECAST_GLOBMULTI_GLOB-30M_FIELD_NC_MARC_WW3-GLOB-30M"
].remote_access(use_xarray=True)

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


# chunk data
IOWAGA = ncss[var_list]
IOWAGA["time"] = np.array([np.datetime64(k0) for k0 in IOWAGA.time.data]).astype(
    "M8[h]"
)
IOWAGA = IOWAGA.rename(
    name_dict={
        "pdir0": "pdp0",
        "pdir1": "pdp1",
        "pdir2": "pdp2",
        "pdir3": "pdp3",
        "pdir4": "pdp4",
        "pdir5": "pdp5",
    }
)


def sel_data(I, lon_range, lat_range, timestamp=None):
    """
    this method returns the selected data in the lon-lat box at an interpolated timestamp
    """
    lon_flag = (lon_range[0] < I.longitude.data) & (I.longitude.data < lon_range[1])
    lat_flag = (lat_range[0] < I.latitude.data) & (I.latitude.data < lat_range[1])
    time_flag = (time_range[0] < I.time.data) & (I.time.data < time_range[1])
    if timestamp is None:
        I = I.isel(latitude=lat_flag, longitude=lon_flag)
    else:
        I = (
            I.isel(latitude=lat_flag, longitude=lon_flag, time=time_flag)
            .sortby("time")
            .interp(time=np.datetime64(timestamp))
        )
    return I


try:
    G_beam = sel_data(IOWAGA, lon_range, lat_range, timestamp).load()
    G_prior = sel_data(G_beam, lon_range, lat_range_prior)

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

        # find closed latituyde with with non-nan data
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
    def draw_range(lon_range, lat_range, *args, **kargs):
        plt.plot(
            [lon_range[0], lon_range[1], lon_range[1], lon_range[0], lon_range[0]],
            [lat_range[0], lat_range[0], lat_range[1], lat_range[1], lat_range[0]],
            *args,
            **kargs,
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

    F = M.figure_axis_xy(4, 3.5, view_scale=0.9, container=True)
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
        cbar = plt.colorbar(cm, cax=ax2, orientation="horizontal", aspect=1, fraction=1)
        cl_ticks = np.linspace(cl[0], cl[-1], 3)

        cbar.set_ticks(np.round(cl_ticks, 3))
        cbar.set_ticklabels(np.round(cl_ticks, 2))

    F.save_pup(path=plot_path, name=plot_name + "_hindcast_data")

    G_beam_masked = G_beam.where(~ice_mask, np.nan)
    ice_mask_prior = ice_mask.sel(latitude=G_prior.latitude)
    G_prior_masked = G_prior.where(~ice_mask_prior, np.nan)

    def test_nan_frac(imask):
        "test if False is less then 0.3"
        return ((~imask).sum() / imask.size).data < 0.3

    while test_nan_frac(ice_mask_prior):
        print(lat_range_prior)
        lat_range_prior = lat_range_prior[0] + 0.5, lat_range_prior[1] + 0.5
        G_prior = sel_data(G_beam, lon_range, lat_range_prior)
        ice_mask_prior = ice_mask.sel(latitude=G_prior.latitude)

    G_prior_masked = G_prior.where(~ice_mask_prior, np.nan)

    ### make pandas table with obs track end postitions

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

    key_list_pairs2 = list()
    for k in key_list_pairs.values():
        key_list_pairs2.append(k[0])
        key_list_pairs2.append(k[1])

    key_list_scaler = set(key_list) - set(key_list_pairs2)

    ### derive angle avearge
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
        "lontigude",
    ]
    Tend["lat"] = [
        ice_mask_prior.latitude[ice_mask_prior.sum("longitude") == 0].mean().data,
        ice_mask_prior.latitude[ice_mask_prior.sum("longitude") == 0].std().data,
        "latitude",
    ]
    Tend = Tend.T

    Prior = dict()
    Prior["incident_angle"] = {
        "value": Tend["mean"]["dp"].astype("float"),
        "name": Tend["name"]["dp"],
    }
    Prior["spread"] = {
        "value": Tend["mean"]["spr"].astype("float"),
        "name": Tend["name"]["spr"],
    }
    Prior["Hs"] = {
        "value": Tend["mean"]["hs"].astype("float"),
        "name": Tend["name"]["hs"],
    }
    Prior["peak_period"] = {
        "value": 1 / Tend["mean"]["fp"].astype("float"),
        "name": "1/" + Tend["name"]["fp"],
    }

    Prior["center_lon"] = {
        "value": Tend["mean"]["lon"].astype("float"),
        "name": Tend["name"]["lon"],
    }
    Prior["center_lat"] = {
        "value": Tend["mean"]["lat"].astype("float"),
        "name": Tend["name"]["lat"],
    }

    target_name = "A02_" + track_name + "_hindcast_success"

    MT.save_pandas_table({"priors_hindcast": Tend}, save_name, save_path)
except:
    target_name = "A02_" + track_name + "_hindcast_fail"

def plot_prior(Prior, axx):
    angle = Prior["incident_angle"][
        "value"
    ]  # incident direction in degrees from North clockwise (Meerological convention)
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
    F = M.figure_axis_xy(2, 4.5, view_scale=0.9, container=False)

    ax1 = F.ax
    lon, lat = G_beam.longitude, G_beam.latitude
    ax1.spines["bottom"].set_visible(False)
    ax1.spines["left"].set_visible(False)
    ax1.tick_params(labelbottom=True, bottom=True)

    plot_prior(Prior, ax1)

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
        print(vsize)
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
except  Exception as e:
    print(e)
    print("print 2nd figure failed")

MT.json_save(
    target_name, save_path, str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))
)

print("done")
