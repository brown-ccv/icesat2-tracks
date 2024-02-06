import os, sys

# execfile(os.environ['PYTHONSTARTUP'])

"""
This file open a ICEsat2 track applied filters and corections and returns smoothed photon heights on a regular grid in an .nc file.
This is python 3
"""

exec(open(os.environ["PYTHONSTARTUP"]).read())
exec(open(STARTUP_2021_IceSAT2).read())

# %matplotlib inline

import ICEsat2_SI_tools.convert_GPS_time as cGPS
import h5py
import ICEsat2_SI_tools.io as io
import ICEsat2_SI_tools.spectral_estimates as spec

import imp
import copy
import spicke_remover
import datetime
import concurrent.futures as futures

# import s3fs

col.colormaps2(21)

# %%
track_name, batch_key, test_flag = io.init_from_input(
    sys.argv
)  # loads standard experiment
# track_name, batch_key, test_flag = '20190605061807_10380310_004_01', 'SH_batch01', False
# track_name, batch_key, test_flag = '20190601094826_09790312_004_01', 'SH_batch01', False
# track_name, batch_key, test_flag = '20190207111114_06260210_004_01', 'SH_batch02', False
# track_name, batch_key, test_flag = '20190219073735_08070210_004_01', 'SH_batch02', False


# print(track_name, batch_key, test_flag)
hemis, batch = batch_key.split("_")
# track_name= '20190605061807_10380310_004_01'
ATlevel = "ATL03"

save_path = mconfig["paths"]["work"] + "/B03_spectra_" + hemis + "/"
save_name = "B03_" + track_name

plot_path = (
    mconfig["paths"]["plot"]
    + "/"
    + hemis
    + "/"
    + batch_key
    + "/"
    + track_name
    + "/A02a_sentenial_prior/"
)
MT.mkdirs_r(plot_path)
MT.mkdirs_r(save_path)
bad_track_path = mconfig["paths"]["work"] + "bad_tracks/" + batch_key + "/"
# %%

all_beams = mconfig["beams"]["all_beams"]
high_beams = mconfig["beams"]["high_beams"]
low_beams = mconfig["beams"]["low_beams"]
# Gfilt   = io.load_pandas_table_dict(track_name + '_B01_regridded', load_path) # rhis is the rar photon data

load_path_WAVE_GLO = (
    mconfig["paths"]["work"]
    + "/CMEMS_WAVE_GLO_L3/WAVE_GLO_WAV_L3_SPC_REP_OBSERVATIONS/"
)
file_name_base = "dataset-wav-sar-l3-spc-rep-global-"


load_path = mconfig["paths"]["work"] + "/B01_regrid_" + hemis + "/"
Gd = io.load_pandas_table_dict(track_name + "_B01_binned", load_path)  #
# %%
G1 = dict()
for b in all_beams:
    # find 1st point
    G1[b] = Gd[b].iloc[abs(Gd[b]["lats"]).argmin()]

G1 = pd.DataFrame.from_dict(G1).T

dlon_deg = 2.5  # degree range aroud 1st point
dlat_deg = 10  # degree range aroud 1st point
dtime = 24  # in hours

lon_range = G1["lons"].min() - dlon_deg, G1["lons"].max() + dlon_deg
lat_range = G1["lats"].min() - dlat_deg, G1["lats"].max() + dlat_deg
timestamp = pd.to_datetime(
    G1[["year", "month", "day", "hour", "minute", "second"]]
).mean()
time_range = np.datetime64(timestamp) - np.timedelta64(dtime, "h"), np.datetime64(
    timestamp
) + np.timedelta64(dtime, "h")
print(time_range)

# create timestamp according to fiels on ftp server:
time_stamps_search = np.arange(
    time_range[0].astype("datetime64[3h]") - np.timedelta64(12 * 24, "h"),
    time_range[1].astype("datetime64[3h]") + np.timedelta64(3, "h"),
    np.timedelta64(3, "h"),
)
time_stamps_search_str = [str(t).replace("-", "") for t in time_stamps_search]

del G1

# %%
import glob

if time_range[0].astype("M8[M]") != time_range[1].astype("M8[M]"):  # spanning two years
    MM_str = str(time_range[0].astype("M8[M]")).replace("-", "")
    f_list = glob.glob(load_path_WAVE_GLO + "/*" + MM_str + "*.nc")

    MM_str = str(time_range[-1].astype("M8[M]")).replace("-", "")
    f_list = glob.glob(load_path_WAVE_GLO + "/*" + MM_str + "*.nc")
    f_list = f_list + f_list
else:
    MM_str = str(time_range[0].astype("M8[M]")).replace("-", "")
    f_list = glob.glob(load_path_WAVE_GLO + "/*" + MM_str + "*.nc")

# %%

f_list.sort()

files_to_use = list()
for f in f_list:
    for ts_stamp in time_stamps_search_str:
        fn = f.split("/")[-1]
        if file_name_base + "s1a_" + ts_stamp in fn:
            files_to_use.append(["s1a_" + ts_stamp, f.split("/")[-1], f])
        elif file_name_base + "s1b_" + ts_stamp in fn:
            files_to_use.append(["s1b_" + ts_stamp, f.split("/")[-1], f])


def draw_range(lon_range, lat_range, *args, **kargs):
    plt.plot(
        [lon_range[0], lon_range[1], lon_range[1], lon_range[0], lon_range[0]],
        [lat_range[0], lat_range[0], lat_range[1], lat_range[1], lat_range[0]],
        *args,
        **kargs,
    )


def check_in_mask(I, lon_range, lat_range, time_range):
    lon_flag = (lon_range[0] < I.longitude.data) & (I.longitude.data < lon_range[1])
    lat_flag = (lat_range[0] < I.latitude.data) & (I.latitude.data < lat_range[1])
    time_flag = (time_range[0] < I.time.data) & (I.time.data < time_range[1])

    return lon_flag & lat_flag & time_flag


# %%
# Go through all files and find tracks that fit goeloation and time


# ID, file_name, path = files_to_use[9]
# #for ID, file_name, path in files_to_use[0:50]:
# FWi = xr.open_mfdataset(path)
# FWi['VPED']
def get_possible_tracks_from_file(ff):

    ID, file_name, path = ff
    D = dict()

    FWi = xr.open_mfdataset(path)
    # print(FWi.load().time.min().data, FWi.load().time.max().data )

    # M.figure_axis_xy(9, 2.5, view_scale= 0.5)
    # plt.suptitle(ID)
    # ax1 = plt.subplot(1, 2, 1)
    # plt.plot(FWi['longitude'], FWi['latitude'], '-')
    # draw_range(lon_range, lat_range, c='black')
    #
    # ax2 = plt.subplot(1, 2, 2)
    # draw_range(time_range, [FWi.obs[0], FWi.obs[-1]], c='black')
    # print(ID)
    for ob in FWi.obs:
        F3 = FWi.sel(obs=ob)
        # ax2.plot(F3['time'][~np.isnan(F3['longitude'])], F3['latitude'][~np.isnan(F3['longitude'])]*0 +ob )

        nmask = ~np.isnan(F3.latitude)
        F3 = F3.isel(propag_time=nmask)

        # check last point
        if F3.propag_time.size > 0:
            if check_in_mask(F3.isel(propag_time=-1), lon_range, lat_range, time_range):
                print(ID, " match")
                D[ID + "_obs" + str(ob.data)] = F3.load()

    return D


# futures.ProcessPoolExecutor
len(files_to_use)
with futures.ProcessPoolExecutor(max_workers=7) as executor_sub:
    D_nested = list(executor_sub.map(get_possible_tracks_from_file, files_to_use))

D = {}
for d in D_nested:
    D.update(d)

# D[next(iter(D.keys()))].load()['VPED']
# D[next(iter(D.keys()))].load()['VTPK']
# D[next(iter(D.keys()))].load()['VAVH']

# Di = D[next(iter(D.keys()))].load()

# %%

# make pandas table with obs track end postitions
key_list = next(iter(D.values())).isel(propag_time=-1).keys()
Tend = pd.DataFrame(index=key_list)


def format_to_list(II):
    k, I = II[0], II[1]
    dlist = list()
    for kk in key_list:
        dlist.append(I.isel(propag_time=-1)[kk].data)

    return k, dlist


for i in list(map(format_to_list, D.items())):
    Tend[i[0]] = i[1]

# Tend['s1a_20190209T09_obs105']['latitude']

timetemp = Tend.T["time"].apply(np.datetime64)
Tend = Tend.T.astype("float")
Tend["time"] = timetemp


# %%

weight = (
    np.array(1e2 / abs(Tend["time"] - np.datetime64(timestamp)).astype("m8[h]")) ** 2
)
weight = weight - weight.min()
weight = weight / weight.max()


def weighted_mean_and_var(a):
    am = (a * weight).sum() / weight.sum()
    av = ((a - am) ** 2 * weight).sum() / weight.sum()
    return am, av


Tfinal = Tend[["VAVH", "VPED", "latitude", "VTPK", "longitude"]]

# rescale variables accorinding to documenation:
Tfinal["VAVH"][Tfinal["VAVH"] == -999] = np.nan
Tfinal["VAVH"] = Tfinal["VAVH"] * 0.001
Tfinal["VTPK"] = Tfinal["VTPK"] * 0.01
Tfinal["VPED"] = Tfinal["VPED"] * 0.1

Tfinal = Tfinal.rename(
    columns={"VPED": "incident_angle", "VTPK": "peak_period", "VAVH": "Hs"}
)
wmeans = Tfinal.apply(weighted_mean_and_var, axis=0)
wmeans.index = ["w_mean", "w_var"]
print(wmeans)
print(np.sqrt(wmeans.T["w_var"]))

# %%

import itertools

col_iter = itertools.cycle(col.circle_big(np.arange(0, 21)))

GridSpec
font_for_print()
F = M.figure_axis_xy(7.5, 3, view_scale=0.7, container=True)
plt.suptitle(track_name)

gs = GridSpec(1, 3, wspace=0.2, hspace=0.7)  # figure=fig,
ax1 = F.fig.add_subplot(gs[0, 0])
plt.title("Geolocation displacement", loc="left")

draw_range(lon_range, lat_range, c="black", linewidth=2)
draw_range(
    [lon_range[0] + dlon_deg, lon_range[1] - dlon_deg],
    [lat_range[0] + dlat_deg, lat_range[1] - dlat_deg],
    c="red",
    linewidth=4,
)

ax2 = F.fig.add_subplot(gs[0, 1])
plt.title("time displacement", loc="left")

kkk = 0

for k, I in D.items():
    cc = next(col_iter)
    ax1.plot(I["longitude"], I["latitude"], "-", color=cc, linewidth=0.7)

    angle = I.VPED[~np.isnan(I.latitude)] / 10
    pos = I["longitude"], I["latitude"]
    ax1.quiver(
        pos[0][::5],
        pos[1][::5],
        -np.sin(angle * np.pi / 180)[::5],
        -np.cos(angle * np.pi / 180)[::5],
        scale=20,
        zorder=12,
        alpha=0.6,
        color="black",
    )

    ax2.plot(I["time"], I["latitude"] * 0 + kkk, color=cc)
    kkk += 1

draw_range(time_range, [-1, kkk + 1], c="black", linewidth=2)

plt.gcf().autofmt_xdate()
ax2.axvline(timestamp, color="red", linewidth=4)


ax3 = F.fig.add_subplot(gs[0, 2])
plt.title("nearest points and angles", loc="left")
# plt.plot( Tend['longitude'] , Tend['latitude'] , '.')

angle = Tfinal["incident_angle"]
pos = np.array(Tfinal["longitude"]), np.array(Tfinal["latitude"])
plt.quiver(
    pos[0],
    pos[1],
    -np.sin(angle * np.pi / 180),
    -np.cos(angle * np.pi / 180),
    scale=10,
    zorder=12,
    alpha=weight,
)
plt.scatter(Tend["longitude"], Tend["latitude"], color="black", s=6, alpha=weight)

draw_range(
    [lon_range[0] + dlon_deg, lon_range[1] - dlon_deg],
    [lat_range[0] + dlat_deg, lat_range[1] - dlat_deg],
    c="red",
    linewidth=2,
)
F.save_pup(path=plot_path, name=track_name)
