import os, sys

# execfile(os.environ['PYTHONSTARTUP'])

"""
This file open a ICEsat2 track applied filters and corections and returns smoothed photon heights on a regular grid in an .nc file.
This is python 3
"""
# exec(open(os.environ['PYTHONSTARTUP']).read())
# exec(open(STARTUP_2019_DP).read())

exec(open(os.environ["PYTHONSTARTUP"]).read())
exec(open(STARTUP_2021_IceSAT2).read())

import datetime
import h5py
from random import sample
import imp
import ICEsat2_SI_tools.convert_GPS_time as cGPS
import ICEsat2_SI_tools.io as io
from spectral_estimates import (
    create_chunk_boundaries_unit_lengths,
    create_chunk_boundaries,
)
import spectral_estimates as spec
import m_tools_ph3 as MT
import filter_regrid as regrid

from numba import jit

import concurrent.futures as futures

# memory test
# from guppy import hpy


def get_size(x):
    from pympler import asizeof

    ss = asizeof.asizeof(x) / 1e6
    return str(ss)


# get_size(hemis)

# import s3fs
# processed_ATL03_20190605061807_10380310_004_01.h5

# imp.reload(io)
ID_name, batch_key, ID_flag = io.init_from_input(sys.argv)  # loads standard experiment
# ID_name, batch_key, ID_flag = '20190605061807_10380310_004_01', 'SH_batch01', False
# ID_name, batch_key, ID_flag = '20190207234532_06340210_004_01', 'SH_batch02', False
# ID_name, batch_key, ID_flag = '20190215184558_07530210_004_01', 'SH_batch02', False
# ID_name, batch_key, ID_flag = '20190219073735_08070210_004_01', 'SH_batch02', False

# ID_name, batch_key, ID_flag = '20190224023410_08800212_004_01', 'SH_batch02', False
# ID_name, batch_key, ID_flag = '20190101020504_00550212_005_01', 'SH_batch04', False

# NH
# ID_name, batch_key, ID_flag = 'NH_20190301_09560205', 'NH_batch05', True # poleward false
# ID_name, batch_key, ID_flag = 'NH_20190301_09570203', 'NH_batch05', True # poleward false
# ID_name, batch_key, test_flag = 'NH_20190301_09580203', 'NH_batch05', True


# SH
# ID_name, batch_key, ID_flag = 'SH_20190101_00550210', 'SH_batch04', True
# ID_name, batch_key, ID_flag = 'SH_20190101_00570212', 'SH_batch04', True

# equatorward track
# ID_name, batch_key, ID_flag = '20190208154150_06440212_004_01', 'SH_batch02', False

# poleward track
# ID_name, batch_key, ID_flag = '20190209150245_06590210_004_01', 'SH_batch02', False
# conner

# for plotting
# rack_name, batch_key, ID_flag = '20190219073735_08070210_004_01', 'SH_batch02', False

ID, _, hemis, batch = io.init_data(
    ID_name,
    batch_key,
    ID_flag,
    mconfig["paths"]["work"],
)

# ID_name= '20190605061807_10380310_004_01'
ATlevel = "ATL03"

load_path = mconfig["paths"]["scratch"] + "/" + batch_key + "/"
load_file = "A01c_" + ATlevel + "_" + ID_name
load_file_str = load_path + load_file + ".h5"

save_path = mconfig["paths"]["work"] + "/" + batch_key + "/B01_regrid/"

plot_path = (
    mconfig["paths"]["plot"] + "/" + hemis + "/" + batch_key + "/" + ID_name + "/B01/"
)
bad_track_path = mconfig["paths"]["work"] + "bad_tracks/" + batch_key + "/"
MT.mkdirs_r(save_path)

# set pars

# define parameters:
Lmeter = 20  # stencil length in units of 'dist'; likely in meters the resulting resolution is L/2
Nphoton_min = 5  # mininum need photons per stancil to return results

Lmeter_large = (
    100e3  # stancil width for testing photon density. stancils do not overlab.
)
minium_photon_density = 0.02  # minimum photon density per meter in Lmeter_large chunk to be counted as real signal

plot_flag = True
Nworkers_process = 6  # number of threads for parallel processing  # outer loop
# %%
# test which beams exist:
all_beams = mconfig["beams"]["all_beams"]
high_beams = mconfig["beams"]["high_beams"]
# low_beams   = mconfig['beams']['low_beams']

if __name__ == "__main__":
    if ID_flag:
        track_poleward = ID["pars"]["poleward"]
        beams = all_beams
        print("take poleward from ID file")
    else:
        f = h5py.File(load_file_str, "r")
        beams = [b if b in f.keys() else None for b in all_beams]
        imp.reload(regrid)
        track_poleward = regrid.track_pole_ward_file(f)
        print("take poleward from ATL03 file")
    print("poleward track is ", track_poleward)


# %%
# open HDF5 file for all tracks
# ATL03       =   h5py.File(save_path_data + '/A01c_ATL03_'+ ID_name+'_2.h5', 'r')


def get_ATL03_beam(ATL03_k):

    DD = pd.DataFrame()  # columns = ATL03.keys())
    for ikey in ATL03_k.keys():
        DD[ikey] = ATL03_k[ikey]
    return DD


# for k in beams:
def load_data_and_cut(k):
    # print(k)
    # k =all_beams[1]

    Tsel = get_ATL03_beam(ATL03[k])
    seg = get_ATL03_beam(ATL03_seg[k])
    # Tsel_c = get_ATL03_beam(ATL03_c[k])

    # imp.reload(io)
    # T, seg = io.getATL03_beam(load_file_str, beam= k)

    print("loaded")
    # T = T[T['mask_seaice']] # only take sea ice points, no ocean points
    # T = T.drop(labels=[ 'year', 'month', 'day', 'hour', 'minute', 'second', 'ph_id_count', 'mask_seaice'], axis= 1)
    # T = T.drop(labels=[ 'ph_id_count', 'mask_seaice'], axis= 1)
    print("T MB " + get_size(Tsel))

    ho = k
    # ho = MT.add_line_var(ho, 'size', str(T.shape[0]))
    # ho = MT.add_line_var(ho, 'by confidence levels:' + str(np.arange(0, 5)), [ (T['signal_confidence'] == i).sum() for i in np.arange(0, 5) ])

    # filter:
    # Tsel    = T[ (T['heights']<100) & (T['heights'] > -100) ]# & (T['delta_time']>5) & (T['delta_time']<24) ]

    # if len(Tsel) == 0:
    #     ho  = MT.add_line_var(ho, 'no photons found', '')
    #     #Tsel= T[(T['signal_confidence'] ==-1 ) & (T['heights']<100)  & (T['heights'] > -100) ]# & (T['delta_time']>5) & (T['delta_time']<24) ]

    # Tsel_c  = io.getATL03_height_correction(load_file_str)
    # Tsel_c  = Tsel_c[Tsel_c['dem_h'] < 1e5] # cute out weird references
    # # needs only dem_h and heihgts
    # Tsel = regrid.correct_heights(Tsel, Tsel_c).reset_index(drop=True)# correct height
    # print('height corrected')

    ### cut data at the rear that has too much variance
    # cut last segments of data until variance is similar
    rear_mask = np.array(Tsel.index) > -1  # True
    nsize0 = Tsel.shape[0]
    N_seg = 20
    cut_flag = True
    dd_old = -1
    dd0 = np.array(Tsel["heights_c"])
    print("inital length", nsize0)

    @jit(nopython=True, parallel=False)
    def adjust_length(var_list, rear_mask, cut_flag, track_poleward):

        var_list = var_list if track_poleward else var_list[::-1]

        if var_list[0:3].mean() * 10 < var_list[-1]:
            # print('cut last '+ str(100/N_seg) +'% of data')
            rear_mask[int(nsize * (N_seg - 1) / N_seg) :] = False
        else:
            cut_flag = False

        rear_mask = rear_mask if track_poleward else rear_mask[::-1]

        return rear_mask, cut_flag

    # @jit(nopython=True, parallel= True)
    def get_var(sti):
        return dd[sti[0] : sti[1]].var()

    while cut_flag:
        dd = dd0[rear_mask]
        nsize = dd.size
        print("new length", nsize)

        stencil_iter = create_chunk_boundaries(
            int(nsize / N_seg), nsize, ov=0, iter_flag=True
        )
        var_list = np.array(list(map(get_var, stencil_iter)))
        # print(k, var_list)
        rear_mask, cut_flag = adjust_length(
            var_list, rear_mask, cut_flag, track_poleward
        )

        if nsize == dd_old:
            print("--- lengthen segments")
            N_seg -= 1
            # cut_flag = False

        dd_old = nsize

    print("Tsel MB " + get_size(Tsel))

    return k, rear_mask, Tsel, seg, ho


# load_data_and_cut(all_beams[1])

# %%

if __name__ == "__main__":
    ATL03 = h5py.File(load_path + "/" + load_file + "_corrected.h5", "r")
    ATL03_seg = h5py.File(load_path + "/" + load_file + "_seg.h5", "r")

    hist = "Beam stats"
    B = dict()
    # B1save  = dict()
    SEG = dict()
    k = beams[0]

    # A = list()
    # for k in all_beams:
    #     A.append(load_data_and_cut(k))

    with futures.ProcessPoolExecutor(max_workers=Nworkers_process) as executor:
        A = list(executor.map(load_data_and_cut, all_beams))

    print("A MB " + get_size(A))
    ATL03.close()
    ATL03_seg.close()
    # ATL03_c.close()

    for I in A:  # collect returns from from mapping
        k, rear_mask, Tsel, seg, ho = I

        Tsel["process_mask"] = rear_mask
        # B1save[k]             = Tsel
        B[k] = Tsel[rear_mask].drop(columns="process_mask")
        SEG[k] = seg

        ho = MT.add_line_var(
            ho,
            "cutted ",
            str(np.round(100 * (rear_mask.size - rear_mask.sum()) / rear_mask.size, 0))
            + "% in the back of the track",
        )
        ho = MT.add_line_var(ho, "selected size", str(Tsel.shape[0]))
        # ho      = MT.add_line_var(ho, 'final size ', str(Tsel_c.shape[0]))
        print(ho)
        hist = MT.write_log(hist, ho)

    print("done with 1st loop")

    del Tsel
    del A

    # print( 'B_save MB '  + get_size(B1save) )
    print("B MB " + get_size(B))
    print("SEG MB " + get_size(SEG))

    # %% define x- coodindate

    # find earliest segment length that is used.
    # this is on the equatorward side for poleward track, or
    # on the poleward side for equatorward tracks.

    total_segment_dist_x_min = list()
    for k, I in SEG.items():
        total_segment_dist_x_min.append(I["segment_dist_x"].min())
    total_segment_dist_x_min = min(total_segment_dist_x_min)


# %%
def make_x_coorindate(k):
    """
    Returns the "true" along track coordindate but finding the correpsonding segment length
    also adds the segment_ID to the main table T
    """
    print(k, " make coodindate")
    T, seg = B[k], SEG[k]
    # make sure data is strictly ordered by delta_time
    T = T.sort_values("delta_time").reset_index(drop=True)
    seg = seg.sort_values("delta_time").reset_index(drop=True)
    # find positions where segmetn length is reset
    # shifts segment length postions in time
    delta_onehalf = seg["delta_time"].diff() / 2
    delta_onehalf.iloc[0] = delta_onehalf.iloc[1]
    seg["delta_half_time"] = seg["delta_time"] - delta_onehalf - 1e-5

    # cur phontos that are not in segmentns
    T = T[
        (T["delta_time"] > seg["delta_half_time"].iloc[0])
        & (T["delta_time"] < seg["delta_half_time"].iloc[-1])
    ]
    bin_labels = np.digitize(T["delta_time"], seg["delta_half_time"], right=True)

    # select relevant data
    SS = seg["segment_dist_x"]
    SS_sid = seg["segment_id"]

    repeats = np.bincount(bin_labels, minlength=SS.shape[0])

    # check if repeats sumup
    if repeats.sum() != T.shape[0]:
        print("repeats do not sum up")

    # repeat  segment dat accoridng to photons
    SS = SS.repeat(repeats)
    SS.index = T.index
    SS_sid = SS_sid.repeat(repeats)
    SS_sid.index = T.index

    # define new coordinate
    T["x"] = SS + T["along_track_distance"]
    T["segment_id"] = SS_sid

    # find bad photons
    def find_anomalie_photons(Ti2, segi):
        x_interp = np.interp(
            Ti2["delta_time"], segi["delta_time"], segi["segment_dist_x"]
        )

        diff_x = x_interp - Ti2["x"]
        diff_x = abs(diff_x - diff_x.mean())
        return diff_x > 3 * diff_x.std(), x_interp

    fail_mask, x_interp = find_anomalie_photons(T, seg)

    print("weird photon fraction:", sum(fail_mask) / fail_mask.size)
    # aply fail mask
    # T= T[~fail_mask]
    T["x"][fail_mask] = x_interp[fail_mask]

    return k, T


if __name__ == "__main__":

    with futures.ProcessPoolExecutor(max_workers=Nworkers_process) as executor:
        A = list(executor.map(make_x_coorindate, all_beams))

    B = dict()
    for I in A:  # collect returns from from mapping
        k = I[0]
        B[k] = I[1][::-1]

        if not track_poleward:  # invert x- coordinate if there is an equatorward track
            print("invert table")
            B[k] = B[k].reset_index(drop=True)
            B[k]["x_true"] = B[k]["x"]
            B[k]["x"] = abs(B[k]["x"] - B[k]["x"].iloc[0])
        else:
            B[k]["x_true"] = B[k]["x"]
            print("no table invert needed")

    dist_list = np.array([np.nan, np.nan])
    for k in B.keys():
        dist_list = np.vstack([dist_list, [B[k]["x"].iloc[0], B[k]["x"].iloc[-1]]])

    print("B MB " + get_size(B))

    del A
    del SEG
    # del T

    # test ave photon density abnd quit if necessary
    track_dist_bounds = [np.nanmin(dist_list[:, 0], 0), np.nanmax(dist_list[:, 1], 0)]
    length_meter = abs(track_dist_bounds[1] - track_dist_bounds[0])
    # length_meter       = (abs(lat_lims[1])  - abs(lat_lims[0])) * 110e3
    p_densities_r = list()
    p_densities_l = list()

    for k, I in B.items():
        if "r" in k:
            p_densities_r.append(I.shape[0] / length_meter)
        else:
            p_densities_l.append(I.shape[0] / length_meter)

    if (np.array(p_densities_l).mean() < 0.5) & (
        np.array(p_densities_r).mean() < 0.5
    ):  # in photons per meter
        print(
            "photon density too low, this track is classified as bad track and pushed to bad list"
        )
        MT.json_save(
            ID_name,
            bad_track_path,
            {
                "densities": [
                    np.array(p_densities_r).mean(),
                    np.array(p_densities_l).mean(),
                ],
                "date": str(datetime.date.today()),
            },
        )
        print("exit.")
        exit()


# %% save corrected data and delete from cash
# io.save_pandas_table(B1save, ID_name + '_B01_corrected'  , save_path) # all photos but heights adjusted and with
# io.save_pandas_table(B, ID_name + '_B01_new_coords'  , save_path) # all photos but heights adjusted and with distance coordinate
# del B1save

# for testing
# T2 = B['gt1r']
# plt.plot( T2['delta_time'] , T2['along_track_distance'] , '.')
# plt.plot(T2['x'])
# for k,I in B.items():
#     plt.plot( I['x']  , I['across_track_distance'] - I[I['seg_ID_local'] == 0]['across_track_distance'].mean(), '.' , markersize = 0.3)
#     #plt.xlim(3e6, 3.25e6)
#
# for k,I in B.items():
#     plt.plot( I['x']  , I['across_track_distance'], '.' , markersize = 0.3)
#     #plt.xlim(3e6, 3.25e6)


# %%
if __name__ == "__main__":
    F = M.figure_axis_xy(4, 3, view_scale=0.7)

    for k, I in B.items():
        plt.plot(I["lats"], I["x"], ".", markersize=0.2)
        # plt.xlim(3e6, 3.25e6)
    plt.xlabel("lats")
    plt.ylabel("x")
    F.save_light(path=plot_path, name="B01_ALT03_" + ID_name + "_tracks_check_lat_x")

    # %
    F = M.figure_axis_xy(4, 3, view_scale=0.7)
    for k, I in B.items():
        plt.plot(I["delta_time"], I["lats"], ".", markersize=0.3)

    plt.xlabel("delta time")
    plt.ylabel("lat")
    F.save_light(path=plot_path, name="B01_ALT03_" + ID_name + "_tracks_check_time_lat")

    F = M.figure_axis_xy(4, 3, view_scale=0.7)
    for k, I in B.items():
        plt.plot(I["delta_time"], I["x"], ".", markersize=0.3)

    plt.xlabel("delta time")
    plt.ylabel("x")

    F.save_light(path=plot_path, name="B01_ALT03_" + ID_name + "_tracks_check_time_x")

# %%
##### 1.) derive common axis for beams and filter out low density area at the beginning
print("filter out low density area at the beginning")


def derive_axis_and_boundaries(key):
    # key = 'gt3r'
    print(key)
    # T2   #      = regrid.derive_axis(B[key], lat_lims).sort_values('dist')
    T2 = B[key]  # ['x']

    x0 = get_better_lower_boundary(Lmeter_large, np.array(T2["x"]))

    print(
        "cut off ",
        100 * (1 - T2[T2["x"] > x0].shape[0] / T2.shape[0]),
        "% off all data points at the beginning",
    )
    T2 = T2[T2["x"] > x0]  # cut off low density start

    return key, T2, [T2["x"].min(), T2["x"].max()]


def get_better_lower_boundary(Lmeter_large, dd):

    # T2         = regrid.derive_axis(B[key], lat_lims)
    # dd = np.array(B['gt1l']['x'])

    stencil_iter = spec.create_chunk_boundaries_unit_lengths(
        Lmeter_large, [dd.min(), dd.max()], ov=0, iter_flag=False
    )
    if stencil_iter.shape[1] == 0:
        while stencil_iter.shape[1] == 0:
            Lmeter_large = int(Lmeter_large / 2)
            print("new Lmeter_large" + str(Lmeter_large))
            stencil_iter = spec.create_chunk_boundaries_unit_lengths(
                Lmeter_large, [dd.min(), dd.max()], ov=0, iter_flag=False
            )

    stencil_iter = spec.create_chunk_boundaries_unit_lengths(
        Lmeter_large, [dd.min(), dd.max()], ov=0, iter_flag=True
    )

    def get_density(sti):
        return sti[0], sum((sti[0] <= dd) & (dd < sti[-1])) / Lmeter_large

    with futures.ThreadPoolExecutor(max_workers=Nworkers_process) as executor_sub:
        # var_list = np.array(list(map(get_density, stencil_iter)))
        var_list = np.array(list(executor_sub.map(get_density, stencil_iter)))

    var_list = np.array(var_list)
    # var_list[:,0] = np.random.rand(10)
    # sort var_list
    print(var_list)
    var_list = var_list[var_list[:, 0].argsort(), :]
    # print(var_list)
    if sum(var_list[:, 1] > minium_photon_density) > 1:
        first_stencil = next(
            (i for i, j in enumerate(var_list[:, 1] > minium_photon_density) if j), None
        )  # - 1
        stencil_iter = spec.create_chunk_boundaries_unit_lengths(
            Lmeter_large, [dd.min(), dd.max()], ov=0, iter_flag=False
        )
        return stencil_iter[0, first_stencil]

    else:
        # first_stencil = next((i for i, j in enumerate(var_list[:,1] > 0) if j), None)# - 1
        print("no sufficient photon density found. return short stencil")
        # first_stencil= len(var_list[:,1]) -1
        # stencil_iter  = spec.create_chunk_boundaries_unit_lengths( Lmeter_large, [ dd.min(), dd.max()],ov =0, iter_flag= False)
        # return stencil_iter[0, first_stencil]

        # #print(first_stencil)
        return var_list[-1, 0]  # [-1,0]


if __name__ == "__main__":

    with futures.ProcessPoolExecutor(max_workers=Nworkers_process) as executor:
        A = list(executor.map(derive_axis_and_boundaries, all_beams))

    # for k in all_beams:
    #     A = derive_axis_and_boundaries(k)

    B2 = dict()
    dist_list = np.array([np.nan, np.nan])
    for I in A:
        k = I[0]
        B2[k] = I[1]
        # B2[k]['dist'] = B2[k]['x']
        dist_list = np.vstack([dist_list, I[2]])

    del A
    del B
    track_dist_bounds = [np.nanmin(dist_list[:, 0], 0), np.nanmax(dist_list[:, 1], 0)]

    print("B2 MB " + get_size(B2))


# %%
if __name__ == "__main__":
    xscale = 1e3
    F = M.figure_axis_xy(5, 3, view_scale=0.6)
    for k, I in B2.items():
        plt.plot(
            I["x"] / xscale, I["across_track_distance"] / xscale, ".", markersize=0.3
        )
        # plt.xlim(3e6, 3.25e6)

    for k in high_beams:

        Ii = B2[k].iloc[0]
        plt.text(
            Ii.x / xscale + 5,
            Ii.across_track_distance / xscale,
            str(Ii[["lats", "lons"]]).split("Name")[0],
        )

        Ii = B2[k].iloc[-1]
        plt.text(
            Ii.x / xscale + 5,
            Ii.across_track_distance / xscale,
            str(Ii[["lats", "lons"]]).split("Name")[0],
            ha="right",
        )

    F.ax.axvline(track_dist_bounds[0] / xscale, color="gray", zorder=2)
    F.ax.axvline(track_dist_bounds[1] / xscale, color="gray", zorder=2)
    F.ax.axhline(0, color="gray", zorder=2)

    plt.title(
        "B01 filter and regrid | "
        + ID_name
        + "\npoleward "
        + str(track_poleward)
        + " \n \n",
        loc="left",
    )
    plt.xlabel("along track distance (km)")
    plt.ylabel("across track distance (km)")

    F.save_light(path=plot_path + "../", name="B01_ALT03_" + ID_name + "_tracks_all")

# %%
# for testing
# track_dist_bounds[1]  = track_dist_bounds[0] + (track_dist_bounds[1] - track_dist_bounds[0])/20
# track_dist_bounds = ts_s[0] ,ts_s[0] + (ts_s[1] - ts_s[0]) /6

# %%
##### 2.) regridding and averaging

print("regrid")


def regridding_wrapper(I):
    key, Ti = I
    print(key, Ti.shape, 2 * Ti.shape[0] / Lmeter)
    stencil_iter = create_chunk_boundaries_unit_lengths(
        Lmeter, track_dist_bounds, iter_flag=False
    )
    Bi = regrid.get_stencil_stats_shift(
        Ti,
        stencil_iter,
        "heights_c",
        "x",
        stancil_width=Lmeter / 2,
        Nphoton_min=Nphoton_min,
    )

    # print( 'Bi MB '  + get_size(Bi) )
    print(key, "done")
    return key, Bi


if __name__ == "__main__":

    # ---define start and end position and same in Json file
    with futures.ProcessPoolExecutor(max_workers=Nworkers_process) as executor:
        B3 = dict(executor.map(regridding_wrapper, B2.items()))

    print("B3 MB " + get_size(B3))
    print("I MB " + get_size(I))

    # I = B3['gt2l'].copy()
    D_info = dict()
    for k, I in B3.items():

        # reset x coordinate
        I["median_dist"] = I["x_median"] - track_dist_bounds[0]  # - Lmeter/2
        I["dist"] = I["x"] - track_dist_bounds[0]  # - Lmeter/2
        # I['index']      = I['x']
        # rename y coordinate
        I = I.rename(columns={"across_track_distance": "y"})

        # find starting and end position
        Di_s = dict(
            I[I["segment_id"] == I["segment_id"].iloc[0]].mean()[
                ["lons", "lats", "segment_id", "delta_time"]
            ]
        )
        Di_s["across_track_distance_0"] = track_dist_bounds[0]

        Di_e = dict(
            I[I["segment_id"] == I["segment_id"].iloc[-1]].mean()[
                ["lons", "lats", "segment_id", "delta_time"]
            ]
        )
        Di_e["across_track_distance_0"] = track_dist_bounds[0]

        D_info[k] = {"start": Di_s, "end": Di_e, "poleward": str(track_poleward)}

        # reorder indexes
        column_names = [
            "x",
            "y",
            "x_median",
            "median_dist",
            "lons",
            "lats",
            "heights_c_weighted_mean",
            "heights_c_median",
            "heights_c_std",
            "N_photos",
        ]
        vars_ad = set(
            list(I[I["segment_id"] == I["segment_id"].iloc[0]].mean().index)
        ) - set(column_names)
        I = I.reindex(columns=column_names + list(vars_ad))

        B3[k] = I

    # save Json
    MT.json_save(ID_name + "_B01_stats", save_path, D_info, verbose=True)

    # saving data
    # io.save_pandas_table(B2, ID_name + '_B01_regridded'  , save_path) # all photos but heights adjusted and with distance coordinate
    # io.save_pandas_table(B3, ID_name + '_B01_binned'     , save_path) # regridding heights

    io.write_track_to_HDF5(
        B2, ID_name + "_B01_regridded", save_path
    )  # all photos but heights adjusted and with distance coordinate
    io.write_track_to_HDF5(B3, ID_name + "_B01_binned", save_path)  # regridding heights


# %% plotting just for checking
if __name__ == "__main__":

    key = "gt2r"

    if plot_flag:
        MT.mkdirs_r(plot_path)
        Ti2 = B3[key]
        T2 = B2[key]

        dl = 25000
        x_key = "x"

        latlims = (Ti2["x"].iloc[0], Ti2["x"].iloc[-1])
        # chunk_list = np.arange(latlims[0],latlims[1], dl )
        # chunk_list = sample(  list(np.arange(latlims[0],latlims[1],dl )[0:80])  ,10)
        chunk_list = np.arange(latlims[0], latlims[1], dl)[::10]
        chunk_list = np.append(chunk_list, latlims[1] - dl - 1)
        for ll in chunk_list:
            F = M.figure_axis_xy(7, 3, view_scale=0.8)

            plt.plot(T2[x_key], T2["heights_c"], "k.", markersize=0.5, alpha=0.8)
            # plt.plot( ALT07['ref']['latitude'] , ALT07['heights']['height_segment_height'] , 'r.', markersize=0.8, alpha = 1, label ='ALT07 seg. heights')

            plt.plot(
                Ti2[x_key],
                Ti2["heights_c_weighted_mean"] - 0.5,
                ".-",
                color="blue",
                linewidth=0.5,
                markersize=2,
                alpha=0.9,
                label="x-gauss weighted mean +1",
            )
            plt.plot(
                Ti2[x_key],
                Ti2["heights_c_median"] + 0.5,
                "r.-",
                linewidth=0.5,
                markersize=2,
                alpha=0.9,
                label="median",
            )

            # plt.plot(Ti2['x'], Ti2['heights_c_mode']-1, 'g.-',  linewidth=0.5, markersize=2,alpha=0.9, label='mode - 1')

            plt.plot(Ti2["x"], Ti2["heights_c_std"] - 1.8, "k-", linewidth=0.5, alpha=1)
            plt.fill_between(
                Ti2["x"], Ti2["heights_c_std"] - 1.8, y2=-1.8, color="gray", alpha=1
            )

            # plt.plot( ALT03['delta_time'], ALT03['heights_c'],   'k.',  markersize= 0.3, alpha =0.2 )
            # plt.plot(ALT07['time']['delta_time'] , ALT07['heights']['height_segment_height'] , 'r.', markersize=0.8, alpha = 1, label ='ALT07 seg. heights')
            plt.legend(loc=1)
            plt.xlim(ll, ll + dl)
            plt.ylim(-4, 4)

            plt.xlabel("Meters from the Sea Ice Edge")
            plt.ylabel("Height Anomalie (meters)")
            F.ax.axhline(y=-1.8, color="black", linewidth=0.5)
            F.save_light(path=plot_path, name="ALT03_filt_compare" + str(ll))
