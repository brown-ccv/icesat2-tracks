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

import concurrent.futures as futures


def get_size(x):
    from pympler import asizeof

    ss = asizeof.asizeof(x) / 1e6
    return str(ss)


all_beams = mconfig["beams"]["all_beams"]

# imp.reload(io)
ID_name, batch_key, ID_flag = io.init_from_input(sys.argv)  # loads standard experiment

# SH example
# ID_name, batch_key, ID_flag = 'SH_20190101_00550210', 'SH_batch04', True
# ID_name, batch_key, ID_flag = 'SH_20190102_00770210', 'SH_batch04', True

# NH example
# ID_name, batch_key, ID_flag = 'NH_20190301_09560203', 'NH_batch05', True # poleward false
# ID_name, batch_key, ID_flag = 'NH_20190301_09560205', 'NH_batch05', True # poleward false

# ID example
# ID_name, batch_key, ID_flag = '20190101015140_00550210_005_01', 'SH_batch04', False


# imp.reload(io)
ID, track_names, hemis, batch = io.init_data(
    ID_name, batch_key, ID_flag, mconfig["paths"]["work"]
)
ID, track_names, hemis, batch

# %%
load_path_data = mconfig["paths"]["scratch"] + "/" + batch_key + "/"
save_path_data = mconfig["paths"]["scratch"] + "/" + batch_key + "/"
bad_track_path = mconfig["paths"]["work"] + "bad_tracks/" + batch_key + "/"


# imp.reload(io)
def ATL03_loader(k):
    """
    returns tables with merged ATL03 data
    """
    print("beam " + k)
    # concat and resize if there are more then 2 tracks involved
    if len(track_names) > 1:
        print("tracklist found")
        T_lists, seg_list, Tsel_c_list = list(), list(), list()
        for ATLi in track_names:
            try:
                T0, seg0 = io.getATL03_beam(load_path_data + ATLi + ".h5", beam=k)
                T_lists.append(T0)
                seg_list.append(seg0)

                Tsel_c = io.getATL03_height_correction(
                    load_path_data + ATLi + ".h5", beam=k
                )
                Tsel_c_list.append(Tsel_c)
            except:
                print(ATLi, " not found!! continue with single file!")

        if len(T_lists) == 0:  # in photons per meter
            print(
                "no files found, this track is classified as bad track and pushed to bad list"
            )
            MT.json_save(
                ID_name,
                bad_track_path,
                {
                    "A01c": "no files found, this track is classified as bad track and pushed to bad list",
                    "track_names": track_names,
                    "date": str(datetime.date.today()),
                },
            )
            print("exit.")
            exit()

        Ti = pd.concat(T_lists)
        Si = pd.concat(seg_list)
        Ti_c = pd.concat(Tsel_c_list)

        if ID["pars"]["poleward"]:
            Ti = Ti[Ti["delta_time"] <= ID["pars"]["end"]["delta_time"]]
            Si = Si[Si["delta_time"] <= ID["pars"]["end"]["delta_time"]]
            Ti_c = Ti_c[Ti_c["delta_time"] <= ID["pars"]["end"]["delta_time"]]
        else:
            Ti = Ti[Ti["delta_time"] >= ID["pars"]["end"]["delta_time"]]
            Si = Si[Si["delta_time"] >= ID["pars"]["end"]["delta_time"]]
            Ti_c = Ti_c[Ti_c["delta_time"] >= ID["pars"]["end"]["delta_time"]]
    else:
        print("only 1 track found")
        T0, seg0 = io.getATL03_beam(load_path_data + track_names[0] + ".h5", beam=k)
        Tsel_c = io.getATL03_height_correction(
            load_path_data + track_names[0] + ".h5", beam=k
        )
        Ti = T0
        Si = seg0
        Ti_c = Tsel_c

    print("mask and correct with dem_h")
    Ti = Ti[Ti["mask_seaice"]]  # only take sea ice points, no ocean points
    # T = T.drop(labels=[ 'year', 'month', 'day', 'hour', 'minute', 'second', 'ph_id_count', 'mask_seaice'], axis= 1)
    Ti = Ti.drop(labels=["ph_id_count", "mask_seaice"], axis=1)

    # Ti = Ti[Ti['mask_seaice']] # only take sea ice points, no ocean points
    # T = T.drop(labels=[ 'year', 'month', 'day', 'hour', 'minute', 'second', 'ph_id_count', 'mask_seaice'], axis= 1)
    # Ti = Ti.drop(labels=[ 'ph_id_count'], axis= 1)

    print("T MB " + get_size(Ti))

    # filter:
    Tsel = Ti[
        (Ti["heights"] < 100) & (Ti["heights"] > -100)
    ]  # & (T['delta_time']>5) & (T['delta_time']<24) ]

    # if len(Tsel) == 0:
    #     ho  = MT.add_line_var(ho, 'no photons found', '')
    #     #Tsel= T[(T['signal_confidence'] ==-1 ) & (T['heights']<100)  & (T['heights'] > -100) ]# & (T['delta_time']>5) & (T['delta_time']<24) ]

    # Tsel_c  = io.getATL03_height_correction(load_file_str)
    Ti_c = Ti_c[Ti_c["dem_h"] < 1e5]  # cute out weird references
    # needs only dem_h and heihgts
    Tsel = regrid.correct_heights(Tsel, Ti_c).reset_index(drop=True)  # correct height
    print("height corrected")

    return k, Tsel, Si


# Nworkers_load = 2
# with futures.ProcessPoolExecutor(max_workers=Nworkers_load) as executor:
#     A = list( executor.map(ATL03_loader, all_beams)  )

A = list()
for bb in all_beams:
    A.append(ATL03_loader(bb))

# %% reorganize loaderd data
TT, SS, TCC = dict(), dict(), dict()
for Ai in A:
    k = Ai[0]
    TT[k] = Ai[1]
    SS[k] = Ai[2]
    # TCC[k] = Ai[3]


# Write data
# MT.save_pandas_table(TT, 'A01c_ATL03_'+ ID_name,        save_path_data)
# MT.save_pandas_table(SS, 'A01c_ATL03_'+ ID_name+'_seg', save_path_data)

imp.reload(io)
io.write_track_to_HDF5(TT, "A01c_ATL03_" + ID_name + "_corrected", save_path_data)
io.write_track_to_HDF5(SS, "A01c_ATL03_" + ID_name + "_seg", save_path_data)
# io.write_track_to_HDF5(TCC, 'A01c_ATL03_'+ ID_name+'_c', save_path_data)


# %% for testing
# plt.plot(TT['gt1l']['lons'], TT['gt1l']['lats'], '.')
# plt.plot(ID['pars']['start']['longitude'],ID['pars']['start']['latitude'], '.g', markersize= 12)
# plt.plot(ID['pars']['end']['longitude'],ID['pars']['end']['latitude'], '.r', markersize= 12)


# %%

# TT['gt1l']['delta_time'].min()
# SS['gt1l']['delta_time'].min()
#
# TT['gt1l']['delta_time'].max()
# SS['gt1l']['delta_time'].max()
