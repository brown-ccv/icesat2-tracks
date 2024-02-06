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

import piecewise_regression

# import s3fs
# processed_ATL03_20190605061807_10380310_004_01.h5

# imp.reload(io)
track_name, batch_key, test_flag = io.init_from_input(
    sys.argv
)  # loads standard experiment
# track_name, batch_key, test_flag = '20190605061807_10380310_004_01', 'SH_batch01', False
# track_name, batch_key, test_flag = '20190207234532_06340210_004_01', 'SH_batch02', False
# track_name, batch_key, test_flag = '20190215184558_07530210_004_01', 'SH_batch02', False
# track_name, batch_key, test_flag = '20190219073735_08070210_004_01', 'SH_batch02', False

# track_name, batch_key, test_flag = '20190101101716_00610201_005_02', 'SH_batch04', False
# track_name, batch_key, test_flag = '20190102130012_00780201_005_02', 'SH_batch04', False
# track_name, batch_key, test_flag = '20190101040007_00570201_005_02', 'SH_batch04', False
# track_name, batch_key, test_flag = '20190101084259_00600201_005_02', 'SH_batch04', False
# track_name, batch_key, test_flag = '20190128235320_04820201_005_02', 'SH_batch04', False


# equatorward track
# track_name, batch_key, test_flag = '20190208154150_06440212_004_01', 'SH_batch02', False

# poleward track
# track_name, batch_key, test_flag = '20190209150245_06590210_004_01', 'SH_batch02', False
#

# print(track_name, batch_key, test_flag)
hemis, batch = batch_key.split("_")
# track_name= '20190605061807_10380310_004_01'

ATlevel = "ATL10-02" if hemis == "SH" else "ATL10-01"

load_path = mconfig["paths"]["scratch"] + "/" + batch_key + "/"
load_file = load_path + ATlevel + "_" + track_name + ".h5"

save_path = mconfig["paths"]["work"] + "/" + batch_key + "/" + "/A01b_ID_" + hemis + "/"

plot_path = (
    mconfig["paths"]["plot"]
    + "/"
    + hemis
    + "/"
    + batch_key
    + "/"
    + track_name
    + "/A01b/"
)
# bad_track_path =mconfig['paths']['work'] +'bad_tracks/'+ batch_key+'/'
MT.mkdirs_r(save_path)

# set pars

# define parameters:
# Nphoton_min = 5 # mininum need photons per stancil to return results


plot_flag = False
# %%
# test which beams exist:
all_beams = mconfig["beams"]["all_beams"]
high_beams = mconfig["beams"]["high_beams"]
# low_beams   = mconfig['beams']['low_beams']
try:
    f = h5py.File(load_file, "r")
except:
    print("file not found, exit")
    MT.json_save(
        name="A01b_success_" + track_name,
        path=save_path,
        data={"reason": "ALT10 file not found, exit"},
    )
    exit()

beams = [b if b in f.keys() else None for b in all_beams]

imp.reload(regrid)
# track_poleward    = regrid.track_pole_ward_file(f, product='ALT10')
# print('poleward track is' , track_poleward)
# ATL03       =   h5py.File(load_file, 'r')
# ATL03['orbit_info'].keys()
# ATL03['orbit_info/lan'][:]
#
# ATL03['gt1l/freeboard_beam_segment/height_segments'].keys()


# %%
def cut_rear_data(xx0, dd0, N_seg=20):
    """
    returns masks that cuts large variance in the back of the data
    """
    rear_mask = xx0 * 0 > -1  # True
    nsize0 = rear_mask.size

    cut_flag = True
    dd_old = -1

    print("inital length", nsize0)

    # @jit(nopython=True, parallel= False)
    def adjust_length(var_list, rear_mask, cut_flag):

        # var_list = var_list if track_poleward else var_list[::-1]

        if var_list[0:3].mean() * 2 < var_list[-1]:
            # print('cut last '+ str(100/N_seg) +'% of data')
            rear_mask[int(nsize * (N_seg - 1) / N_seg) :] = False
        else:
            cut_flag = False

        # rear_mask = rear_mask if track_poleward else rear_mask[::-1]

        return rear_mask, cut_flag

    def get_var(sti):
        return dd[sti[0] : sti[1]].var()

    while cut_flag:
        dd = dd0[rear_mask]
        nsize = dd.size
        print("new length", nsize)
        if (nsize / N_seg) < 1:
            break
        stencil_iter = create_chunk_boundaries(
            int(nsize / N_seg), nsize, ov=0, iter_flag=True
        )
        var_list = np.array(list(map(get_var, stencil_iter)))
        # print(k, var_list)
        rear_mask, cut_flag = adjust_length(var_list, rear_mask, cut_flag)

        if nsize == dd_old:
            print("--- lengthen segments")
            N_seg -= 1
            # cut_flag = False

        dd_old = nsize

    return rear_mask


def get_breakingpoints(xx, dd, Lmeter=3000):

    nsize = dd.size
    stencil_iter = spec.create_chunk_boundaries_unit_lengths(
        Lmeter, [xx.min(), xx.max()], ov=Lmeter * 3 / 4, iter_flag=True
    )
    iter_x = spec.create_chunk_boundaries_unit_lengths(
        Lmeter, [xx.min(), xx.max()], ov=Lmeter * 3 / 4, iter_flag=False
    )[1, :]

    def get_var(sti):
        mask = (sti[0] < xx) & (xx <= sti[1])
        return np.nanvar(dd[mask])

    var_list = np.array(list(map(get_var, stencil_iter)))

    x2, y2 = iter_x / 1e3, var_list

    x2 = x2[~np.isnan(y2)]
    y2 = y2[~np.isnan(y2)]

    convergence_flag = True
    n_breakpoints = 1
    while convergence_flag:
        pw_fit = piecewise_regression.Fit(x2, y2, n_breakpoints=1)
        print("n_breakpoints", n_breakpoints, pw_fit.get_results()["converged"])
        convergence_flag = not pw_fit.get_results()["converged"]
        n_breakpoints += 1
        if n_breakpoints == 4:
            convergence_flag = False

    pw_results = pw_fit.get_results()
    if pw_results["converged"]:
        if pw_results["estimates"]["alpha1"]["estimate"] < 0:
            print("decay at the front")
            print("n_breakpoints", pw_fit.n_breakpoints)

        breakpoint = pw_results["estimates"]["breakpoint1"]["estimate"]
        return pw_results["estimates"]["alpha1"]["estimate"], pw_fit, breakpoint

    else:
        return np.nan, pw_fit, False


DD_slope = pd.DataFrame(index=beams, columns=["TF1", "TF2"])
DD_data = pd.DataFrame(index=beams, columns=["TF1", "TF2"])
DD_region = pd.DataFrame(index=beams, columns=["TF1", "TF2"])

plot_flag = True
for k in beams:
    # k = beams[1]
    # imp.reload(io)
    print(k)
    try:
        T_freeboard, T_leads = io.getATL10_beam(load_file, beam=k)
    except:
        print("failed to load beam")
        slope_test = False
        data_flag = False
        # return data_flag, slope_test
        print("break -------", k, TF, data_flag, slope_test)
        continue
    ###### for SH tracks

    # split tracks:
    long_diffs = T_freeboard["freeboard"]["longitude"].diff()
    isplit = abs(long_diffs).argmax()
    if abs(long_diffs[isplit]) < 90:
        TF1 = T_freeboard
        TF2 = None
    else:
        TF1 = T_freeboard.iloc[0:isplit]
        TF2 = T_freeboard.iloc[isplit:]

    def pole_ward_table(T):
        """
        Returns true if table goes poleward
        hdf5_file is a an HFD5 object in read mode
        """

        time = T["time"]["delta_time"]
        lat = T["ref"]["latitude"]
        print(
            "1st lat =" + str(abs(lat.iloc[time.argmin()])),
            ";last lat =" + str(abs(lat.iloc[time.argmax()])),
        )

        return abs(lat.iloc[time.argmax()]) > abs(lat.iloc[time.argmin()])

    TF1_poleward = pole_ward_table(TF1)
    if TF2 is not None:
        TF2_poleward = pole_ward_table(TF2)

        if TF1_poleward & TF2_poleward:
            raise ValueError("both parts are acending or decending")
    else:
        TF2_poleward = not TF1_poleward

    # flip the beam section that is not poleward
    if TF1_poleward & (TF2 is not None):
        print("TF2 is ", TF2_poleward)
        TF2 = TF2.sort_values(("ref", "seg_dist_x"), ascending=False).reset_index()
        # Tsel2 = T_leads.sort_values('lead_dist_x', ascending=False).reset_index()
        # Tsel2['x'] = abs(Tsel2['lead_dist_x'] -Tsel2['lead_dist_x'][0])
    else:
        print("TF1 is ", TF2_poleward)
        TF1 = TF1.sort_values(("ref", "seg_dist_x"), ascending=False).reset_index()
        # TF1['x'] = abs(TF1['ref']['seg_dist_x'] -TF1['ref']['seg_dist_x'][0])
        # Tsel2 = T_leads#['lead_dist_x']
        # Tsel2['x'] = abs(Tsel2['lead_dist_x'] -Tsel2['lead_dist_x'][0])

    # create local x axis
    TF1["x"] = abs(TF1["ref"]["seg_dist_x"] - TF1["ref"]["seg_dist_x"].iloc[0])
    if TF2 is not None:
        TF2["x"] = abs(TF2["ref"]["seg_dist_x"] - TF2["ref"]["seg_dist_x"].iloc[0])

    # define Region

    for TF, Tsel, TF_polward in zip(
        ["TF1", "TF2"], [TF1, TF2], [TF1_poleward, TF2_poleward]
    ):

        if (hemis == "SH") & TF_polward:
            region = "10"
        elif (hemis == "SH") & (not TF_polward):
            region = "12"
        elif (hemis == "NH") & (TF_polward):
            region = ("03", "04")
        elif (hemis == "NH") & (not TF_polward):
            region = ("05", "04")
        else:
            region = False

        # cut high sigma values
        if Tsel is None:
            slope_test = False
            data_flag = False
            # return data_flag, slope_test
            print("break -------", k, TF, data_flag, slope_test)
            continue

        Tsel = Tsel[Tsel["ref"]["beam_fb_sigma"] < 1e2]
        if Tsel.size <= 50:
            # print('too small table, skip ')
            Tsel = None

        if Tsel is None:
            slope_test = False
            data_flag = False
            # return data_flag, slope_test
            print("break -------", k, TF, data_flag, slope_test)
            continue
        else:
            data_flag = Tsel.shape[0] / abs(
                Tsel["x"].max() - Tsel["x"].min()
            )  # datapoints per meters

        # plt.plot(Tsel['x']/1e3, Tsel['ref']['beam_fb_sigma'], '.k')
        # plt.plot(Tsel['x']/1e3, Tsel['ref']['beam_fb_height'], '.r')
        # plt.plot(Tsel['x']/1e3, Tsel['freeboard']['height_segment_height'], '.')
        # #plt.xlim(60,70)
        # plt.ylim(-1, 5)

        # % curt data in the back
        xx0, dd0 = np.array(Tsel["x"]), np.array(
            Tsel["freeboard"]["height_segment_height"]
        )
        rear_mask = cut_rear_data(xx0, dd0)

        # print('density post cutting', len(xx0[rear_mask])/abs(xx0[rear_mask].max() - xx0[rear_mask].min()) )
        if len(xx0[rear_mask]) < 500:  # if cutted data is too short
            slope_test = False
            data_flag = False
            # return data_flag, slope_test
            print("break -------", k, TF, data_flag, slope_test)
            continue

        # estmiate slope at the beginning
        slope_test, pw_fit, breakpoint = get_breakingpoints(
            xx0[rear_mask], dd0[rear_mask], Lmeter=3000
        )

        if plot_flag:
            plt.figure()
            plt.plot(xx0[rear_mask] / 1e3, dd0[rear_mask], ".k", markersize=0.4)
            pw_fit.plot()
            plt.title(
                k
                + " "
                + TF
                + ",  data="
                + str(data_flag)
                + ", slope="
                + str(slope_test)
                + "\n"
                + track_name,
                loc="left",
            )
            M.save_anyfig(
                plt.gcf(),
                name="A01b_" + track_name + "_" + k + "_" + TF,
                path=plot_path,
            )
            plt.close()
            # plt.show()

        DD_slope.loc[k, TF] = slope_test
        DD_data.loc[k, TF] = data_flag
        DD_region.loc[k, TF] = region
        print("result-------", k, TF, data_flag, slope_test)

# DD_data
DD_slope_mask = DD_slope < 0

if (DD_slope_mask.sum() > 1).sum() > 0:
    print("download data")

else:
    print("no suffcient data, quit()")
    ll_name = ATlevel + "_stats_" + track_name + "_fail"
    DD_merge = pd.concat({"density_Nperm": DD_data, "slopes": DD_slope}, axis=1)
    DD_merge.to_html(save_path + ll_name + ".html")
    # DD_merge.columns = ['-'.join(col).strip() for col in DD_merge.columns.values]
    MT.save_pandas_table({"T": DD_merge}, ll_name, save_path)
    # DD_merge.columns = ['-'.join(col).strip() for col in DD_merge.columns.values]
    # MT.json_save(name=ll_name, path=save_path, data= DD_merge.where(pd.notnull(DD_merge), 0).T.to_dict())
    MT.json_save(
        name="A01b_success_" + track_name,
        path=save_path,
        data=DD_slope.where(pd.notnull(DD_slope), 0).to_dict(),
    )
    exit()

region_list = DD_region[DD_slope_mask].to_numpy().flatten().astype("float")
region_list = list(set(region_list[~np.isnan(region_list)]))
region_list = [str(int(i)) for i in region_list]
print("region(s) ", region_list)


ALT03_list = list()
for i in region_list:
    track_segs = track_name.split("_")
    track_segs[1] = track_segs[1][0:-2] + i  # correct cycle number
    track_segs[3] = "01"  # correct version number
    ALT03_trackname = "_".join(track_segs)
    print(ALT03_trackname)
    ALT03_list.append(ALT03_trackname)

# DD_data
# DD_region
print("data density N/meter")
print(DD_data)

print("slopes")
print(DD_slope)

# DD_slope.to_html()
for ll in ALT03_list:
    ll_name = "ALT03_stats_" + ll
    DD_merge = pd.concat({"density_Nperm": DD_data, "slopes": DD_slope}, axis=1)
    DD_merge.to_html(save_path + ll_name + ".html")
    # DD_merge.columns = ['-'.join(col).strip() for col in DD_merge.columns.values]
    MT.save_pandas_table({"T": DD_merge}, ll_name, save_path)
    # MT.json_save(name=ll_name, path=save_path, data= DD_merge.where(pd.notnull(DD_merge), 0).T.to_dict())

    # DD_merge.to_json(save_path+ll_name+'.json', orient="records", lines=True)

MT.json_save(
    name="A01b_success_" + track_name,
    path=save_path,
    data=DD_slope.where(pd.notnull(DD_slope), 0).to_dict(),
)
