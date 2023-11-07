import os, sys
#execfile(os.environ['PYTHONSTARTUP'])
from icesat2_tracks.config.IceSAT2_startup import mconfig
"""
This script opens an ATL07 track and tests if there is sufficient data and maybe waves:
1) disects each beam such that they start in the MIZ and end at the pole.
2) checks if the variance decays from the MIZ poleward if data density is high enough.
3) generates plausible ATL03 track_names
4) Saves dummy files tracknames and reporting A01b_success_'track_name'.json file

"""



from threadpoolctl import threadpool_info, threadpool_limits
from pprint import pprint

import datetime
import h5py
from random import sample
import imp
import icesat2_tracks.ICEsat2_SI_tools.convert_GPS_time as cGPS
import icesat2_tracks.ICEsat2_SI_tools.io as io
from icesat2_tracks.ICEsat2_SI_tools.spectral_estimates import create_chunk_boundaries_unit_lengths, create_chunk_boundaries
import icesat2_tracks.ICEsat2_SI_tools.spectral_estimates as spec
import icesat2_tracks.ICEsat2_SI_tools.filter_regrid as regrid

import icesat2_tracks.local_modules.m_tools_ph3 as MT


import concurrent.futures as futures

import piecewise_regression

#import s3fs
#processed_ATL03_20190605061807_10380310_004_01.h5

#imp.reload(io)
track_name, batch_key, test_flag = io.init_from_input(sys.argv) # loads standard experiment

# track NH
#track_name, batch_key, test_flag = '20190301004639_09560201_005_01', 'NH_batch05', False # <-- !
#track_name, batch_key, test_flag = '20190101180843_00660201_005_01', 'SH_batch04', False

# track SH
#track_name, batch_key, test_flag = '20190101084259_00600201_005_01', 'SH_batch04', False
#track_name, batch_key, test_flag = '20190102130012_00780201_005_01', 'SH_batch04', False
#track_name, batch_key, test_flag = '20190101005132_00550201_005_01', 'SH_batch04', False # <-- !
#track_name, batch_key, test_flag = '20190101225136_00690201_005_01', 'SH_batch04', False

#track_name, batch_key, test_flag = '20190219063727_08070201_005_01', 'SH_publish', False


#track_name, batch_key, test_flag = '20190208142818_06440201_005_01', 'SH_publish', False


#print(track_name, batch_key, test_flag)
hemis, batch = batch_key.split('_')
#track_name= '20190605061807_10380310_004_01'

ATlevel= 'ATL07-02' if hemis == 'SH' else 'ATL07-01'
load_path   = mconfig['paths']['scratch'] +'/'+ batch_key +'/'
load_file   = load_path + ATlevel+'_'+track_name+'.h5'

save_path  = mconfig['paths']['work'] +'/'+ batch_key +'/A01b_ID/'
scratch_path = mconfig['paths']['scratch'] +'/'+ batch_key +'/'
plot_path  = mconfig['paths']['plot']+ '/'+hemis+'/'+batch_key+'/A01b/'
#bad_track_path =mconfig['paths']['work'] +'bad_tracks/'+ batch_key+'/'
MT.mkdirs_r(save_path)
plot_flag   = True
test_flag   = False #writes dummy variable for download files, instead of downloading
N_process   = 4
# username= "mhell@ucsd.edu"
# pword   = "@[49[4tK\-qBWB%5"

# %%
# test which beams exist:
all_beams = mconfig['beams']['all_beams']


try:
    f     = h5py.File(load_file, 'r')
except:
    print('file not found, exit')
    MT.json_save(name='A01b_'+track_name+'_success', path=save_path, data= {'reason':'ATL07 file not found, exit'})
    exit()

beams     = [b if b in f.keys() else None for b in all_beams]
imp.reload(regrid)
#track_poleward    = regrid.track_pole_ward_file(f, product='ATL10')
# print('poleward track is' , track_poleward)
# ATL03       =   h5py.File(load_file, 'r')
# ATL03['orbit_info'].keys()
# ATL03['orbit_info/lan'][:]
#
# ATL03['gt1l/freeboard_beam_segment/height_segments'].keys()

# %%
def cut_rear_data(xx0, dd0, N_seg= 20):
    """
    returns masks that cuts large variance in the back of the data
    """
    rear_mask = xx0*0 > -1 # True
    nsize0 = rear_mask.size

    cut_flag = True
    dd_old = -1

    print('inital length' , nsize0)

    #@jit(nopython=True, parallel= False)
    def adjust_length(var_list, rear_mask, cut_flag):

        #var_list = var_list if track_poleward else var_list[::-1]

        if var_list[0:3].mean()*2 < var_list[-1]:
            #print('cut last '+ str(100/N_seg) +'% of data')
            rear_mask[int(nsize* (N_seg-1) / N_seg):] = False
        else:
            cut_flag =  False

        #rear_mask = rear_mask if track_poleward else rear_mask[::-1]

        return rear_mask, cut_flag

    def get_var(sti):
        return dd[sti[0]: sti[1]].var()

    while cut_flag:
        dd= dd0[rear_mask]
        nsize = dd.size
        print('new length', nsize)
        if (nsize/N_seg) < 1:
            break
        stencil_iter = create_chunk_boundaries( int(nsize/N_seg), nsize,ov =0, iter_flag=True )
        var_list = np.array(list(map(get_var, stencil_iter)))
        #print(k, var_list)
        rear_mask, cut_flag = adjust_length(var_list, rear_mask, cut_flag)

        if nsize == dd_old:
            print('--- lengthen segments')
            N_seg -=1
            #cut_flag = False

        dd_old = nsize

    return rear_mask

def get_breakingpoints(xx, dd ,Lmeter= 3000):

    nsize = dd.size
    stencil_iter = spec.create_chunk_boundaries_unit_lengths( Lmeter, [ xx.min(), xx.max()],ov =Lmeter*3/4, iter_flag= True)
    iter_x = spec.create_chunk_boundaries_unit_lengths( Lmeter, [ xx.min(), xx.max()],ov =Lmeter*3/4, iter_flag= False)[1,:]

    def get_var(sti):
        mask = (sti[0] < xx) & (xx <= sti[1])
        return np.nanvar(dd[mask])

    var_list = np.array(list(map(get_var, stencil_iter)))

    x2, y2 =  iter_x/1e3, var_list

    x2= x2[~np.isnan(y2)]
    y2= y2[~np.isnan(y2)]

    convergence_flag =True
    n_breakpoints= 1
    while convergence_flag:
        pw_fit = piecewise_regression.Fit(x2, y2, n_breakpoints=1)
        print('n_breakpoints', n_breakpoints, pw_fit.get_results()['converged'])
        convergence_flag = not pw_fit.get_results()['converged']
        n_breakpoints += 1
        if n_breakpoints == 4:
            convergence_flag = False

    pw_results = pw_fit.get_results()
    if pw_results['converged']:
        if pw_results['estimates']['alpha1']['estimate'] < 0:
            print('decay at the front')
            print('n_breakpoints',pw_fit.n_breakpoints )

        breakpoint = pw_results['estimates']['breakpoint1']['estimate']
        return pw_results['estimates']['alpha1']['estimate'], pw_fit, breakpoint

    else:
        return np.nan, pw_fit, False

DD_slope  = pd.DataFrame(index =beams, columns= ['TF1', 'TF2'])
DD_data   = pd.DataFrame(index =beams, columns= ['TF1', 'TF2'])
DD_region = pd.DataFrame(index =beams, columns= ['TF1', 'TF2'])
DD_region[:] = (np.nan)
# DD_pos_start = pd.DataFrame(index =beams, columns= ['TF1_lon', 'TF1_lat', 'TF2_lon', 'TF2_lat'])
# DD_pos_end   = pd.DataFrame(index =beams, columns= ['TF1_lon', 'TF1_lat', 'TF2_lon', 'TF2_lat'])
DD_pos_start = pd.DataFrame(index =beams, columns= ['TF1', 'TF2'])
DD_pos_end   = pd.DataFrame(index =beams, columns= ['TF1', 'TF2'])


plot_flag = False
for k in beams:

    #k = 'gt2r'#beams[0]
    #imp.reload(io)
    print(k)
    try:
        T_freeboard = io.getATL07_beam(load_file, beam= k)
    except:
        print('failed to load beam')
        slope_test = False
        data_density  = False
        #return data_density, slope_test
        print('break -------', k, TF,  data_density, slope_test)
        continue

    # find devide such that each hemisphere is split into two parts, if data is there
    if hemis == 'SH':
        ###### for SH tracks
        lon_list = T_freeboard['ref']['longitude']
        mask1 = (lon_list[0]-5 < lon_list) & (lon_list < lon_list[0]+5)
        mask2 = ~mask1
        tot_size =T_freeboard['ref']['latitude'].shape[0]

        TF2, TF1 = None, None
        if (sum(mask1) > 1000):
            if mask1.iloc[-1]:
                TF2 = T_freeboard[mask1]
            else:
                TF1 = T_freeboard[mask1]
        if (sum(mask2) > 1000):
            TF2 = T_freeboard[mask2]

    else:
        ###### for NH tracks
        from scipy.ndimage.measurements import label
        #mask1 = label(T_freeboard['ref']['latitude'] < 88)[0] ==1

        break_point = abs(T_freeboard['ref']['latitude']-90).argmin()
        mask1 = T_freeboard['ref']['latitude'].index <= break_point
        mask2 = ~mask1
        tot_size =T_freeboard['ref']['latitude'].shape[0]

        # cut data accordingly
        if (sum(mask1)/tot_size < 0.05) or (sum(mask1) < 1000):
            TF1 = None
            TF2 = T_freeboard[mask2]
        elif (sum(mask2)/tot_size < 0.05) or (sum(mask2) < 1000):
            TF1 = T_freeboard[mask1]
            TF2 = None
        else:
            TF1 = T_freeboard[mask1]
            TF2 = T_freeboard[mask2]


    # check if sub-taable goes equatorward or not, then sort accordingly and define along-track axis
    def pole_ward_table(T):
        """
        Returns true if table goes poleward
        hdf5_file is a an HFD5 object in read mode
        """
        if T is None:
            return None
        time = T['time']['delta_time']
        lat = T['ref']['latitude']
        print('1st lat =' + str(abs(lat.iloc[time.argmin()])) , ';last lat =' + str(abs(lat.iloc[time.argmax()])) )

        return abs(lat.iloc[time.argmax()]) > abs(lat.iloc[time.argmin()])

    TF1_poleward = pole_ward_table(TF1)
    TF2_poleward = pole_ward_table(TF2)

    if TF1_poleward is None:
        TF1_poleward = not TF2_poleward

    if TF2_poleward is None:
        TF2_poleward = not TF1_poleward

    if TF1_poleward & TF2_poleward:
        raise ValueError('both parts are acending or decending')

    # assign Region to each subset, hemisphere dependent
    for TF,Tsel,TF_poleward in zip(['TF1', 'TF2'], [TF1, TF2], [TF1_poleward, TF2_poleward]):

        print(TF,TF_poleward)
        if (hemis == 'SH') & TF_poleward:
            region = ('10') # SO region
        elif (hemis == 'SH') & (not TF_poleward):
            region = ('12') # SO region
        elif (hemis == 'NH') & (TF_poleward):
            region = ('03','04') # assign subarctic and high-arctic region
        elif (hemis == 'NH') & (not TF_poleward):
            region = ('05','04') # assign subarctic and high-arctic region
        else:
            region =False

        if (Tsel is None):
            slope_test = False
            data_density  = False
            #return data_density, slope_test
            print('break -------', k, TF,  data_density, slope_test)
            continue

        else:
            # flip the beam section that is not poleward
            if not TF_poleward:
                print('TF polewards is ', TF_poleward)
                Tsel = Tsel.sort_values(('ref','seg_dist_x'), ascending=False).reset_index()

            # create local x axis
            Tsel['x'] = abs(Tsel['ref']['seg_dist_x'] -Tsel['ref']['seg_dist_x'].iloc[0])

        # ignore bad segments
        Tsel = Tsel[Tsel['heights']['height_segment_surface_error_est'] < 1e2]
        if (Tsel.size <= 50):
            #print('too small table, skip ')
            Tsel = None

        # if Tsel is None skip this itteration
        if Tsel is None:
            slope_test = False
            data_density  = False
            #return data_density, slope_test
            print('break -------', k, TF,  data_density, slope_test)
            continue
        else:
            data_density =Tsel.shape[0]/abs(Tsel['x'].max() - Tsel['x'].min()) # datapoints per meters

        # plt.plot(Tsel['x']/1e3, Tsel['ref']['beam_fb_sigma'], '.k')
        # plt.plot(Tsel['x']/1e3, Tsel['ref']['beam_fb_height'], '.r')
        # plt.plot(Tsel['x']/1e3, Tsel['freeboard']['height_segment_height'], '.')
        # #plt.xlim(60,70)
        # plt.ylim(-1, 5)
        # limit number of processes
        # % cut data in the back: only usefull for SH:
        xx0, dd0 = np.array(Tsel['x']), np.array(Tsel['heights']['height_segment_height'])
        if hemis is 'SH':
            # cut data around the contiental margin
            with threadpool_limits(limits=N_process, user_api='blas'):
                rear_mask = cut_rear_data(xx0, dd0)
        else:
            # assume all data points are valid for NH ...
            rear_mask = np.array(Tsel['x'] > -1)

        #print('density post cutting', len(xx0[rear_mask])/abs(xx0[rear_mask].max() - xx0[rear_mask].min()) )

        # if cutted data is too short, skip loop
        if len(xx0[rear_mask]) < 500:
            slope_test = False
            data_density  = False
            #return data_density, slope_test
            print('break -------', k, TF,  data_density, slope_test)
            continue

        # estmiate slope at the beginning
        with threadpool_limits(limits=N_process, user_api='blas'):
            #pprint(threadpool_info())
            slope_test, pw_fit, breakpoint = get_breakingpoints(xx0[rear_mask], dd0[rear_mask], Lmeter= 3000)

        if plot_flag:
            plt.figure()
            plt.plot(xx0[rear_mask]/1e3, dd0[rear_mask], '.k', markersize= 0.4)
            pw_fit.plot()
            plt.title(k +' '+ TF + ',  data=' +str(data_density) +  ', slope='+str(slope_test)  +  '\n' + track_name , loc= 'left')
            M.save_anyfig(plt.gcf(), name='A01b_'+track_name+'_'+k +'_'+ TF , path=plot_path)
            plt.close()
            #plt.show()

        # assign to tables
        DD_slope.loc[ k, TF] = slope_test
        DD_data.loc[  k, TF] = data_density
        DD_region.loc[k, TF] = region
        DD_pos_start.loc[k, TF]  =  Tsel.iloc[0]['ref']['longitude']  , Tsel.iloc[0]['ref']['latitude'],  Tsel.iloc[0]['ref']['seg_dist_x'],  Tsel.iloc[0]['time']['delta_time']
        DD_pos_end.loc[k,   TF]  =  Tsel.iloc[-1]['ref']['longitude'] , Tsel.iloc[-1]['ref']['latitude'],  Tsel.iloc[-1]['ref']['seg_dist_x'], Tsel.iloc[-1]['time']['delta_time']
        # DD_pos_start.loc[k, [TF+'_lon', TF+'_lat']]  =  Tsel.iloc[0]['ref']['longitude']  , Tsel.iloc[0]['ref']['latitude'],  Tsel.iloc[0]['ref']['seg_dist_x'],  Tsel.iloc[0]['time']['delta_time']
        # DD_pos_end.loc[k, [TF+'_lon', TF+'_lat']]    =  Tsel.iloc[-1]['ref']['longitude'] , Tsel.iloc[-1]['ref']['latitude'],  Tsel.iloc[-1]['ref']['seg_dist_x'], Tsel.iloc[-1]['time']['delta_time']
        print('result-------', k, TF, data_density, slope_test)


# %% check decisions

TT_start, TT_end = dict(), dict()
for Ti in DD_pos_start:
    print(Ti)

    ddtime_start, ddtime_end = list(), list()
    for k in all_beams:
        if (type(DD_pos_start[Ti][k]) is not tuple):
            ddtime_start.append(np.nan) # get latitude
            ddtime_end.append(np.nan)     # get latitude
        else:
            ddtime_start.append(DD_pos_start[Ti][k][1]) # get latitude
            ddtime_end.append(DD_pos_end[Ti][k][1])     # get latitude
            print('poleward check ', k , abs(DD_pos_start[Ti][k][1]) < abs(DD_pos_end[Ti][k][1]), abs(DD_pos_start[Ti][k][1]) ,  abs(DD_pos_end[Ti][k][1]))

    #print(ddtime_start, ddtime_end)
    TT_start[Ti] = DD_pos_start[Ti].iloc[np.array(ddtime_start).argmin()]
    TT_end[Ti] = DD_pos_end[Ti].iloc[np.array(ddtime_end).argmax()]
    try:
        print('poleward check sum', abs(TT_start[Ti][1]) < abs(TT_end[Ti][1]), abs(TT_start[Ti][1]) ,  abs(TT_end[Ti][1]))
    except:
        pass

#if plot_flag:

font_for_pres()
F = M.figure_axis_xy(10, 4  ,container = True)
#ax = F.fig.add_subplot(122, projection='polar')

for Ti,figp in zip(DD_pos_start, [121, 122]):
    ax = F.fig.add_subplot(figp, projection='polar')
    print(Ti)
    for k in all_beams:
        if (type(DD_pos_start[Ti][k]) is tuple):
            plt.scatter(  np.deg2rad( DD_pos_start[Ti][k][0]), DD_pos_start[Ti][k][1] ,s=20, color='green')#, label='start')
        if (type(DD_pos_end[Ti][k]) is tuple):
            plt.scatter(  np.deg2rad( DD_pos_end[Ti][k][0])  , DD_pos_end[Ti][k][1] ,s=20, color='red')#, label='end')

        if (type(DD_pos_start[Ti][k]) is tuple) & (type(DD_pos_end[Ti][k]) is tuple):
            plt.plot( [np.deg2rad( DD_pos_start[Ti][k][0]),  np.deg2rad( DD_pos_end[Ti][k][0])], [DD_pos_start[Ti][k][1], DD_pos_end[Ti][k][1]], color='black', linewidth=0.5 )# , width=0.5, edgecolor='none', facecolor= 'black')
        # plt.quiver( DD_pos_start[Ti][k][0], DD_pos_start[Ti][k][1], DD_pos_end[Ti][k][0]-DD_pos_start[Ti][k][0] , DD_pos_end[Ti][k][1]- DD_pos_start[Ti][k][1], scale=5)# , width=0.5, edgecolor='none', facecolor= 'black')

    #plt.ylim(TT_start[Ti][1]-1, TT_end[Ti][1] +1)

plt.title(track_name + ' '+hemis, loc= 'left')
plt.legend()
M.save_anyfig(plt.gcf(), name='A01b_'+track_name+'_'+ hemis+'_'+k  , path=plot_path)


# %%

# plt.plot(Tsel['x'], Tsel['ref']['latitude'], 'b.', label ='TF2', zorder=0)
# plt.scatter(Tsel['x'].iloc[0], Tsel['ref']['latitude'].iloc[0],s=50, color='green')
# plt.scatter(Tsel['x'].iloc[-1], Tsel['ref']['latitude'].iloc[-1],s=50, color='red')


#plt.close()
#plt.show()

# DD_pos_end
#
# DD_data
# DD_slope
#
# %%

# Test if 1st slope segment is negative. There might be better way to test for waves in the data
DD_slope_mask = DD_slope < 1e-3
# DD_slope < 1e-3
# DD_slope < 0

# if there is at leat one slope pair download data, otherwise write files and exit
if ( (DD_slope_mask.sum() > 1).sum() > 0) :
    print('download data')

else:
    print('no suffcient data, quit()')
    MT.json_save(name='A01b_'+track_name+'_success', path=save_path, \
    data= {'failed': 'True', 'reason':'no sufficient data' ,'slope': DD_slope.where(pd.notnull(DD_slope), 0).to_dict(), 'density':DD_data.where(pd.notnull(DD_data), 0).to_dict() })
    exit()

# %%
# initiate ID files
ATL03_proposed, ATL03_remote_link, ATL03_remote= [],[],[]
imp.reload(io)
for TF,TF_poleward in zip(['TF1', 'TF2'], [TF1_poleward, TF2_poleward]):
    iregion = DD_region[TF][DD_slope_mask[TF]]
    if len(iregion) !=0:
        #iregion2 = iregion[0][0] # for testing
        iregion2 = iregion[0]
        print(iregion2)
        # create track dict
        CID = io.case_ID(hemis+'_'+track_name)
        if type(iregion2) is str:  # This is the SH case
            CID.GRN = iregion2
            CID.set() # reset Case ID
            DD= {'case_ID':  CID.ID ,  'tracks' : {} }

            CIDr = io.case_ID(hemis+'_'+track_name)
            CIDr.GRN = iregion2
            CIDr.RL = '005'
            #ATL03_list.append('ATL03_'+CIDr.set_ATL03_trackname())
            #print(CIDr.get_granule() in remote_names)

            #ATL03_dummy= [CIDr.set_dummy()]
            ATL03_list = ['ATL03_'+CIDr.set_ATL03_trackname()]

            org_files, remote_files, remote_names = io.nsidc_icesat2_get_associated_file(ATL03_list, 'ATL03')

            # DD['tracks']['ATL03']   = ['ATL03_'+i for i in ATL03_list]
            # #DD['tracks']['ATL03_dummy']   = ATL03_dummy#['ATL03_'+i for i in ATL03_dummy]
            # ATL03_proposed         += org_files
            # ATL03_remote_link      +=remote_files
            # ATL03_remote           +=remote_names

        else: # this is the NH case
            CID.GRN =iregion2[0]
            CID.set() # reset Case ID
            DD= {'case_ID':  CID.ID ,  'tracks' : {} }

            ATL03_list= list()
            for i in iregion2:
                CIDr = io.case_ID(hemis+'_'+track_name)
                CIDr.GRN = i
                ATL03_list.append('ATL03_'+CIDr.set_ATL03_trackname())

            org_files, remote_files, remote_names = io.nsidc_icesat2_get_associated_file(ATL03_list, 'ATL03')

        DD['tracks']['ATL03']   = [rem.split('.')[0] for rem in remote_names] #['ATL03_'+i for i in ATL03_list]
        ATL03_proposed         += org_files
        ATL03_remote_link      += remote_files
        ATL03_remote           += remote_names

        DD['tracks']['ATL10']   = 'ATL10-' +CID.set_ATL10_trackname()
        DD['tracks']['ATL07']   = 'ATL07-' +CID.set_ATL10_trackname()

        # add other pars:
        DD['pars'] ={
        'poleward':TF_poleward, 'region': iregion2,
        'start': {'longitude': TT_start[TF][0], 'latitude': TT_start[TF][1], 'seg_dist_x': TT_start[TF][2], 'delta_time': TT_start[TF][3]},
        'end': {'longitude': TT_end[TF][0], 'latitude': TT_end[TF][1], 'seg_dist_x': TT_end[TF][2], 'delta_time': TT_end[TF][3]},
        'beams':list(DD_data[TF].index), 'density':list(DD_data[TF]), 'slope': list(DD_slope[TF])
            }
        # write .json ID file if files found


        if len(remote_names) !=0:
            MT.json_save2(name='A01b_ID_'+CID.ID, path=save_path, data= DD)
        else:
            print('no ATL03 track found for CID.ID')


ATL03_proposed      = list(set(ATL03_proposed))
ATL03_remote_link   = list(set(ATL03_remote_link))
ATL03_remote        = list(set(ATL03_remote))


if len(ATL03_remote) ==0:
    print('no ATL03 tracks found! quit()')
    MT.json_save(name='A01b_'+track_name+'_success', path=save_path, \
    data= {'failed': 'True', 'reason':'no ATL03 track found' ,'slope': DD_slope.where(pd.notnull(DD_slope), 0).to_dict(), 'density':DD_data.where(pd.notnull(DD_data), 0).to_dict() })
    exit()

product_directory, sd = ATL03_remote_link[0].split('/')[4], ATL03_remote_link[0].split('/')[5]


# %% download ATL03 file to scratch folder
def ATL03_download_worker(fname):
    io.ATL03_download(None,None, scratch_path, product_directory, sd,fname)
    print(fname, ' done')

if test_flag:
    for rname in remote_names:
        MT.save_pandas_table({'dummy_download':DD_slope},rname.split('.')[0], scratch_path)
else:
    with futures.ThreadPoolExecutor(max_workers=3) as executor:
        A = list( executor.map(ATL03_download_worker, ATL03_remote)  )


# linear version
# for rname in remote_names:
#     io.ATL03_download(username,pword, save_path, product_directory, sd,rname)

# %%
# print results and write files to exit
print('data density N/meter')
print(DD_data)

print('slopes')
print(DD_slope)

# write slope data for fun
#DD_merge = pd.concat({'density_Nperm':DD_data , 'slopes':DD_slope}, axis=1)
#DD_merge.to_html(save_path+ll_name+'.html')
#DD_merge.columns = ['-'.join(col).strip() for col in DD_merge.columns.values]
#MT.save_pandas_table({'T':DD_merge},ll_name, save_path)
#MT.json_save(name=ll_name, path=save_path, data= DD_merge.where(pd.notnull(DD_merge), 0).T.to_dict())
#DD_merge.to_json(save_path+ll_name+'.json', orient="records", lines=True)

# write success file
MT.json_save(name='A01b_'+track_name+'_success', path=save_path, \
data= {'failed': 'False', 'slope': DD_slope.where(pd.notnull(DD_slope), 0).to_dict(), 'density':DD_data.where(pd.notnull(DD_data), 0).to_dict() })
