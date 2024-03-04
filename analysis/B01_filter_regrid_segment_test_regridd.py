import os, sys
#execfile(os.environ['PYTHONSTARTUP'])

"""
This file open a ICEsat2 track applied filters and corections and returns smoothed photon heights on a regular grid in an .nc file.
This is python 3
"""
# exec(open(os.environ['PYTHONSTARTUP']).read())
# exec(open(STARTUP_2019_DP).read())

exec(open(os.environ['PYTHONSTARTUP']).read())
exec(open(STARTUP_2021_IceSAT2).read())

import datetime
import h5py
from random import sample
import imp
import ICEsat2_SI_tools.convert_GPS_time as cGPS
import ICEsat2_SI_tools.io as io
from spectral_estimates import create_chunk_boundaries_unit_lengths, create_chunk_boundaries
import spectral_estimates as spec
import m_tools_ph3 as MT
import filter_regrid as regrid

import concurrent.futures as futures



#import s3fs
#processed_ATL03_20190605061807_10380310_004_01.h5

#imp.reload(io)
track_name, batch_key, test_flag = io.init_from_input(sys.argv) # loads standard experiment
#track_name, batch_key, test_flag = '20190605061807_10380310_004_01', 'SH_batch01', False
#track_name, batch_key, test_flag = '20190207234532_06340210_004_01', 'SH_batch02', False
#track_name, batch_key, test_flag = '20190215184558_07530210_004_01', 'SH_batch02', False
#track_name, batch_key, test_flag = '20190219073735_08070210_004_01', 'SH_batch02', False

track_name, batch_key, test_flag = '20190215184558_07530210_004_01', 'SH_batch02', False



# equatorward track
#track_name, batch_key, test_flag = '20190208154150_06440212_004_01', 'SH_batch02', False

# poleward track
#track_name, batch_key, test_flag = '20190209150245_06590210_004_01', 'SH_batch02', False
#

#print(track_name, batch_key, test_flag)
hemis, batch = batch_key.split('_')
#track_name= '20190605061807_10380310_004_01'
ATlevel= 'ATL03'

load_path   = mconfig['paths']['scratch'] +'/'+ batch_key +'/'
load_file   = load_path + 'processed_'+ATlevel+'_'+track_name+'.h5'

save_path  = mconfig['paths']['work'] +'/B01_regrid_'+hemis+'/'

plot_path = mconfig['paths']['plot']+ '/'+hemis+'/'+batch_key+'/'+track_name +'/B01/'
bad_track_path =mconfig['paths']['work'] +'bad_tracks/'+ batch_key+'/'
MT.mkdirs_r(save_path)

# set pars

# define parameters:
Lmeter      = 20 # stencil length in units of 'dist'; likely in meters the resulting resolution is L/2
Nphoton_min = 5 # mininum need photons per stancil to return results

Lmeter_large= 100e3 # stancil width for testing photon density. stancils do not overlab.
minium_photon_density = 0.02 # minimum photon density per meter in Lmeter_large chunk to be counted as real signal

plot_flag   = True
Nworkers    = 4        # number of threads for parallel processing
Nworkers_process = 3  # number of threads for parallel processing
# %%
# test which beams exist:
all_beams   = mconfig['beams']['all_beams']
high_beams  = mconfig['beams']['high_beams']
# low_beams   = mconfig['beams']['low_beams']

f         = h5py.File(load_file, 'r')
beams     = [b if b in f.keys() else None for b in all_beams]
imp.reload(regrid)
track_poleward    = regrid.track_pole_ward_file(f)
print('poleward track is ' , track_poleward)
# Load fata and apply height corrections
# This needs version 2 of the ALT 03 dataset

# ATL03       =   h5py.File(load_file, 'r')

#ATL03[k+'/heights/ph_id_channel'][0:100]
#accent = regrid.track_type( B[beams_list[0]] )


# %%
hist    = 'Beam stats'
B       = dict()
B1save  = dict()
SEG     = dict()
for k in beams:
    #k = beams[0]
    imp.reload(io)
    print(k)

    T, seg = io.getATL03_beam(load_file, beam= k)



    print('loaded')
    T = T[T['mask_seaice']] # only take sea ice points, no ocean points

    ho = k
    ho = MT.add_line_var(ho, 'size', str(T.shape[0]))
    ho = MT.add_line_var(ho, 'by confidence levels:' + str(np.arange(0, 5)), [ (T['signal_confidence'] == i).sum() for i in np.arange(0, 5) ])

    # filter:
    Tsel    = T[ (T['heights']<100)  & (T['heights'] > -100) ]# & (T['delta_time']>5) & (T['delta_time']<24) ]

    # if len(Tsel) == 0:
    #     ho  = MT.add_line_var(ho, 'no photons found', '')
    #     #Tsel= T[(T['signal_confidence'] ==-1 ) & (T['heights']<100)  & (T['heights'] > -100) ]# & (T['delta_time']>5) & (T['delta_time']<24) ]

    Tsel_c  = io.getATL03_height_correction(load_file)
    Tsel_c  = Tsel_c[Tsel_c['dem_h'] < 1e5] # cute out weird references
    Tsel2 = regrid.correct_heights(Tsel, Tsel_c).reset_index(drop=True)# correct height
    print('height corrected')

    ### cut data at the rear that has too much variance
    # cut last segments of data until variance is similar
    rear_mask = np.array(Tsel2.index) > -1 # True
    shape_old = Tsel2.shape
    N_seg= 20

    cut_flag = True
    while cut_flag:
        dd= Tsel2['heights_c'][rear_mask]
        nsize = dd.size
        stencil_iter = create_chunk_boundaries( int(nsize/N_seg), nsize,ov =0, iter_flag=True )

        def get_var(sti):
            return dd[sti[0]: sti[1]].var()

        var_list = np.array(list(map(get_var, stencil_iter)))
        #print(var_list)

        if track_poleward:
            if var_list[0:3].mean()*10 < var_list[-1]:
                #print('cut last '+ str(100/N_seg) +'% of data')
                rear_mask[int(nsize* (N_seg-1) / N_seg):] = False
            else:
                cut_flag =  False
        else:
            if var_list[-3:].mean()*10 < var_list[0]:
                #print('cut last '+ str(100/N_seg) +'% of data')
                #int(nsize* (N_seg-1) / N_seg)
                rear_mask[: int(nsize* 1 / N_seg)] = False
            else:
                cut_flag =  False


    Tsel2['process_mask'] = rear_mask
    B1save[k]             = Tsel2
    B[k]                  = Tsel2[rear_mask].drop(columns='process_mask')
    SEG[k]                = seg

    ho      = MT.add_line_var(ho, 'cutted ', str( np.round( 100 *(rear_mask.size -rear_mask.sum() ) / rear_mask.size, 0)) + '% in the back of the track' )
    ho      = MT.add_line_var(ho, 'selected size', str(Tsel.shape[0]))
    ho      = MT.add_line_var(ho, 'final size ', str(Tsel_c.shape[0]))
    print(ho)

    hist    = MT.write_log(hist, ho)

print('done with 1st loop')

# %% define x- coodindate

# find earliest segment length that is used.
# this is on the equatorward side for poleward track, or
# on the poleward side for equatorward tracks.

total_segment_dist_x_min= list()
for k,I in SEG.items():
    total_segment_dist_x_min.append( I['segment_dist_x'].min() )
total_segment_dist_x_min = min(total_segment_dist_x_min)



#
#
# plt.plot( np.diff( T2.index), '.', markersize = 0.3 )
#
# plt.plot( np.diff( T.index), '.', markersize = 0.3 )
#
# T3 = T2
# plt.plot( np.diff( T3.index), '.', markersize = 0.3 )
#
# plt.plot(T['delta_time'] ,  T.index , '.', markersize = 0.3 )
T = B['gt1r']
seg = SEG['gt1r']
T
#T = T.sort_values('lats').reset_index(drop=True)
#T.sort_values('lats')
# find positions where segmetn length is reset
time_ticks = np.insert(np.diff(T['along_track_distance']), 0,0 )

#plt.plot(T['along_track_distance'][0:500], '.')

Ti = T[-500:]
plt.plot(Ti['along_track_distance'], '.')
plt.plot(np.diff(Ti['along_track_distance']))


T
# %%
#Ti = T[10200:10250]
Ti = T[20000:20500]
#Ti = T[20000:20550]

#plt.plot(Ti['delta_time'], Ti['along_track_distance'], '.')
#plt.plot(Ti['delta_time'], np.insert(np.diff(Ti['along_track_distance']), 0,0 ) )
delta_onehalf =  seg['delta_time'].diff()/2
delta_onehalf.iloc[0] = delta_onehalf.iloc[1]
seg['delta_half_time']= seg['delta_time']  - delta_onehalf #+ 5e-5
#seg['delta_half_time']= seg['delta_time']  - seg['delta_time'].diff().mean()/2# - 1e-5
#seg['segment_length']
segi  =seg[ (Ti['delta_time'].min() - 2 * delta_onehalf <= seg['delta_time'] ) & ( Ti['delta_time'].max() + 3 * delta_onehalf >= seg['delta_time'] ) ]
#segi.shape

#Ti['ph_id_count']
#segi['ph_index_beg'] + segi['reference_photon_index'] -1

# %%

# %
plt.plot(Ti['delta_time'], Ti['delta_time'] *0, '.')

plt.plot( segi['delta_time'], segi['delta_time'] *0    , '.r' )
plt.plot(Ti['delta_time'], Ti['along_track_distance'], '.')

# cutting using delta time from segments
for i in segi['delta_time']:
    plt.gca().axvline(i, linewidth = 0.5, color= 'black')
plt.plot(Ti['delta_time'], np.digitize(Ti['delta_time'], segi['delta_time'] ) , '.k')

# cutting usind delta time from segments shifted
for i in segi['delta_half_time']:
    plt.gca().axvline(i, linewidth = 0.5, color= 'blue')

plt.xlabel('delta_time')
#np.digitize(T['delta_time'], T[ time_ticks  < - 10 ]['delta_time'])

# %%

plt.plot(Ti['delta_time'], Ti['delta_time'] *0, '.')

plt.plot( segi['delta_time'], segi['delta_time'] *0    , '.r' )
plt.plot(Ti['delta_time'], Ti['along_track_distance'], '.')

# cutting usind delta time from segments shifted
for i in segi['delta_half_time']:
    plt.gca().axvline(i, linewidth = 0.5, color= 'blue')

segi['delta_half_time'].max() < Ti['delta_time'].max()

Ti2 = Ti[ (Ti['delta_time'] > segi['delta_half_time'].iloc[0]) & (Ti['delta_time'] < segi['delta_half_time'].iloc[-1])]
bin_labels = np.digitize( Ti2['delta_time'], segi['delta_half_time'], right = True )
plt.plot(Ti['delta_time'], bin_labels +5 , '.b')

Ti2['delta_time0'] = Ti2['delta_time'] - Ti2['delta_time'].min()
Ti2['lats0'] = Ti2['lats'] - Ti2['lats'].max()
for i in np.arange(bin_labels.max()):
    print( Ti2[[ 'along_track_distance', 'delta_time0', 'lats0']][bin_labels ==i] )

#list(Ti2['delta_time'])
#bin_labels.min()
#bin_labels.max()
repeats.shape

SS = segi['segment_dist_x']
repeats = np.bincount(bin_labels, minlength =SS.shape[0])

# check if repeats sumup
if repeats.sum() != Ti2.shape[0]:
    print('repeats do not sum up')

SS = SS.repeat(repeats)
SS.index = Ti2.index

Ti2['x'] =  SS + Ti2['along_track_distance']
#list(SS.repeat(repeats).reset_index(drop=True)

#segi[segi['ph_index_beg'] != 0]['seg_ID_local'] = list(set(bin_labels))

#Tg= Ti.groupby(by= 'seg_ID_local', axis=0, group_keys=seg['delta_time'] )

#Tii, segii = Ti.loc[Tg.groups[1]], segi


# %%



# %%

#T2 = Tg.apply(add_seg_length, segi)

#list(T2['x']  - T2['x'].iloc[0] )
plt.plot(Ti2['delta_time'],  Ti2['x']     , 'b.')
plt.xlabel('delta_time')
#np.digitize(T['delta_time'], T[ time_ticks  < - 10 ]['delta_time'])
#x_interp = np.interp(Ti2['delta_time'],  segi['delta_time'], segi['segment_dist_x'] )

plt.plot( segi['delta_half_time'], segi['segment_dist_x'] + segi['segment_length'] , 'g-' )
# plt.plot( Ti2['delta_time'], x_interp  - x_interp[0], '.' )


def find_anomalie_photons(Ti2, segi):
    x_interp = np.interp(Ti2['delta_time'],  segi['delta_half_time'], segi['segment_dist_x'] + segi['segment_length'])

    diff_x  = x_interp -  Ti2['x']
    diff_x = abs(diff_x-diff_x.mean())
    return diff_x > 3 *diff_x.std(), x_interp

fail_mask, x_interp = find_anomalie_photons(Ti2, segi)

#Ti3= Ti2[~fail_mask]

Ti2['x'][fail_mask] = x_interp[fail_mask]

plt.plot(Ti2['delta_time'],  Ti2['x']    , 'r.')

plt.plot(Ti2['delta_time'][fail_mask],  x_interp[fail_mask]    , 'g.', markersize= 20)
#plt.plot(Ti2['delta_time'],  x_interp   , 'g.')




# %%
# def make_x_coorindate(k):
#
#     """
#     Returns the "true" along track coordindate but finding the correpsonding segment length
#     also adds the segment_ID to the main table T
#     """
#     print(k, ' make coodindate')
#     T, seg= B[k], SEG[k]
#
#     # make sure data is strictly ordered by delta_time
#     T = T.sort_values('delta_time').reset_index(drop=True)
#     # find positions where segmetn length is reset
#     time_ticks = np.insert(np.diff(T['along_track_distance']), 0,0 )
#     # define separators for photons
#     T['seg_ID_local'] = np.digitize(T['delta_time'], T[ time_ticks  < - 10 ]['delta_time'])
#     # group data by this separation
#     Tg= T.groupby(by= 'seg_ID_local', axis=0, group_keys=seg['delta_time'] )
#
#     def add_seg_length(Ti, seg):
#         try:
#             seg_data    = seg[seg['delta_time'] < Ti['delta_time'].iloc[0]].iloc[-1]
#             seg_l       = seg_data['segment_dist_x'] - total_segment_dist_x_min
#             seg_ID      = seg_data['segment_id']
#         except:
#             seg_l = np.nan
#             seg_ID =  np.nan
#
#         Ti['x'] = seg_l.astype('float64') + Ti['along_track_distance'].astype('float64')
#         Ti['seg_ID'] = seg_ID
#         return Ti

# # apply segment addition and return
# return k, Tg.apply(add_seg_length, seg)


def make_x_coorindate(k):

    """
    Returns the "true" along track coordindate but finding the correpsonding segment length
    also adds the segment_ID to the main table T
    """
    print(k, ' make coodindate')
    T, seg= B[k], SEG[k]

    # make sure data is strictly ordered by delta_time
    T = T.sort_values('delta_time').reset_index(drop=True)

    # find positions where segmetn length is reset
    # shifts segment length postions in time
    delta_onehalf =  seg['delta_time'].diff()/2
    delta_onehalf.iloc[0] = delta_onehalf.iloc[1]
    seg['delta_half_time']= seg['delta_time']  - delta_onehalf - 1e-5

    # cur phontos that are not in segmentns
    T2 = T[ (T['delta_time'] > seg['delta_half_time'].iloc[0]) & (T['delta_time'] < seg['delta_half_time'].iloc[-1])]
    bin_labels = np.digitize( T2['delta_time'], seg['delta_half_time'], right = True )

    # select relevant data
    SS = seg['segment_dist_x']
    SS_sid = seg['segment_id']

    repeats = np.bincount(bin_labels, minlength =SS.shape[0])

    # check if repeats sumup
    if repeats.sum() != T2.shape[0]:
        print('repeats do not sum up')

    # repeat  segment dat accoridng to photons
    SS = SS.repeat(repeats)
    SS.index = T2.index
    SS_sid = SS_sid.repeat(repeats)
    SS_sid.index = T2.index
    # define new coordinate
    T2['x'] =  SS + T2['along_track_distance']
    T2['segment_id'] =  SS_sid

    # find bad photons
    def find_anomalie_photons(Ti2, segi):
        x_interp = np.interp(Ti2['delta_time'],  segi['delta_time'], segi['segment_dist_x'] )

        diff_x  = x_interp -  Ti2['x']
        diff_x = abs(diff_x-diff_x.mean())
        return diff_x > 3 *diff_x.std() , x_interp

    fail_mask, x_interp = find_anomalie_photons(T2, seg)

    print('weird photon fraction:' ,  sum(fail_mask)/ fail_mask.size)
    # aply fail mask
    #T2= T2[~fail_mask]
    T2['x'][fail_mask] = x_interp[fail_mask]

    return k, T2


#make_x_coorindate(k)

with futures.ProcessPoolExecutor(max_workers=Nworkers_process) as executor:
    A = list( executor.map(make_x_coorindate, all_beams)  )

# %%
B= dict()
for I in A: # collect returns from from mapping
    k               = I[0]
    B[ k ]          = I[1][::-1]

    if ~track_poleward: # invert x- coordinate if there is an equatorward track
        B[k]            = B[k].reset_index(drop=True)
        B[k]['x_true']  = B[k]['x']
        B[k]['x']       = abs(B[k]['x'] - B[k]['x'].iloc[0])
    else:
        B[k]['x_true']  = B[k]['x']


dist_list   = np.array([np.nan, np.nan])
for k in B.keys():
    dist_list = np.vstack([ dist_list, [  B[k]['x'].iloc[0] , B[k]['x'].iloc[-1] ]  ])

del A

# define latitude limits
# lat_lims, lon_lims, accent = lat_min_max(B, all_beams, accent = None)


# %% test ave photon density abnd quit if necessary
track_dist_bounds   = [ np.nanmin(dist_list[:, 0], 0) , np.nanmax(dist_list[:, 1], 0) ]
length_meter        = abs(track_dist_bounds[1] - track_dist_bounds[0])
#length_meter       = (abs(lat_lims[1])  - abs(lat_lims[0])) * 110e3
p_densities_r       = list()
p_densities_l       = list()

for k, I in B.items():
    if 'r' in k:
        p_densities_r.append( I.shape[0] /length_meter)
    else:
        p_densities_l.append( I.shape[0] /length_meter)

if (np.array(p_densities_l).mean() < 0.5) & (np.array(p_densities_r).mean() < 0.5): # in photons per meter
    print('photon density too low, this track is classified as bad track and pushed to bad list')
    MT.json_save(track_name, bad_track_path, {'densities': [ np.array(p_densities_r).mean(), np.array(p_densities_l).mean()] , 'date': str(datetime.date.today()) })
    print('exit.')
    exit()


# %% save corrected data and delete from cash
#io.save_pandas_table(B1save, track_name + '_B01_corrected'  , save_path) # all photos but heights adjusted and with distance coordinate
del B1save

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
F = M.figure_axis_xy(4, 3, view_scale = 0.7)


for k,I in B.items():
    lmask = (I['lats'] > -68.1) & (I['lats'] < -67.7)
    plt.plot(I['x'][lmask] - I['x'].iloc[0]  ,  I['lats'][lmask]   ,'.' , markersize = 0.2)


#plt.xlim(, -67.8)

plt.xlabel('x')
plt.ylabel('lat')
#F.save_light(path= plot_path, name='B01_ALT03_'+track_name+'_tracks_check_lat_x')

# %%
F = M.figure_axis_xy(4, 3, view_scale = 0.7)
for k,I in B.items():
    plt.plot( I['delta_time']  , I['lats'], '.' , markersize = 0.3)


plt.xlabel('delta time')
plt.ylabel('lat')
F.save_light(path= plot_path, name='B01_ALT03_'+track_name+'_tracks_check_time_lat')
# %%
F = M.figure_axis_xy(4, 3, view_scale = 0.7)

for k,I in B.items():
    plt.plot( I['delta_time']  , I['x'], '.' , markersize = 0.3)

plt.xlabel('delta time')
plt.ylabel('x')

F.save_light(path= plot_path, name='B01_ALT03_'+track_name+'_tracks_check_time_x')

# %%
##### 1.) derive common axis for beams and filter out low density area at the beginning
print('filter out low density area at the beginning')

def derive_axis_and_boundaries(key):
    #key = 'gt3r'
    print(key)
    #T2   #      = regrid.derive_axis(B[key], lat_lims).sort_values('dist')
    T2 = B[key]#['x']

    x0         = get_better_lower_boundary(Lmeter_large, np.array(T2['x']))

    print( 'cut off ' , 100 * (1 - T2[T2['x'] > x0].shape[0]/T2.shape[0])  , '% off all data points at the beginning' )
    T2         = T2[T2['x'] >x0] # cut off low density start

    return key, T2, [T2['x'].min(), T2['x'].max()]

def get_better_lower_boundary(Lmeter_large, dd):

    # T2         = regrid.derive_axis(B[key], lat_lims)
    #dd = np.array(T2['x'])
    stencil_iter = spec.create_chunk_boundaries_unit_lengths( Lmeter_large, [ dd.min(), dd.max()],ov =0, iter_flag= True)

    def get_density(sti):
        return sti[0], sum( (sti[0] <= dd) & (dd < sti[-1]) ) / Lmeter_large

    with futures.ThreadPoolExecutor(max_workers= Nworkers_process) as executor_sub:
        #var_list = np.array(list(map(get_density, stencil_iter)))
        var_list = np.array(list(executor_sub.map(get_density, stencil_iter)))

    var_list = np.array(var_list)
    #var_list[:,0] = np.random.rand(10)
    #sort var_list
    var_list = var_list[var_list[:,0].argsort(), :]
    #print(var_list)
    if sum(var_list[:,1] > minium_photon_density) > 1:
        first_stencil = next((i for i, j in enumerate(var_list[:,1] > minium_photon_density) if j), None) #- 1
        stencil_iter  = spec.create_chunk_boundaries_unit_lengths( Lmeter_large, [ dd.min(), dd.max()],ov =0, iter_flag= False)
        return stencil_iter[0, first_stencil]

    else:
        #first_stencil = next((i for i, j in enumerate(var_list[:,1] > 0) if j), None)# - 1
        print('no sufficient photon density found. return short stencil')
        # first_stencil= len(var_list[:,1]) -1
        # stencil_iter  = spec.create_chunk_boundaries_unit_lengths( Lmeter_large, [ dd.min(), dd.max()],ov =0, iter_flag= False)
        # return stencil_iter[0, first_stencil]

        # #print(first_stencil)
        return var_list[-1,0]#[-1,0]


with futures.ProcessPoolExecutor(max_workers=Nworkers_process) as executor:
    A = list( executor.map(derive_axis_and_boundaries, all_beams)  )

# %%
B2          = dict()
dist_list   = np.array([np.nan, np.nan])
for I in A:
    k         = I[0]
    B2[k]     = I[1]
    #B2[k]['dist'] = B2[k]['x']
    dist_list = np.vstack([dist_list,I[2] ])

del A
track_dist_bounds     = [ np.nanmin(dist_list[:, 0], 0) , np.nanmax(dist_list[:, 1], 0) ]

# for testing:
#ts_s = np.copy(track_dist_bounds)

# %%
xscale= 1e3
F= M.figure_axis_xy(5, 3, view_scale= 0.6)
for k,I in B2.items():
    plt.plot( I['x']/xscale  , I['across_track_distance']/xscale , '.' , markersize = 0.3)
    #plt.xlim(3e6, 3.25e6)


for k in high_beams:

    Ii = B2[k].iloc[0]
    plt.text(Ii.x/xscale+ 5, Ii.across_track_distance/xscale , str(Ii[[ 'lats', 'lons'] ]).split('Name')[0] )

    Ii = B2[k].iloc[-1]
    plt.text(Ii.x/xscale+ 5, Ii.across_track_distance/xscale , str(Ii[[ 'lats', 'lons'] ]).split('Name')[0], ha ='right' )

F.ax.axvline(track_dist_bounds[0]/xscale, color='gray', zorder= 2)
F.ax.axvline(track_dist_bounds[1]/xscale, color='gray', zorder= 2)
F.ax.axhline(0, color='gray', zorder= 2)

plt.title('B01 filter and regrid | ' + track_name +'\npoleward '+str(track_poleward)+' \n \n', loc='left')
plt.xlabel('along track distance (km)')
plt.ylabel('across track distance (km)')

F.save_light(path= plot_path +'../', name='B01_ALT03_'+track_name+'_tracks_all')

# %%
# for testing
#track_dist_bounds[1]  = track_dist_bounds[0] + (track_dist_bounds[1] - track_dist_bounds[0])/20
#track_dist_bounds = ts_s[0] ,ts_s[0] + (ts_s[1] - ts_s[0]) /6

# %%
#imp.reload(regrid)
##### 2.) regridding and averaging
print('regrid')
def regridding_wrapper(I):
    key, Ti = I
    print(key, Ti.shape,2* Ti.shape[0]/Lmeter)
    stencil_iter = create_chunk_boundaries_unit_lengths( Lmeter, track_dist_bounds, iter_flag=True )
    with futures.ThreadPoolExecutor(max_workers= Nworkers) as executor_sub:
        Bi = regrid.get_stencil_stats( Ti, stencil_iter, 'heights_c', 'x' , stancil_width= Lmeter/2, Nphoton_min=Nphoton_min, map_func = executor_sub.map)
    #B3[key] =
    print(key, 'done')
    return key, Bi


with futures.ProcessPoolExecutor(max_workers=Nworkers_process) as executor:
    B3 = dict( executor.map(regridding_wrapper, B2.items() )  )

# %% ---define start and end position and same in Json file

#I = B3['gt2l'].copy()
D_info = dict()
for k,I in B3.items():

    # reset x coordinate
    I['median_dist']   = I['median_x'] - track_dist_bounds[0] #- Lmeter/2
    I['dist']          = I['x']        - track_dist_bounds[0] #- Lmeter/2
    #I['index']      = I['x']
    # rename y coordinate
    I = I.rename(columns={'across_track_distance': 'y'})

    # find starting and end position
    Di_s  = dict(I[I['seg_ID'] == I['seg_ID'].iloc[0] ].mean()[['lons', 'lats', 'seg_ID', 'delta_time']])
    Di_s['across_track_distance_0'] =track_dist_bounds[0]

    Di_e  = dict(I[I['seg_ID'] == I['seg_ID'].iloc[-1] ].mean()[['lons', 'lats', 'seg_ID', 'delta_time']])
    Di_e['across_track_distance_0'] =track_dist_bounds[0]

    D_info[k] = {'start':Di_s,  'end':Di_e , 'poleward': str(track_poleward) }

    # reorder indexes
    column_names = ['index', 'x', 'y', 'median_x', 'lons', 'lats' ,'heights_c_weighted_mean', 'heights_c_median', 'heights_c_std',  'N_photos', ]
    vars_ad = set(list(I[I['seg_ID'] == I['seg_ID'].iloc[0] ].mean().index)) - set(column_names)
    I = I.reindex(columns=column_names  + list(vars_ad))

    B3[k] = I


# save Json
MT.json_save(track_name + '_B01_stats',save_path, D_info, verbose= True )

# %% saving data
io.save_pandas_table(B2, track_name + '_B01_regridded'  , save_path) # all photos but heights adjusted and with distance coordinate
io.save_pandas_table(B3, track_name + '_B01_binned'     , save_path) # regridding heights


# %% plotting just for checking
key         = 'gt2r'
if plot_flag:
    MT.mkdirs_r(plot_path)
    Ti2 = B3[key]
    T2  = B2[key]

    dl = 4000
    latlims = (Ti2['x'].iloc[0] , Ti2['x'].iloc[-1] )
    #chunk_list = np.arange(latlims[0],latlims[1], dl )
    #chunk_list = sample(  list(np.arange(latlims[0],latlims[1],dl )[0:80])  ,10)
    chunk_list = np.arange(latlims[0],latlims[1], dl )[::10]
    chunk_list = np.append( chunk_list, latlims[1]- dl-1)
    for ll in chunk_list:
        F = M.figure_axis_xy(7, 3, view_scale=0.8)

        plt.plot( T2['x'], T2['heights_c'],   'k.',  markersize= 0.5, alpha =0.8 )
        #plt.plot( ALT07['ref']['latitude'] , ALT07['heights']['height_segment_height'] , 'r.', markersize=0.8, alpha = 1, label ='ALT07 seg. heights')

        plt.plot(Ti2['x'], Ti2['heights_c_weighted_mean'] +1, '.-', color='darkblue', linewidth=0.5, markersize=2,alpha=0.9, label='x-gauss weighted mean +1')
        plt.plot(Ti2['x'], Ti2['heights_c_median'], 'r.-',  linewidth=0.5, markersize=2,alpha=0.9, label='median')

        #plt.plot(Ti2['x'], Ti2['heights_c_mode']-1, 'g.-',  linewidth=0.5, markersize=2,alpha=0.9, label='mode - 1')

        plt.plot(Ti2['x'], Ti2['heights_c_std'] - 1.8, 'k-', linewidth=0.5,alpha=1)
        plt.fill_between(  Ti2['x'], Ti2['heights_c_std'] -1.8 , y2=-1.8, color='gray',alpha=1)

        # plt.plot( ALT03['delta_time'], ALT03['heights_c'],   'k.',  markersize= 0.3, alpha =0.2 )
        # plt.plot(ALT07['time']['delta_time'] , ALT07['heights']['height_segment_height'] , 'r.', markersize=0.8, alpha = 1, label ='ALT07 seg. heights')
        plt.legend(loc=1)
        plt.xlim(ll, ll+dl)
        plt.ylim(-4, 4)

        plt.xlabel('Meters from the Sea Ice Edge')
        plt.ylabel('Height Anomalie (meters)')
        F.ax.axhline(y =-1.8, color='black', linewidth=0.5)
        F.save_light(path= plot_path, name='ALT03_filt_compare'+ str(ll))
