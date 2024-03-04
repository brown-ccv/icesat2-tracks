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



#print(track_name, batch_key, test_flag)
hemis, batch = batch_key.split('_')
#track_name= '20190605061807_10380310_004_01'
ATlevel= 'ATL03'

load_path   = mconfig['paths']['scratch'] +'/'+ batch_key +'/'
load_file   = load_path + 'processed_'+ATlevel+'_'+track_name+'.h5'

save_path  = mconfig['paths']['work'] +'/B01_regrid_'+hemis+'/'

plot_path = mconfig['paths']['plot']+ '/'+hemis+'/'+batch_key+'/'+track_name +'/B_ov/'
bad_track_path =mconfig['paths']['work'] +'bad_tracks/'+ batch_key+'/'
MT.mkdirs_r(save_path)

# set pars

# define parameters:
Lmeter      = 20 # stencil length in units of 'dist'; likely in meters the resulting resolution is L/2
Nphoton_min = 5 # mininum need photons per stancil to return results

Lmeter_large= 100e3 # stancil width for testing photon density. stancils do not overlab.
minium_photon_density = 0.05 # minimum photon density per meter in Lmeter_large chunk to be counted as real signal

plot_flag   = True
Nworkers    = 3         # number of threads for parallel processing
Nworkers_process    =3         # number of threads for parallel processing
# %%
# test which beams exist:
all_beams   = mconfig['beams']['all_beams']
# high_beams  = mconfig['beams']['high_beams']
# low_beams   = mconfig['beams']['low_beams']

f         = h5py.File(load_file, 'r')
beams     = [b if b in f.keys() else None for b in all_beams]
accent    = regrid.track_type_beam(f)

print('accent is ' , accent)
# Load fata and apply height corrections
# This needs version 2 of the ALT 03 dataset


hist    = 'Beam stats'
B       = dict()
B1save  = dict()
for k in beams:
    #k = beams[0]
    #imp.reload(io)
    print(k)
    T = io.getATL03_beam(load_file, beam= k)

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
    print('corrected')

    ### cut data at the rear that has too much variance
    # cut last segments of data until variance is similar
    rear_mask = np.array(Tsel2.index) > -1
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

        if var_list[0:3].mean()*10 < var_list[-1]:
            #print('cut last '+ str(100/N_seg) +'% of data')
            rear_mask[int(nsize* (N_seg-1) / N_seg):] = False
        else:
            cut_flag =  False


    Tsel2['process_mask'] = rear_mask
    B1save[k]             = Tsel2
    B[k]                  = Tsel2[rear_mask].drop(columns='process_mask')

    ho      = MT.add_line_var(ho, 'cutted ', str( np.round( 100 *(rear_mask.size -rear_mask.sum() ) / rear_mask.size, 0)) + '% in the back of the track' )
    ho      = MT.add_line_var(ho, 'selected size', str(Tsel.shape[0]))
    ho      = MT.add_line_var(ho, 'final size ', str(Tsel_c.shape[0]))
    print(ho)

    hist    = MT.write_log(hist, ho)

print('done with 1st loop')

# %%
# define latitude limits
lat_lims = regrid.lat_min_max(B, all_beams, accent = accent)

# %% test ave photon density
length_meter = (abs(lat_lims[1])  - abs(lat_lims[0])) * 110e3
p_densities_r = list()
p_densities_l = list()

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




# %%
##### 1.) derive common axis for beams and filter out low density area at the beginning
print('derive common axis for beams and filter out low density area at the beginning')


def derive_axis_and_boundaries(key):
    #key = 'gt3r'
    print(key)
    T2         = regrid.derive_axis(B[key], lat_lims).sort_values('dist')
    T2.shape
    x0         = get_better_lower_boundary(Lmeter_large, np.array(T2['dist']))
    print( 'cut off ' , 100 * (1 - T2[T2['dist'] > x0].shape[0]/T2.shape[0])  , '% off all data points at the beginning' )
    T2         = T2[T2['dist'] >x0] # cut off low density start
    #B2[key]    = T2
    return key, T2, [T2['dist'].min(), T2['dist'].max()]

def get_better_lower_boundary(Lmeter_large, dd):

    # T2         = regrid.derive_axis(B[key], lat_lims)
    # dd = np.array(T2['dist'])
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
        first_stencil = next((i for i, j in enumerate(var_list[:,1] > minium_photon_density) if j), None) - 1
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



# for bb in all_beams:
#     A = derive_axis_and_boundaries(bb)

with futures.ProcessPoolExecutor(max_workers=Nworkers_process) as executor:
    A = list( executor.map(derive_axis_and_boundaries, all_beams)  )
# %%
B2          = dict()
dist_list   = np.array([np.nan, np.nan])
for I in A:
    k         = I[0]
    B2[k]     = I[1]
    dist_list = np.vstack([dist_list,I[2] ])

del A

track_dist_bounds     = [ np.nanmin(dist_list[:, 0], 0) , np.nanmax(dist_list[:, 1], 0) ]
# for testing:
#track_dist_bounds[1]  = track_dist_bounds[0] + (track_dist_bounds[1] - track_dist_bounds[0])/20


# %%
##### 2.) regridding and averaging
print('regrid')
def regridding_wrapper(I):
    key, Ti = I
    print(key, Ti.shape, 2* Ti.shape[0]/Lmeter)
    stencil_iter = create_chunk_boundaries_unit_lengths( Lmeter, track_dist_bounds, iter_flag=True )
    with futures.ThreadPoolExecutor(max_workers= Nworkers) as executor_sub:
        Bi = regrid.get_stencil_stats( Ti, stencil_iter, 'heights_c', stancil_width= Lmeter/2, Nphoton_min=Nphoton_min, map_func = executor_sub.map)
    #B3[key] =
    print(key, 'done')
    return key, Bi


with futures.ProcessPoolExecutor(max_workers=Nworkers_process) as executor:
    B3 = dict( executor.map(regridding_wrapper, B2.items() )  )


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
    latlims = (Ti2['dist'].iloc[0] , Ti2['dist'].iloc[-1] )
    #chunk_list = np.arange(latlims[0],latlims[1], dl )
    #chunk_list = sample(  list(np.arange(latlims[0],latlims[1],dl )[0:80])  ,10)
    chunk_list = np.arange(latlims[0],latlims[1], dl )[::10]
    chunk_list = np.append( chunk_list, latlims[1]- dl-1)
    for ll in chunk_list:
        F = M.figure_axis_xy(7, 3, view_scale=0.8)

        plt.plot( T2['dist'], T2['heights_c'],   'k.',  markersize= 0.5, alpha =0.8 )
        #plt.plot( ALT07['ref']['latitude'] , ALT07['heights']['height_segment_height'] , 'r.', markersize=0.8, alpha = 1, label ='ALT07 seg. heights')

        plt.plot(Ti2['dist'], Ti2['heights_c_weighted_mean'] +1, '.-', color='darkblue', linewidth=0.5, markersize=2,alpha=0.9, label='x-gauss weighted mean +1')
        plt.plot(Ti2['dist'], Ti2['heights_c_median'], 'r.-',  linewidth=0.5, markersize=2,alpha=0.9, label='median')

        #plt.plot(Ti2['dist'], Ti2['heights_c_mode']-1, 'g.-',  linewidth=0.5, markersize=2,alpha=0.9, label='mode - 1')

        plt.plot(Ti2['dist'], Ti2['heights_c_std'] - 1.8, 'k-', linewidth=0.5,alpha=1)
        plt.fill_between(  Ti2['dist'], Ti2['heights_c_std'] -1.8 , y2=-1.8, color='gray',alpha=1)

        # plt.plot( ALT03['delta_time'], ALT03['heights_c'],   'k.',  markersize= 0.3, alpha =0.2 )
        # plt.plot(ALT07['time']['delta_time'] , ALT07['heights']['height_segment_height'] , 'r.', markersize=0.8, alpha = 1, label ='ALT07 seg. heights')
        plt.legend(loc=1)
        plt.xlim(ll, ll+dl)
        plt.ylim(-6, 6)

        plt.xlabel('Meters from the Sea Ice Edge')
        plt.ylabel('Height Anomalie (meters)')
        F.ax.axhline(y =-1.8, color='black', linewidth=0.5)
        F.save_light(path= plot_path, name='ALT03_filt_compare'+ str(ll))
