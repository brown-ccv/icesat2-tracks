import os, sys
#execfile(os.environ['PYTHONSTARTUP'])

"""
This file open a ICEsat2 track applied filters and corections and returns smoothed photon heights on a regular grid in an .nc file.
This is python 3
"""
# exec(open(os.environ['PYTHONSTARTUP']).read())
# exec(open(STARTUP_2019_DP).read())

base_path='/Users/Shared/Projects/2021_IceSAT2_tracks/'
sys.path.append(base_path +'modules/')
sys.path.append(base_path +'modules/ICEsat2_SI_tools/')

import matplotlib.pyplot as plt


#import m_general as M
#import m_tools as MT
import numpy as np

import m_general_ph3 as M
import m_tools_ph3 as MT

import netCDF4
import datetime
import os
from netCDF4 import Dataset
import xarray as xr
import pandas as pd
import h5py

import ICEsat2_SI_tools.convert_GPS_time as cGPS
import ICEsat2_SI_tools.io as io
from spectral_estimates import create_chunk_boundaries_unit_lengths
from random import sample
import imp

#import s3fs

# Python reader based on Pandas. Other reader examples available in readers.py
track_name= 'ATL03_20190515060436_07170312_002_02'
load_path   = base_path + 'data/data1/'
load_file   = load_path + 'processed_'+track_name+'.h5'

track_name= 'ATL03_20191215230028_12220512_004_01'
track_name= 'ATL03_20210414065545_03121112_004_01'
load_path   = base_path + 'data/data4/'
load_file   = load_path + 'processed_'+track_name+'.h5'
# %%
track_name= 'ATL03_20200520190502_08440701_003_01'
load_path   = base_path + 'modules/read-ICESat-2/scripts/2020.05.20/'
load_file   = load_path +track_name+'.h5'

# %%

# test which beams exist:
all_beams = ['gt1l', 'gt1r', 'gt2l', 'gt2r', 'gt3l', 'gt3r']
low_beams = ['gt1l',  'gt2l',  'gt3l']
high_beams = ['gt1r',  'gt2r',  'gt3r']

f         = h5py.File(load_file, 'r')
beams     = [b if b in f.keys() else None for b in all_beams]

def correct_heights(T03, T03c, coord = 'delta_time'):
    T03['heights_c']= T03['heights'] -  np.interp( T03[coord],T03c[coord], T03c['dem_h'] )
    return T03

# Load fata and apply height corrections
# This needs version 2 of the ALT 03 dataset
hist= 'Beam stats'
B= dict()
for k in beams:

    T = io.getATL03_beam(load_file, beam= k)

    ho = k
    ho = MT.add_line_var(ho, 'size', str(T.shape[0]))
    ho = MT.add_line_var(ho, 'by confidence levels:' + str(np.arange(0, 5)), [ (T['signal_confidence'] == i).sum() for i in np.arange(0, 5) ])

    # filter:
    Tsel    = T[(T['signal_confidence']>2) & (T['heights']<100)  & (T['heights'] > -100) ]# & (T['delta_time']>5) & (T['delta_time']<24) ]
    if len(Tsel) == 0:
        ho  = MT.add_line_var(ho, 'no photons found', '')
        Tsel= T[(T['signal_confidence'] ==-1 ) & (T['heights']<100)  & (T['heights'] > -100) ]# & (T['delta_time']>5) & (T['delta_time']<24) ]


    Tsel_c  = io.getATL03_height_correction(load_file)
    Tsel_c  = Tsel_c[Tsel_c['dem_h'] < 1e5] # cute out weird references
    B[k]    = correct_heights(Tsel, Tsel_c)

    ho      = MT.add_line_var(ho, 'selected size', str(Tsel.shape[0]))
    ho      = MT.add_line_var(ho, 'final size ', str(Tsel_c.shape[0]))

    print(ho)
    hist    = MT.write_log(hist, ho)


# %%
plt.plot( B[k].lats , B[k].heights_c, '.')

# %%

def track_type(T):
    """
    Returns if track acending or desending
    T is a pandas table
    """
    #T = B[k]
    #T = B[beams_list[0]]
    return (T['lats'].iloc[T['delta_time'].argmax()] - T['lats'].iloc[T['delta_time'].argmin()] ) < 0

def lat_min_max(B, beams_list):
    """
    defines common boundaries for beams_list in B
    iunputs:
    beams_list list of concidered beams
    B is dict of Pandas tables with beams

    returns:
    min_lat, max_lat, accent   min and max latitudes of the beams, (True/False) True if the track is accending
    """
    #B, beams_list = B , high_beams
    accent = track_type( B[beams_list[0]] )

    if B[beams_list[0]]['lats'].iloc[0] < 0:
        hemis = 'SH'
    else:
        hemis = 'NH'

    track_lat_mins, track_lat_maxs= list(), list()
    for k in beams_list:
        track_lat_mins.append( B[k]['lats'].min() )
        track_lat_maxs.append( B[k]['lats'].max() )

    if hemis == 'SH':
        return max(track_lat_maxs) , min(track_lat_mins), accent
    else:
        return min(track_lat_mins), max(track_lat_maxs), accent


# %%

def derive_axis(TT, lat_lims = None):
    """
    returns TT distance along track 'dist' in meters
    input:
    TT pandas table with ICEsat2 track data
    lat_lims (None) tuple with the global latitude limits used to define local coodinate system
    returns:
    TT with x,y,dist and order by dist
    """
    #TT, lat_lims = B[key], lat_lims_high
    # derive distances in meters
    r_e= 6.3710E+6
    dy= r_e*2*np.pi/360.0
    #deglon_in_m= np.cos(T2['lats']*np.pi/180.0)*dy

    # either use position of the 1st photon or use defined start latitude
    if lat_lims is None:
        TT['y']=(TT['lats'].max() - TT['lats']) *dy
    else:
        TT['y']=(lat_lims[0] - TT['lats']) *dy

    #TT['y']     =   (TT['lats']) *dy


    if (lat_lims[2] == True):
        # accending track
        lon_min = TT['lons'].max()
    else:
        # decending track
        lon_min = TT['lons'].min()

    #print(lon_min)
    TT['x']     = (TT['lons'] - lon_min) * np.cos( TT['lats']*np.pi/180.0 ) * dy
    #TT['x']     = (TT['lons'] ) * np.cos( TT['lats']*np.pi/180.0 ) * dy
    TT['dist']  =   np.sqrt(TT['x']**2 + TT['y']**2)

    # set 1st dist to 0, not used if global limits are used
    if lat_lims is None:
        TT['dist']= TT['dist']- TT['dist'].min()
    else:
        TT['dist']= TT['dist']#- lat_lims[0]

    TT=TT.sort_values(by='dist')
    return TT

def reduce_to_height_distance(TT, key, dx=1, lat_lims = None):
    """
    interpolates key (photos heights) to regular grid using 'dist' in pandas table TT.
    dx          is the interpolation interval
    lat_lims    (None) tuple with the global latitude limits used to define local coodinate system
                if None 'dist' min and max are used

    returns:
    x1, y1     position, height
    """
    from scipy.interpolate import interp1d
    if type(dx) is np.ndarray:
        x1 = dx
    else:
        x1 = np.arange(0,TT['dist'].max(), dx)
    y1 = np.interp(x1, TT['dist'], TT[key] )

    return x1, y1

# this is not need anymore
def poly_correct(x, y, poly_order=7, plot_flag=False):

    """
    subtracts a fitted polynom to y
    inputs:
    x,y     position, height
    poly_order  order of polynom
    plot_flag   if true plots the fit
    returns
    y'      y - polynom fit
    """
    z = np.polyfit(x , y , poly_order)
    p = np.poly1d(z)
    if plot_flag:
        plt.plot(x,y, '.',  markersize=0.2,)
        plt.plot(x, p(x), '-',  markersize=0.2,)
    #return z
    return y - p(x)


# define latitude limits
lat_lims_high = lat_min_max(B, high_beams)
lat_lims_low = lat_min_max(B, low_beams)

##### 1.) derive common axis for beams

# %% 1st all stong beams
B2=dict()
colors = iter(['red','blue','orange','green','black','yellow'])
dist_list =np.array([np.nan, np.nan])
for key in high_beams:
    T2         = derive_axis(B[key], lat_lims_high)

    # the ends of the tracks are about 600m appart.. i don't know why ..
    # this should have no influence on the distrance along the track
    #plt.plot( list(T2['x'][-200:]), list(T2['y'][-200:]), c = next(colors))
    #plt.plot( list(T2['x'][15000:15005]), list(T2['y'][15000:15005]), '.', c = next(colors))
    #plt.plot( list(T2['lons'][:2]), list(T2['lats'][:2]), c = next(colors))
    #plt.axis('equal')

    # depreciated
    #T2['heights_adjusted']  = poly_correct(T2['dist'], T2['heights']) # 'dist' is length from start position

    # rolling 100 photon window. May have to be replaced by a window of fixed length.
    # if rollin window is replaced by fixed lengrh the following step is obsolute.
    #T2['heights_smth']      = T2['heights_adjusted'].rolling(100).median()
    B2[key]    = T2
    dist_list  = np.vstack([ dist_list, [T2['dist'].min(), T2['dist'].max()] ])


##### 2.) regridding and averaging
# %% define functions

def get_mode(y, bins = np.arange(-5,5,  0.1)):
    "returns modes of histogram of y defined by bins"
    hist, xbin = np.histogram(y, bins = bins )
    return xbin[hist.argmax()]

def weighted_mean(x_rel, y):
    "returns the gaussian weighted mean for stencil"

    def weight_fnk(x):
        "returns gaussian weight given the distance to the center x"
        return np.exp(- (x/.5)**2 )

    w = weight_fnk(x_rel)
    return (w*y).sum()/w.sum()


# this function is applied to beam:
def get_stencil_stats(T2, stencil_iter,  key , Nphoton_min = 5):

    """
    T2              pd.DAtaframe with beam data needs at least 'dist' and key
    stencil_iter    iterable that constains the stancil boundaries and center [left boundary, center, right boundary]
    key             coloumn index used in T2
    Nphoton_min     minimum required photots needed to return meaning full averages

    returns:
    pandas DataFrame with the same as T2 but not taken the median of each column
    the following columns are also added:
    key+ '_weighted_mean'   x-weighted gaussian mean of key for each stencil
    key+ '_mode'            mode of key for each stencil
    'N_photos'              Number of Photons for each stencil
    key+ '_std'             standard deviation for each stencil

    the column 'key' is rename to key+'_median'

    """

    x_data = T2['dist']
    y_data = T2[key]
    # apply this funcion to each stancil
    def calc_stencil_stats(istencil):

        "returns stats per stencil"

        i_mask=(x_data >= istencil[0])  & (x_data < istencil[2])
        Nphoton = i_mask.sum()

        if Nphoton < Nphoton_min:

            Tmedian = T2[i_mask].median()

            Tmedian[key+ '_weighted_mean']  = np.nan
            #Tmedian[key+ '_mode']           = np.nan
            Tmedian['N_photos']             = i_mask.sum()
            Tmedian[key+ '_std']            = np.nan

            return istencil[1], Tmedian


        Tmedian = T2[i_mask].median()

        x_rel   = (x_data[i_mask] - istencil[1])/(L/2)
        y       = y_data[i_mask]

        Tmedian[key+ '_weighted_mean']      = weighted_mean(x_rel, y)
        #Tmedian[key+ '_mode']               = get_mode(y)
        Tmedian['N_photos']                 = i_mask.sum()
        Tmedian[key+ '_std']                = y.std()

        return istencil[1], Tmedian

    # apply func to all stancils
    D_filt = dict(map(calc_stencil_stats,stencil_iter))

    DF_filt = pd.DataFrame.from_dict(D_filt, orient='index')
    DF_filt = DF_filt.rename(columns={key: key+'_median', 'dist': 'median_dist'})
    DF_filt['dist'] = DF_filt.index
    DF_filt = DF_filt.reset_index()

    return DF_filt


# %% old version
# define common dist_grid:
#dx= 5 # 2 * resolution in meters, datapoint +-dx are used to take the mean
#dist_grid = np.arange( np.nanmin(dist_list[:, 0], 0) , np.nanmax(dist_list[:, 1], 0), dx )

# derive bin means
def bin_means(T2, dist_grid):
    dF_mean = pd.DataFrame(index =T2.columns)
    ilim    = int(len(dist_grid))
    N_i     = list()

    for i in np.arange(1,ilim-1, 1):
        if i % 5000 ==0:
            print(i)
        i_mask=(T2['dist'] >= dist_grid[i-1])  & (T2['dist'] < dist_grid[i+1])
        #if ( (T2['dist'] >= dist_grid[i-1])  & (T2['dist'] < dist_grid[i+1]) ).sum() > 0:
        dF_mean[i] = T2[i_mask].mean()
        #dF_median[i] = T2[i_mask].median()
        N_i.append(i_mask.sum())

    dF_mean             = dF_mean.T
    dF_mean['N_photos'] = N_i
    dF_mean['dist'] = dist_grid[np.arange(1,ilim-1, 1)]

    return dF_mean

# %% estimating wave length for given period
T = 5
g= 9.81
lam = g *T**2 / (2 * np.pi)
print(lam)

# define parameters:
L = 20 # stencil length in units of 'dist'; likely in meters the resulting resolution is L/2
Nphoton_min = 5 # mininum need photons per stancil to return results

G=dict()
B3 = dict()
for key,Ti in B2.items():

    print(key)
    stencil_iter = create_chunk_boundaries_unit_lengths( L, [ np.nanmin(dist_list[:, 0], 0) , np.nanmax(dist_list[:, 1], 0) ], iter_flag=True )

    Ti2 = get_stencil_stats(Ti, stencil_iter, 'heights_c', Nphoton_min=Nphoton_min)
    Ti2['heights_c_median'][ np.isnan(Ti2['heights_c_std']) ]= np.nan # replace median calculation with nans

    B3[key] =Ti2 # store in dict

    # % save relevant data as xarray
    # G[key] =xr.DataArray(Ti2['heights_c_median'], coords={'dist': Ti2['dist']}, dims='dist', name=key)
    print(key, 'done')


# %% saving data
# Gnew = xr.merge(G.values())
# Gnew.to_netcdf(load_path+'/'+track_name +'_filtered_photon_heights.nc')

io.save_pandas_table(B2, track_name + '_B01_corrected' , load_path) # all photos but heights adjusted and with distance coordinate
io.save_pandas_table(B3, track_name + '_B01_binned' , load_path) # regridding heights

# %% plotting just for checking
plot_flag = True
if plot_flag:

    import m_tools_ph3 as MT
    plot_path = base_path+'plots/B01_regridding/'+track_name+'/'
    MT.mkdirs_r(plot_path)


    Ti2 = B3[key]
    T2  = B2[key]

    dl = 4000
    latlims = (Ti2['dist'].iloc[0] , Ti2['dist'].iloc[-1] )
    for ll in sample(  list(np.arange(latlims[0],latlims[1],dl )[0:80])  ,10):
        F = M.figure_axis_xy(7, 3, view_scale=0.8)

        plt.plot( T2['dist'], T2['heights_c'],   'k.',  markersize= 0.5, alpha =0.8 )
        #plt.plot( ALT07['ref']['latitude'] , ALT07['heights']['height_segment_height'] , 'r.', markersize=0.8, alpha = 1, label ='ALT07 seg. heights')

        plt.plot(Ti2['dist'], Ti2['heights_c_weighted_mean'] +1, '.-', color='darkblue', linewidth=0.5, markersize=2,alpha=0.9, label='x-gauss weighted mean +1')
        plt.plot(Ti2['dist'], Ti2['heights_c_median'], 'r.-',  linewidth=0.5, markersize=2,alpha=0.9, label='median')

        plt.plot(Ti2['dist'], Ti2['heights_c_mode']-1, 'g.-',  linewidth=0.5, markersize=2,alpha=0.9, label='mode - 1')

        plt.plot(Ti2['dist'], Ti2['heights_c_std'] - 1.8, 'k-', linewidth=0.5,alpha=1)
        plt.fill_between(  Ti2['dist'], Ti2['heights_c_std'] -1.8 , y2=-1.8, color='gray',alpha=1)

        # plt.plot( ALT03['delta_time'], ALT03['heights_c'],   'k.',  markersize= 0.3, alpha =0.2 )
        # plt.plot(ALT07['time']['delta_time'] , ALT07['heights']['height_segment_height'] , 'r.', markersize=0.8, alpha = 1, label ='ALT07 seg. heights')
        plt.legend(loc=1)
        plt.xlim(ll, ll+dl)
        plt.ylim(-2, 3)

        plt.xlabel('Meters from the Sea Ice Edge')
        plt.ylabel('Height Anomalie (meters)')
        F.ax.axhline(y =-1.8, color='black', linewidth=0.5)
        F.save_light(path= plot_path, name='ALT03_filt_compare'+ str(ll))


# %% plot
# plot_flag = True
# if plot_flag:
#     F = M.figure_axis_xy(5, 6.5, view_scale=0.8)
#
#     plt.subplot(2,1 , 1)
#     plt.title('Filtered Photon Heights in Sea Ice')
#     plt.plot(T2['dist'], T2['heights_c'], 'ko',  markersize=1,alpha=0.5)
#
#     #plt.plot(T2['dist'], T2['heights_smth'], 'bo',  markersize=1, alpha=0.5)
#
#
#     plt.plot(x1, y1, '-r',  markersize=2)
#     #plt.xlim(70000, 80000)
#     plt.xlim(140000, 145000)
#     plt.xlabel('Meters from the Sea Ice Edge')
#     plt.ylabel('Height Anomalie (meters)')
#
#     plt.show()
#
#
#     F = M.figure_axis_xy(4.5, 3, view_scale=0.8)
#
#     #plt.subplot(2,1, 2)
#     plt.plot(T2['dist']/1e3, T2['heights_adjusted'], 'ko',  markersize=1,alpha=0.5, label='rar data')
#
#     #plt.plot(T3['dist']/1e3, T3['heights_smth'], 'bo',  markersize=1, alpha=0.5)
# # %%
#     plt.plot(x1/1e3, y1, '-r',  linewidth=1, label='best estimate')
#     #plt.xlim(2*3400/1e3, 2*7000/1e3)
#     #plt.xlim(10000, 10500)
#     plt.xlabel('km from the Sea Ice Edge')
#     plt.ylabel('Height Anomalie (meters)')
#     plt.legend()
#     # F.save_pup(path=bpath, name='filtered_photon_heights_pup')
#     # F.save_light(path=bpath, name='filtered_photon_heights_pup_lght')
#     #plt.ylim(5, 12)
