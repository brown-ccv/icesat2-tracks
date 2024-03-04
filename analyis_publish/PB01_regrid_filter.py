# %%
import os, sys
#execfile(os.environ['PYTHONSTARTUP'])

"""
This file open a ICEsat2 track applied filters and corections and returns smoothed photon heights on a regular grid in an .nc file.
This is python 3
"""

exec(open(os.environ['PYTHONSTARTUP']).read())
exec(open(STARTUP_2021_IceSAT2).read())

#%matplotlib inline

import ICEsat2_SI_tools.convert_GPS_time as cGPS
import h5py
import ICEsat2_SI_tools.io as io
import ICEsat2_SI_tools.spectral_estimates as spec

import time
import imp
import copy
import spicke_remover
import datetime
import generalized_FT as gFT
from scipy.ndimage.measurements import label

fig_sizes = mconfig['fig_sizes']['Cryosphere']


#import s3fs
# %%
ID_name, batch_key, test_flag = io.init_from_input(sys.argv) # loads standard experiment
#ID_name, batch_key, test_flag = '20190605061807_10380310_004_01', 'SH_batch01', False
#ID_name, batch_key, test_flag = '20190601094826_09790312_004_01', 'SH_batch01', False
#ID_name, batch_key, test_flag = '20190207111114_06260210_004_01', 'SH_batch02', False
#ID_name, batch_key, test_flag = '20190208152826_06440210_004_01', 'SH_batch01', False
#ID_name, batch_key, test_flag = '20190213133330_07190212_004_01', 'SH_batch02', False
#ID_name, batch_key, test_flag = '20190210143705_06740210_004_01', 'SH_batch02', False

#ID_name, batch_key, test_flag = '20190215184558_07530210_004_01', 'SH_batch02', False

# local best case:
#ID_name, batch_key, test_flag = '20190219073735_08070210_004_01', 'SH_batch02', False


# ID_name, batch_key, test_flag = 'SH_20190219_08070212', 'SH_publish', True
# ID_name, batch_key, test_flag = 'SH_20190502_05160312', 'SH_publish', True
# ID_name, batch_key, test_flag = 'SH_20190502_05180312', 'SH_publish', True

ID_name, batch_key, ID_flag = 'SH_20190224_08800210', 'SH_publish', True

ID, _, hemis, batch = io.init_data(ID_name, batch_key, ID_flag, mconfig['paths']['work'],  )

#print(ID_name, batch_key, test_flag)
#ID_name= '20190605061807_10380310_004_01'
ATlevel= 'ATL03'



load_path_scratch = mconfig['paths']['scratch'] +'/'+ batch_key +'/'
load_path_work    = mconfig['paths']['work'] +'/'+ batch_key +'/'

#load_file   = 'A01c_'+ATlevel+'_'+ID_name
#load_file_str   = load_path + load_file+'.h5'

# load_path       = mconfig['paths']['work'] +'/B01_regrid_'+hemis+'/'
# load_file       = load_path + 'processed_' + ATlevel + '_' + ID_name + '.h5'

plot_path   = mconfig['paths']['plot'] + '/'+hemis+'/'+batch_key+'/publish/' + ID_name + '/'
MT.mkdirs_r(plot_path)
# %%

all_beams   = mconfig['beams']['all_beams']
high_beams  = mconfig['beams']['high_beams']
low_beams   = mconfig['beams']['low_beams']

#B00 = io.getATL03_beam(load_path_scratch +'/'+ID['tracks']['ATL03'][0]+ '.h5')

#B00_hdf5    = h5py.File(load_path_scratch +'/A01c_ATL03_'+ID_name+ '_corrected.h5', 'r')

B0_hdf5    = h5py.File(load_path_scratch +'/A01c_ATL03_'+ID_name+ '_corrected.h5', 'r')
B2_hdf5    = h5py.File(load_path_work +'B01_regrid'+'/'+ID_name + '_B01_regridded.h5', 'r')
B3_hdf5    = h5py.File(load_path_work +'B01_regrid'+'/'+ID_name + '_B01_binned.h5', 'r')

B0, B2, B3 = dict(), dict(), dict()
for b in all_beams:
    B0[b] = io.get_beam_hdf_store(B0_hdf5[b])
    B2[b] = io.get_beam_hdf_store(B2_hdf5[b])
    B3[b] = io.get_beam_hdf_store(B3_hdf5[b])

B0_hdf5.close(), B2_hdf5.close(), B2_hdf5.close()

# B0   = io.load_pandas_table_dict(ID_name + '_B01_corrected'  , load_path)
#B1   = io.load_pandas_table_dict(ID_name + '_B01_new_coords'  , load_path)
#B2   = io.load_pandas_table_dict(ID_name + '_B01_regridded', load_path) # rhis is the rar photon data
#B3   = io.load_pandas_table_dict(ID_name + '_B01_binned' , load_path)  #


load_file_ATL10 =  'processed_' + 'ATL07-02' + '_20190219063727_08070201_005_01.h5'

# %%

B07, B07_c = dict(), dict()
for k in all_beams:
    B07[k] = io.getATL07_beam(load_path_scratch + ID['tracks']['ATL07']+'.h5', beam= k)
    #B07_c[k] = io.getATL07_height_corrections(load_path_scratch + load_file_ATL10, beam= k)


dist_list   = np.array([np.nan, np.nan])
for I in B2.values():
    #B2[k]['dist'] = B2[k]['x']
    dist_list = np.vstack([dist_list, [I['x'].min(), I['x'].max()] ] )

track_dist_bounds     = [ np.nanmin(dist_list[:, 0], 0) , np.nanmax(dist_list[:, 1], 0) ]



# %%
xscale= 1e3
F= M.figure_axis_xy(fig_sizes['two_column'][0], fig_sizes['two_column'][1]* 0.8, view_scale= 0.6)

# for k,I in B2.items():
#     dist=  I['x']        - track_dist_bounds[0]
#     plt.plot( dist/xscale  , I['across_track_distance']/xscale , '.', color= col.black , alpha = 0.7, markersize = 0.3)


ATL07min, ATL07max = list(),list()
for k,I in B2.items():
    dist=  I['x']        - track_dist_bounds[0]
    plt.plot( dist/xscale  , I['across_track_distance']/xscale , '.' , color = col.rels[k] , markersize = 0.1)
    I2= B07[k]
    I2['dist']                   = np.interp(I2['time']['delta_time'], I['delta_time'][::-1] , dist[::-1])
    I2['across_track_distance']  = np.interp(I2['time']['delta_time'], I['delta_time'][::-1] , I['across_track_distance'][::-1])
    plt.plot( I2['dist']/xscale  , I2['across_track_distance']/xscale , '.' , color = 'black' , markersize = 0.2, alpha =0.3)

    ATL07min.append(I2['dist'].min()), ATL07max.append(I2['dist'].max())

F.ax.axvline(np.array(ATL07min).min()/xscale                                            , color='gray', zorder= 2, linewidth = 0.8)
F.ax.axvline(np.array(ATL07max).max()/xscale                                            , color='gray', zorder= 2, linewidth = 0.8)
F.ax.axvspan(np.array(ATL07min).min()/xscale, np.array(ATL07max).max()/xscale, color='gray', zorder= 0, alpha= 0.4)

#TT = Ii[[ 'lats', 'lons'] ]
def pstr(TT):
    return str(np.round(TT['lons'], 3)) + '$^{\circ}$E\n' + str(np.round(TT['lats'], 3)) + '$^{\circ}$N'

#print(pstr(TT))

for k in high_beams:

    Ii = B2[k].iloc[0]
    dist=  track_dist_bounds[1]      - track_dist_bounds[0]
    plt.text(dist/xscale+ 5, Ii.across_track_distance/xscale , pstr(Ii[[ 'lats', 'lons'] ]), ha ='left', va ='center' )

    Ii = B2[k].iloc[-1]
    dist=  Ii.x      - track_dist_bounds[0]
    plt.text(0- 8, Ii.across_track_distance/xscale , pstr(Ii[[ 'lats', 'lons'] ]), ha ='right', va ='center' )

plt.text(np.mean([np.array(ATL07min).min()/xscale,np.array(ATL07max).max()/xscale]) - 8, 1.5 +(B2['gt2l']['across_track_distance'].mean()/xscale) , 'ATL07', ha ='center', va ='center', fontsize=12 )


F.ax.axvline(0/xscale                                            , color='black', zorder= 2, linewidth = 0.8)
F.ax.axvline((track_dist_bounds[1] - track_dist_bounds[0])/xscale , color='gray', zorder= 2, linewidth = 0.8)
F.ax.axhline(0, color='black', zorder= 2, linewidth = 0.8)

plt.text(0-5, 0+0.2 , 'origin', horizontalalignment ='right', zorder= 12 )
plt.plot(0, 0, '.', color = 'black', markersize= 10 , zorder= 12)


plt.xlim(-200, (track_dist_bounds[1] - track_dist_bounds[0])/xscale +50 )
#plt.ylim(-3, 5 )

plt.title('Beams in the along-track coordinate system\n' + io.ID_to_str(ID_name), loc='left')
plt.xlabel('along track distance x (km)')
plt.ylabel('across track distance y (km)')

#F.save_pup(path= plot_path , name='B01_ALT03_'+ID_name+'_regridded_tracks')
F.save_light(path= plot_path, name='B01_ALT03_'+ID_name+'_regridded_tracks')

# %%

key         = 'gt2r'
#lead_color= col.rels[key]
lead_color= col.rels['group2']
ALT07_color= col.rels['aug1']
MT.mkdirs_r(plot_path)
T2         = B2[key].copy()
T2['dist'] = T2['x'] - track_dist_bounds[0]
T3         = B3[key].copy()

T07        = B07[key]

# # %%
T07['dist']  = np.interp(T07['time']['delta_time'], T3['delta_time'] , T3['dist'])

# T07.T
# T3.T
#
#font_for_pres()
#
# plt.plot(T07['time']['delta_time'] , T07['ref']['longitude'], '.b', markersize= 2)
# plt.plot( T3['delta_time'], T3['lons'], '.', color='red', markersize= 6, zorder=0)
#
# # regrid T07 on T03 dist
#
# plt.plot(T2_large['lons'], T2_large['heights_c'], '.', color='gray', markersize= 0.8)
# plt.plot(T3_large['lons'], T3_large['heights_c_weighted_mean'], '.r', markersize= 1)
# plt.plot(T07['ref']['longitude'],T07['heights']['height_segment_height'], '.b', markersize= 0.9)
# plt.xlim(T3_large['lons'].min(), T3_large['lons'].max())
# #plt.ylim(T3_large['lats'].min(), T3_large['lats'].max())
#
#
#

x_key= 'dist'
latlims = (T3['dist'].iloc[0] , T3['dist'].iloc[-1] )
dl = 2500
#chunk_list = np.arange(latlims[0],latlims[1], dl )
#chunk_list = sample(  list(np.arange(latlims[0],latlims[1],dl )[0:80])  ,10)
chunk_list = np.arange( latlims[0], latlims[1], dl )[::1]


#for ll in chunk_list:
font_for_print()
xscale=1e3

# %%
for chunk_list_i in np.arange(chunk_list.size)[0:1]:

    chunk_list_i= 252
    # for chunk_list_i in np.arange(chunk_list_i-5, chunk_list_i+5, 1):
    fn = copy.copy(lstrings)
    F = M.figure_axis_xy(fig_sizes['two_column'][0], fig_sizes['two_column'][1], view_scale=0.8, container =True)

    gs = GridSpec(3,8,  wspace=0.1,  hspace=0.7)#figure=fig,

    ax1 = F.fig.add_subplot(gs[0, :]) #plt.subplot(1, 6, fp)
    #ax1.tick_params(labelbottom=False)
    ll_large = chunk_list[chunk_list_i]+2000
    tt_large = ll_large + 12000

    T2_large  = T2[ (T2['dist'] > ll_large) & (T2['dist'] < tt_large) ]
    T3_large =  T3[ (T3['dist'] > ll_large) & (T3['dist'] < tt_large) ]
    T07_large=  T07[ (T07['dist'] > ll_large) & (T07['dist'] < tt_large) ]


    plt.plot( T2_large[x_key]/xscale, T2_large['heights_c'],   'k.',  markersize= 0.5, alpha =0.8 )
    plt.plot( T3_large[x_key]/xscale, T3_large['heights_c_weighted_mean'] , '.', color=lead_color, linewidth=0.5, markersize=1,alpha=0.9, label='x-gauss weighted mean')


    grad_y_offset = -1.25
    uncertainty_y_offset =-1
    AT07_bool_offset = -0.75
    AT07_cat_offset = 0


    col.colormaps2(5)
    #set(T07_large['heights']['height_segment_type'])

    htype_cmap = [col.orange, col.cascade3, col.cascade2, col.cascade1]
    htype_list =  ['cloud_covered','other', 'specular_lead_low_w_bkg', 'specular_lead_low','specular_lead_high_w_bkg', 'specular_lead_high', 'dark_lead_smooth_w_bkg', 'dark_lead_smooth', 'dark_lead_rough_w_bkg' ,'dark_lead_rough', 'off_pointing']

    htype_list= ['cloud_covered','other', 'specular_lead', 'dark_lead' , 'other']
    htype_list
    for htype, hcolor, htype_str in zip( [0, 1, (2, 5), (6, 9)] , htype_cmap , htype_list ):

        if type(htype) is tuple:
            imask = (T07_large['heights']['height_segment_type'] >= htype[0]) & (T07_large['heights']['height_segment_type'] <= htype[1])
        else:
            imask = T07_large['heights']['height_segment_type'] == htype
        pdata = T07_large[imask]
        plt.plot( pdata[x_key]/xscale, pdata['heights']['height_segment_height'] + AT07_cat_offset, '.', color =hcolor, markersize=0.8,alpha=0.9, label='ATL07 height | ' +htype_str)

    for htype, hcolor, htype_str, hsize in zip( [0, 1] , [col.gridcolor, col.red] , ['sea ice', 'ssh'] , [1, 5]):

        pdata = T07_large[T07_large['heights']['height_segment_ssh_flag'] == htype]
        plt.plot( pdata[x_key]/xscale, pdata['heights']['height_segment_height']*0+AT07_bool_offset, '.', color =hcolor, markersize=0.8,alpha=0.9, label='ATL07 height | ' +htype_str)

    #plt.xlabel('Meters from the Sea Ice Edge')
    plt.ylabel('Photon height (m)')

    ax2 = F.fig.add_subplot(gs[1:, 0:6]) #plt.subplot(1, 6, fp)
    #ax1.tick_params(labelbottom=False)

    ll = chunk_list[chunk_list_i]+ 4000
    tt = ll+ 3000

    T2_small  = T2[ (T2['dist'] > ll) & (T2['dist'] < tt) ]
    T3_small =  T3[ (T3['dist'] > ll) & (T3['dist'] < tt) ]
    T07_small=  T07[ (T07['dist'] > ll) & (T07['dist'] < tt) ]

    plt.plot( T2_small[x_key]/xscale, T2_small['heights_c'],   'k.',  markersize= 0.5, alpha =0.8 , label='ATL03 photon heights' )
    plt.plot( T3_small[x_key]/xscale, T3_small['heights_c_weighted_mean'] , '.-', color=lead_color, linewidth=0.5, markersize=2,alpha=0.9, label='Gaussian-weighted mean')

    for htype, hcolor, htype_str in zip( [0, 1, (2, 5), (6, 9), -1] , htype_cmap , htype_list ):

        if type(htype) is tuple:
            imask = (T07_small['heights']['height_segment_type'] >= htype[0]) & (T07_small['heights']['height_segment_type'] <= htype[1])
        else:
            imask = T07_small['heights']['height_segment_type'] == htype

        pdata = T07_small[ imask ]
        plt.plot( pdata[x_key]/xscale, pdata['heights']['height_segment_height'] +AT07_cat_offset , '.', color =hcolor, markersize=1.7,alpha=1, label= str(htype) +' '+ htype_str)

    #plt.plot( T07_small[x_key]/xscale, T07_small['heights']['height_segment_height'] +AT07_cat_offset+0.1 , '.', color ='green', markersize=1.7,alpha=1, label= str(htype) +' '+ htype_str)

    for htype, hcolor, htype_str, hsize in zip( [0, 1] , [col.gridcolor, col.red] , ['sea ice', 'ssh'] , [1.5, 5]):

        pdata = T07_small[T07_small['heights']['height_segment_ssh_flag'] == htype]
        plt.plot( pdata[x_key]/xscale, pdata['heights']['height_segment_height']*0+AT07_bool_offset, '.', color =hcolor, markersize=hsize, alpha=1, label= 'ATL07 ' + htype_str )



    #plt.plot(T3_small[x_key], T3_small['heights_c_std'] - 1.8, 'k-', linewidth=0.5,alpha=1)
    #uncertainty_y_offset


    box_lims= T3_small['dist'].min()+30, T3_small['dist'].max()
    ax1.axvspan(box_lims[0]/xscale, box_lims[-1]/xscale , color =col.gridcolor, alpha = 0.4, zorder= 0)
    ax2.axvspan(box_lims[0]/xscale, box_lims[-1]/xscale ,color =col.gridcolor, alpha = 0.4, zorder= 0)

    #ax2.set_xlim( T3_small['dist'].min(), T3_small['dist'].max() )


    hkey    = 'heights_c_weighted_mean'
    x       = T3['dist']
    #xlims   = x.iloc[0], x.iloc[-1]
    dd      = np.copy(T3[hkey])


    dd_error = np.copy(T3['heights_c_std'])
    dd_error[np.isnan(dd_error)] = 100
    #plt.hist(1/dd_weight, bins=40)
    #plt.plot(x, dd, 'gray', label='displacement (m) ')

    # compute slope spectra !!
    dd      = np.gradient(dd)
    dd, _   = spicke_remover.spicke_remover(dd, spreed=10, verbose=False)
    dd_nans = (np.isnan(dd) ) + (T3['N_photos'] <= 5)

    dd_filled = np.copy(dd)
    dd_filled[dd_nans] = 0
    #win = create_weighted_window(dd_filled)

    # using gappy data
    dd_no_nans = dd[~dd_nans] # windowing is applied here
    x_no_nans  = x[~dd_nans]
    dd_error_no_nans = dd_error[~dd_nans]

    # using gappy data
    T2_grad_x  =  x[ (x > ll_large) & (x< tt_large) ]
    T2_grad_dd =  dd[ (x > ll_large) & (x< tt_large) ]

    T3_grad_x  =  x_no_nans[ (x_no_nans > ll) & (x_no_nans< tt) ]
    T3_grad_dd =  dd_no_nans[ (x_no_nans > ll) & (x_no_nans< tt) ]

    T3_grad_x  =  x[ (x > ll) & (x< tt) ]
    T3_grad_dd =  dd[ (x > ll) & (x< tt) ]


    #T3_grad_dd[dd_nans]
    ax1.plot(T2_grad_x/xscale, T2_grad_dd +grad_y_offset, '-', color=  col.cascade1, linewidth = 0.6, label='slope data (m/m)')
    ax1.fill_between( T3_grad_x/xscale, T3_grad_dd +grad_y_offset, y2=grad_y_offset, color=col.cascade3,alpha=0.4)
    ax1.axhline(grad_y_offset, linewidth= 0.5 , color = 'black')

    ax2.plot( T3_grad_x/xscale, T3_grad_dd +grad_y_offset, '-', color=  col.cascade1, linewidth = 0.6, label='slope data (m/m)')
    ax2.fill_between( T3_grad_x/xscale, T3_grad_dd+ grad_y_offset, y2=grad_y_offset, color=col.cascade3,alpha=0.5)
    ax2.axhline(grad_y_offset, linewidth= 0.5 , color = 'black')


    #plt.fill_between(  T3_large[x_key]/xscale, uncertainty_y_offset+ T3_large['heights_c_std']/2 , y2=uncertainty_y_offset -T3_large['heights_c_std']/2, color=col.cascade2,alpha=1)
    #ax1.axhline(uncertainty_y_offset, linewidth= 0.5 , color = 'black')
    dx = np.median(np.diff(T3_small['dist']))
    #sum(np.isnan(T3_small['heights_c_std']))
    plt.fill_between(  T3_small[x_key]/xscale, uncertainty_y_offset + T3_small['heights_c_std'] /dx , y2=uncertainty_y_offset-T3_small['heights_c_std']/dx, color=col.cascade2,alpha=1, label='uncertrainty')
    ax2.axhline(uncertainty_y_offset, linewidth= 0.5 , color = 'black')

    ax1.spines['bottom'].set_visible(False)
    ax1.spines['top'].set_visible(True)
    ax1.spines['top'].set_linewidth(0.2)

    ax1.xaxis.set_ticks_position('top')
    ax1.xaxis.set_label_position('top')
    ax1.set_xlabel('Along track distance x (km)')

    xlims_large = T2_large['dist'].min()/xscale, T2_large['dist'].max()/xscale
    ax1.set_xlim(xlims_large[0], xlims_large[1]  )

    ax1.spines['left'].set_position(('outward', 10))
    y_ticks =  MT.tick_formatter(   np.arange(-2, 1.4, 1),  interval=1, rounder=1, expt_flag=False, shift=0 )
    ax1.set_yticks(y_ticks[1])
    ax1.set_yticklabels(y_ticks[0])

    #ax1.xaxis.label.set_color(col.gray)
    ax1.tick_params(axis='both', colors=col.gray)
    ax1.spines['left'].set_color(col.gray)


    x_ticks =  MT.tick_formatter(   np.arange(np.round(xlims_large[0]) , np.round(xlims_large[1]), 1),  interval=2, rounder=1, expt_flag=False, shift=1 )
    ax1.set_xticks(x_ticks[1])
    ax1.set_xticklabels(x_ticks[0])


    ax2.spines['bottom'].set_visible(False)
    ax2.spines['top'].set_visible(True)
    ax2.spines['top'].set_linewidth(0.2)
    ax2.xaxis.set_ticks_position('top')
    ax2.xaxis.set_label_position('top')

    ax2.spines['left'].set_position(('outward', 10))


    plt.ylabel('Photon height (m)')


    ax2.set_yticks(y_ticks[1])
    ax2.set_yticklabels(y_ticks[0])

    x_ticks =  MT.tick_formatter(   np.arange(np.round(xlims_large[0]) , np.round(xlims_large[1]), 1),  interval=1, rounder=1, expt_flag=False, shift=1 )

    ax2.set_xticks(x_ticks[1])
    ax2.set_xticklabels(x_ticks[0])
    ax2.set_xlim( box_lims[0]/xscale, box_lims[-1]/xscale  )

    ax2.tick_params(axis='both', colors=col.gray)
    ax2.spines['left'].set_color(col.gray)
    ax2.set_xlabel('Along track distance x (km)')

    ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax1.set_title(next(fn), loc= 'left', y =1.2)
    ax2.set_title(next(fn), loc= 'left', y =1.1)
    plt.suptitle('Example Photon Heights in the Marginal Ice Zone\n' + io.ID_to_str(ID_name), x=0.125, y = 1.03, ha='left')

    F.save_light(path= plot_path, name='B01_ALT03_signal_process_'+ID_name +'_'+ str(chunk_list_i))
    F.save_pup(path= plot_path, name='B01_ALT03_signal_process_'+ID_name)



# %%

# Introductionary figure
font_for_print()
F = M.figure_axis_xy(8, 1.1, view_scale=0.8, container =True)

gs = GridSpec(3,1,  wspace=0.1,  hspace=0.7)#figure=fig,
ax1 = F.fig.add_subplot(gs[0:2, :]) #plt.subplot(1, 6, fp)
#ax1 = F.ax

key         = 'gt2r'
T2         = B2[key].copy()
T2['dist'] = T2['x'] - track_dist_bounds[0]
T2_large  = T2[ (T2['dist'] > ll_large) & (T2['dist'] < tt_large) ]


#plt.title('Beam ' + str(key), loc='left')
#, s= 1,  marker='o', color='black',   alpha =0.02, edgecolors= 'none' )
plt.plot( T2_large[x_key]/xscale, T2_large['heights_c'],   'o', color= col.rels[key],  markersize= 0.4, alpha =0.04 )
#plt.plot( T3_large[x_key]/xscale, T3_large['heights_c_weighted_mean'] , '.', color=lead_color, linewidth=0.5, markersize=1,alpha=0.9, label='x-gauss weighted mean +1')

y_ticks =  MT.tick_formatter(   np.arange(0, 2+1, 1),  interval=1, rounder=1, expt_flag=False, shift=0 )
ax1.spines['bottom'].set_visible(False)
# ax1.spines['top'].set_visible(True)
ax1.spines['bottom'].set_linewidth(0.5)
ax1.spines['left'].set_linewidth(0.5)
ax1.xaxis.set_ticks_position('bottom')
ax1.xaxis.set_label_position('bottom')
ax1.tick_params(bottom=False, labelbottom= False)
#ax1.axhline(0, linewidth= 0.5 , color = 'black')
# ax1.set_yticks(y_ticks[1])
# ax1.set_yticklabels(y_ticks[0])
ax1.set_ylim(0, 2)
ax1.set_xlim(xlims_large[0], xlims_large[1]  )
plt.ylabel('meters')


ax1 = F.fig.add_subplot(gs[1:3, :]) #plt.subplot(1, 6, fp)

key         = 'gt2l'
T2         = B2[key].copy()
T2['dist'] = T2['x'] - track_dist_bounds[0]
T2_large  = T2[ (T2['dist'] > ll_large) & (T2['dist'] < tt_large) ]

#plt.title('Beam ' + str(key), loc='left')
#, s= 1,  marker='o', color='black',   alpha =0.02, edgecolors= 'none' )
plt.plot( T2_large[x_key]/xscale, T2_large['heights_c'],   'o', color= col.rels[key],  markersize= 0.4, alpha =0.04 )
#plt.plot( T3_large[x_key]/xscale, T3_large['heights_c_weighted_mean'] , '.', color=lead_color, linewidth=0.5, markersize=1,alpha=0.9, label='x-gauss weighted mean +1')


xlims_large = T2_large['dist'].min()/xscale, T2_large['dist'].max()/xscale
ax1.set_xlim(xlims_large[0], xlims_large[1]  )
#plt.xlim(T2_large[x_key][0]/xscale, T2_large[x_key][-1]/xscale)
#plt.xlabel('Meters from the Sea Ice Edge')
ax1.set_xlabel('Along track distance x (km)')
#ax2.xaxis.set_ticks_position('top')
#ax2.xaxis.set_label_position('top')

ax1.set_facecolor((1.0, 1.00, 1.00, 0))
ax1.spines['bottom'].set_visible(False)
ax1.spines['right'].set_visible(True)
ax1.spines['left'].set_visible(False)
ax1.spines['bottom'].set_linewidth(0.5)
ax1.spines['right'].set_linewidth(0.5)
ax1.xaxis.set_ticks_position('bottom')
ax1.xaxis.set_label_position('bottom')
ax1.tick_params(bottom=False, right = True, left = False, labelright= True, labelleft= False, labelbottom= True)
#ax1.axhline(0, linewidth= 0.5 , color = 'black')
ax1.set_yticks(y_ticks[1])
ax1.set_yticklabels(y_ticks[0])
ax1.set_ylim(-2, 2)
# plt.ylabel('meters')

F.save_light(path= plot_path, name='B01_ALT03_intro_'+ID_name)
F.save_pup(path= plot_path, name='B01_ALT03_intro_'+ID_name)


# %%
# GFT alternation figure

dl = 25e3
chunk_list = np.arange( np.floor(latlims[0]/1e3/10) *1e4 , latlims[1], dl )[::1]

chunk_list_i= 1
ll_large = chunk_list[chunk_list_i]#+2000
tt_large = ll_large + 25e3 *3


font_for_print()

F = M.figure_axis_xy(fig_sizes['two_column'][0], fig_sizes['two_column'][1]*0.25, view_scale=0.8, container =True)

gs = GridSpec(3,1,  wspace=0.1,  hspace=0.7)#figure=fig,
ax1 = F.fig.add_subplot(gs[0:2, :]) #plt.subplot(1, 6, fp)
#ax1 = F.ax

key         = 'gt2l'
T2         = B2[key].copy()
T2['dist'] = T2['x'] - track_dist_bounds[0]
T2_large  = T2[ (T2['dist'] > ll_large) & (T2['dist'] < tt_large) ]
xlims_large = T2_large['dist'].min()/xscale, T2_large['dist'].max()/xscale


#plt.title('Beam ' + str(key), loc='left')
#, s= 1,  marker='o', color='black',   alpha =0.02, edgecolors= 'none' )
plt.plot( T2_large[x_key]/xscale, T2_large['heights_c'],   'o', color= col.rels[key],  markersize= 0.4, alpha =0.04 )
#plt.plot( T3_large[x_key]/xscale, T3_large['heights_c_weighted_mean'] , '.', color=lead_color, linewidth=0.5, markersize=1,alpha=0.9, label='x-gauss weighted mean +1')

y_ticks =  MT.tick_formatter(   np.arange(0, 2+1, 1),  interval=1, rounder=1, expt_flag=False, shift=0 )
ax1.spines['bottom'].set_visible(False)
# ax1.spines['top'].set_visible(True)
ax1.spines['bottom'].set_linewidth(0.5)
ax1.spines['left'].set_linewidth(0.5)
ax1.xaxis.set_ticks_position('bottom')
ax1.xaxis.set_label_position('bottom')
#ax1.tick_params(bottom=False, labelbottom= False)
#ax1.axhline(0, linewidth= 0.5 , color = 'black')
# ax1.set_yticks(y_ticks[1])
# ax1.set_yticklabels(y_ticks[0])
ax1.set_ylim(-2, 2)
ax1.set_xlim(xlims_large[0], xlims_large[1]  )
plt.ylabel('meters')

#plt.grid('horizontal')

x_ticks =  MT.tick_formatter( np.arange(ll_large/1e3, tt_large/1e3, 12.5)  ,  interval=1, rounder=1, expt_flag=False, shift=0 )
ax1.set_xticks(x_ticks[1])
ax1.set_xticklabels(x_ticks[0])
ax1.axhline(0, linewidth= 0.5 , color = 'black')
ax1.grid(which='major', axis='x', linestyle='-', color='black', linewidth= 0.5)

# ax1 = F.fig.add_subplot(gs[1:3, :]) #plt.subplot(1, 6, fp)
#
# key         = 'gt2l'
# T2         = B2[key].copy()
# T2['dist'] = T2['x'] - track_dist_bounds[0]
# T2_large  = T2[ (T2['dist'] > ll_large) & (T2['dist'] < tt_large) ]
#
# #plt.title('Beam ' + str(key), loc='left')
# #, s= 1,  marker='o', color='black',   alpha =0.02, edgecolors= 'none' )
# plt.plot( T2_large[x_key]/xscale, T2_large['heights_c'],   'o', color= col.rels[key],  markersize= 0.4, alpha =0.04 )
# #plt.plot( T3_large[x_key]/xscale, T3_large['heights_c_weighted_mean'] , '.', color=lead_color, linewidth=0.5, markersize=1,alpha=0.9, label='x-gauss weighted mean +1')


# xlims_large = T2_large['dist'].min()/xscale, T2_large['dist'].max()/xscale
#ax1.set_xlim(xlims_large[0], xlims_large[1]  )
#plt.xlim(T2_large[x_key][0]/xscale, T2_large[x_key][-1]/xscale)
#plt.xlabel('Meters from the Sea Ice Edge')
ax1.set_xlabel('Along track distance x (km)')
#ax2.xaxis.set_ticks_position('top')
#ax2.xaxis.set_label_position('top')

# ax1.set_facecolor((1.0, 1.00, 1.00, 0))
# ax1.spines['bottom'].set_visible(False)
# ax1.spines['right'].set_visible(True)
# ax1.spines['left'].set_visible(False)
# ax1.spines['bottom'].set_linewidth(0.5)
# ax1.spines['right'].set_linewidth(0.5)
# ax1.xaxis.set_ticks_position('bottom')
# ax1.xaxis.set_label_position('bottom')
#ax1.tick_params(bottom=False, right = True, left = False, labelright= True, labelleft= False, labelbottom= True)
#ax1.axhline(0, linewidth= 0.5 , color = 'black')
#ax1.set_yticks(y_ticks[1])
#ax1.set_yticklabels(y_ticks[0])
#ax1.set_ylim(-2, 2)
# plt.ylabel('meters')

F.save_light(path= plot_path, name='B01_ALT03_GFT_alter_'+ID_name)
F.save_pup(path= plot_path, name='B01_ALT03_GFT_alter_'+ID_name)
