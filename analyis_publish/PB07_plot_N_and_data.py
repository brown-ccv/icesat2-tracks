
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

#import xarray as xr
xr.set_options(display_style='text')
#import s3fs
# %%
ID_name, batch_key, ID_flag = io.init_from_input(sys.argv) # loads standard experiment
#ID_name, batch_key, ID_flag = '20190605061807_10380310_004_01', 'SH_batch01', False
#ID_name, batch_key, ID_flag = '20190601094826_09790312_004_01', 'SH_batch01', False
#ID_name, batch_key, ID_flag = '20190207111114_06260210_004_01', 'SH_batch02', False
#ID_name, batch_key, ID_flag = '20190208152826_06440210_004_01', 'SH_batch01', False
#ID_name, batch_key, ID_flag = '20190213133330_07190212_004_01', 'SH_batch02', False
#ID_name, batch_key, ID_flag = '20190207002436_06190212_004_01', 'SH_batch02', False
#ID_name, batch_key, ID_flag = '20190206022433_06050212_004_01', 'SH_batch02', False


#ID_name, batch_key, ID_flag = '20190215184558_07530210_004_01', 'SH_batch02', False
#ID_name, batch_key, ID_flag = 'SH_20190219_08070210', 'SH_publish', True
#ID_name, batch_key, ID_flag = 'SH_20190502_05160312', 'SH_publish', True
#ID_name, batch_key, ID_flag = 'SH_20190502_05180312', 'SH_publish', True



#ID_name, batch_key, ID_flag =  'SH_20190213_07190212', 'SH_publish', True


# used in paper:
#ID_name, batch_key, ID_flag =  'SH_20190219_08070210', 'SH_publish', True
#ID_name, batch_key, ID_flag = 'SH_20190224_08800210', 'SH_publish', True
#ID_name, batch_key, ID_flag = 'SH_20190502_05160312', 'SH_publish', True # no ATL07 data

ID_name, batch_key, ID_flag = 'SH_20190502_05180312', 'SH_publish', True

TND =mconfig['track_name_dict']

# % 1 X
# % Track 1
# % SH_20190224_08800210
#
# % 3 X
# % Track 2
# % SH_20190219_08070210
#
# % 4 X
# % Track 3
# % SH_20190502_05160312
#
# % 1 X
# % Track 4
# % SH_20190502_05180312


ID, _, hemis, batch = io.init_data(ID_name, batch_key, ID_flag, mconfig['paths']['work'],  )
#print(ID_name, batch_key, ID_flag)
hemis, batch = batch_key.split('_')

plot_path   = mconfig['paths']['plot'] + '/'+hemis+'/'+batch_key+'/publish/B07/'
MT.mkdirs_r(plot_path)

## -------------- use lower level data ------------------
# ATlevel= 'ATL03'
#
# load_path_scratch = mconfig['paths']['scratch'] +'/'+ batch_key +'/'
# load_path_work    = mconfig['paths']['work'] +'/'+ batch_key +'/'
#
# #B0_hdf5    = h5py.File(load_path_scratch +'/A01c_ATL03_'+ID_name+ '_corrected.h5', 'r')
# B2_hdf5    = h5py.File(load_path_work +'B01_regrid'+'/'+ID_name + '_B01_regridded.h5', 'r')
# B3_hdf5    = h5py.File(load_path_work +'B01_regrid'+'/'+ID_name + '_B01_binned.h5', 'r')
#
# B0, B2, B3 = dict(), dict(), dict()
# for b in all_beams:
#     #B0[b] = io.get_beam_hdf_store(B0_hdf5[b])
#     B2[b] = io.get_beam_hdf_store(B2_hdf5[b])
#     B3[b] = io.get_beam_hdf_store(B3_hdf5[b])
#
# B2_hdf5.close(), B2_hdf5.close()
#
# load_path   = mconfig['paths']['work']+ batch_key  +'/B02_spectra/'
# load_file   = load_path + 'B02_' + ID_name #+ '.nc'
# #MT.mkdirs_r(plot_path)
#
# Gk   = xr.open_dataset(load_file+'_gFT_k.nc')
# Gx   = xr.open_dataset(load_file+'_gFT_x.nc')

## -------------------- use final prodiucts
all_beams   = mconfig['beams']['all_beams']
high_beams  = mconfig['beams']['high_beams']
low_beams   = mconfig['beams']['low_beams']

load_path_work  = mconfig['paths']['work'] +'/'+ batch_key +'/'
load_path       = load_path_work  +'/B06_corrected_separated/'

B2        = io.load_pandas_table_dict('B06_' + ID_name+ '_B06_corrected_resid', load_path)
B3              = io.load_pandas_table_dict('B06_' + ID_name+ '_binned_resid', load_path)

load_file       = load_path + 'B06_' + ID_name #+ '.nc'
Gk              = xr.open_dataset(load_file+'_gFT_k_corrected.nc')
Gx              = xr.open_dataset(load_file+'_gFT_x_corrected.nc')

ATL07_path = mconfig['paths']['scratch']+'/'+ batch_key +'/'
os.listdir(ATL07_path)
B07= dict()
try:
    for b in all_beams:
        B07[b]= io.getATL07_beam(ATL07_path +ID['tracks']['ATL07']+'.h5', beam=b)
    B07_flag = True

except:
    B07_flag = False

#B07_flag = False
# print(Gk)
# print(Gx)

# %% check paths (again)
col.colormaps2(21)
col_dict= col.rels
col_d = col.__dict__['rels']


#~np.isnan(Gk.mean('beam').mean('k').gFT_PSD_data.data)
x_pos_sel =  np.arange(Gk.x.size)#[(Gk.mean('beam').mean('k').gFT_PSD_data.data >0 )]
# x_pos_max = Gk.mean('beam').mean('k').gFT_PSD_data[~np.isnan(Gk.mean('beam').mean('k').gFT_PSD_data)].argmax().data
# xpp = x_pos_sel[ [int(i) for i in np.round(np.linspace(0, x_pos_sel.size-1, 4))]]
#xpp = np.insert(xpp, 0, x_pos_max)
xpp =x_pos_sel

x_sel = Gk.x/1e3#[(Gk.mean('beam').mean('k').gFT_PSD_data.data >0 )]/1e3

dx = np.diff(x_sel)[0]/2
edge_pos = np.insert(x_sel.data, 0, 0)#+dx


VAR_stats_sum = None
N_sample_stats_sum = None

N_stack= 0
for k in all_beams:

    N_sample_stats  = pd.DataFrame(index=['ATL03', 'ATL03_used', 'ATL07'] )
    VAR_stats       = pd.DataFrame(index=['ATL03_photon', 'ATL03_wave_model', 'ATL03_smth_data','ATL03_smth_wave_model', 'ATL07_heights'] )

    print(k)

    for i in xpp:
        Gx_1 = Gx.isel(x= i).sel(beam = k)
        Gk_1 = Gk.isel(x= i).sel(beam = k)

        #k_thresh = Gk_1.k_lim.data
        dist_stencil = Gx_1.eta + Gx_1.x
        dist_stencil_lims = dist_stencil[0].data, dist_stencil[-1].data

        # cutting Table data
        # photon data
        # gridded data
        mask_x_bin = ( (B3[k]['dist']  >= dist_stencil_lims[0]) & (B3[k]['dist']  <= dist_stencil_lims[1]) )
        T3_sel = B3[k].loc[mask_x_bin]

        T2 = B2[k]#.sort_index(ascending= False)
        mask_x_true = (T2['x_true'] >= T3_sel['x_true'].min()) & (T2['x_true'] <= T3_sel['x_true'].max())
        T2_sel = B2[k].loc[mask_x_true]
        #sum(mask_x_true)

        if B07_flag:
            B07_sel=B07[k][(T3_sel['delta_time'].min() < B07[k]['time']['delta_time']) & (B07[k]['time']['delta_time'] < T3_sel['delta_time'].max()) ]


        ## photon counts
        N_sample_stats[i]   = [T2_sel.shape[0], Gk_1.N_photons.data, B07_sel['env']['n_photons_actual'].sum() ]

        # vanriance estimates
        # ATL03
        T2_photon_var       = T2_sel['heights_c'].var()
        T2_wave_model_var   = T2_sel['heights_c_model'].var()
        T3_data_var         = T3_sel['heights_c_weighted_mean'].var()
        T3_wave_model_var   = T3_sel['heights_c_model'].var()

        # T2_resid_var        = T2_sel['heights_c_residual'].var()
        # T2_rebin_var        = T2_photon_var - (T2_wave_model_var +  T2_resid_var)

        # ATL 03
        if B07_flag:
            B07_total_var = B07_sel['heights']['height_segment_height'].var()
        else:
            B07_total_var = T3_wave_model_var * np.nan

        VAR_stats[i]   = [T2_photon_var, T2_wave_model_var, T3_data_var, T3_wave_model_var, B07_total_var ]

    if VAR_stats_sum is None:
        VAR_stats_sum       = VAR_stats
        N_sample_stats_sum  = N_sample_stats
    else:
        print('add')
        VAR_stats_sum       += VAR_stats
        N_sample_stats_sum  += N_sample_stats

    N_stack += 1



VAR_stats_sum = VAR_stats_sum.T/N_stack
VAR_stats_sum.index       = x_sel

N_sample_stats_sum = N_sample_stats_sum.T/N_stack
N_sample_stats_sum.index  = x_sel

VAR_stats_sum[VAR_stats_sum == 0] = np.nan
N_sample_stats_sum[N_sample_stats_sum == 0] = np.nan

# %%

# VAR_stats_sum['ATL03_photon'].plot()
# VAR_stats_sum['ATL03_wave_model'].plot()
#
# VAR_stats_sum['ATL03_smth_data'].plot()
# VAR_stats_sum['ATL03_smth_wave_model'].plot()
#
# VAR_stats_sum['ATL07_heights'].plot()
# plt.ylim(0, 0.21 *6)

font_for_print()
fn = copy.copy(lstrings)



#F = M.figure_axis_xy(5.5, 6.5, container =True, view_scale= 0.8)
#F = M.figure_axis_xy( fig_sizes['23rd_width'][0] ,  fig_sizes['23rd_width'][1]*2.3 , container =True, view_scale= 0.8)
F = M.figure_axis_xy( fig_sizes['one_column_high'][0] ,  fig_sizes['one_column_high'][1] * 1.5 , container =True, view_scale= 0.8)

#plt.suptitle('ALT03 Decomposition\n'+ io.ID_to_str(ID_name), y = 0.93, x = 0.13, horizontalalignment ='left')
#plt.suptitle('Explained Variance Decomposition\n'+ io.ID_to_str(ID_name), y = 0.93, x = 0.13, horizontalalignment ='left')
plt.suptitle('Explained Variance \nDecomposition\n'+ TND[ID_name] , y = 0.955, x = 0.13, horizontalalignment ='left')


#Photon height reconstruction | x='+str(Gk.x[i].data)+' \n' + ID_name, y = 0.95)
gs = GridSpec(10, 4,  wspace=0,  hspace=1)#figure=fig,

ax0 = F.fig.add_subplot(gs[0:3, :])
plt.title(' '+next(fn)+ 'ATL03 Expl. Variance', loc='left', y= 0.96)

#edge_pos = np.insert(VAR_stats_sum.index, VAR_stats_sum.index.size, VAR_stats_sum.index[-1])
plt.stairs(VAR_stats_sum['ATL03_photon'],  edge_pos, baseline=0, fill=True, color= col.gridcolor, alpha=1, label = 'Photon variance (<20 meter)')
plt.stairs(VAR_stats_sum['ATL03_smth_data'],  edge_pos, baseline=0, fill=True, color= col.cascade2, alpha=0.6, label = '$h_c$ variance (> 20 meters)')
plt.stairs(VAR_stats_sum['ATL03_wave_model'],  edge_pos, baseline=0, fill=True, edgecolor=col.black, color=  col.cascade1, alpha=1, label = 'wave variance (model)', linewidth=0.8)
#plt.stairs(VAR_stats_sum['ATL03_smth_wave_model'],  edge_pos, baseline=0, fill=False, color= col.green, alpha=1, label = 'photon variance')


# plt.stairs(no_nan_sum * V1_list/V0_list,  edge_pos, baseline=0, fill=True, color= col_d[k] ,            label = 'mean photon variance')
# plt.stairs(no_nan_sum * V2_list/V0_list,  edge_pos, baseline=0, fill=True, color= lead_color,           label = 'wave variance')
# plt.stairs(no_nan_sum * (V3_list/V0_list+ V2_list/V0_list) ,  edge_pos, baseline=no_nan_sum * V2_list/V0_list, fill=True, color= col.green, label = 'residual variance')

#plt.legend(ncol= 4, bbox_to_anchor=(-0.02, 0), loc= 2)
#plt.legend(ncol= 2, bbox_to_anchor=(+0.52, 1.30), loc=2)
plt.legend(ncol= 1, bbox_to_anchor=(+0.55, 1.45), loc=2)

y_max  = np.median(VAR_stats_sum['ATL03_photon'])
#y_max  = np.quantile(VAR_stats_sum['ATL03_photon'], 0.9) *1.2
y_max  = np.nanquantile(VAR_stats_sum['ATL03_photon'], 0.8) *1.5

# residual
#ax0.set_xticks(eta_ticks)
#ax0.set_xticklabels(eta_ticks/1e3)
#ax0.set_ylabel('Slope (m/m)')
#ax1.spines['top'].set_visible(True)
#ax1.spines['top'].set_linewidth(0.2)

ax1 = F.fig.add_subplot(gs[1+2:3+2, :])

#\com{change label: "photon variance" -> "Photon variance (<20 meter)"; "observed hc mean" --> "$h_c$ variance > 20 meters"; "wave variance (model)"}


plt.title(' '+next(fn)+ 'ATL07 Observed Variance', loc='left', y= 0.95)

plt.stairs(VAR_stats_sum['ATL07_heights'],     edge_pos, baseline=0, fill=True, color=col.orange, edgecolor=col.gray, alpha=1, label = 'ATL07 variance')
plt.stairs(VAR_stats_sum['ATL03_wave_model'],  edge_pos, baseline=0, fill=False, edgecolor=col.black, alpha=1, linewidth=1, label = 'wave variance as in (a)')

dmask = np.isnan(VAR_stats_sum['ATL07_heights'])
hatch_data = np.ones(VAR_stats_sum['ATL07_heights'].size) +y_max
hatch_data[~dmask] =np.nan
plt.stairs( hatch_data,  edge_pos, baseline=0, fill=True,  color=  col.gridcolor,  alpha=0.3)
plt.stairs( hatch_data,  edge_pos, baseline=0, fill=False, color=  col.black    ,  alpha=0.3, label = 'no data', hatch='//')

plt.legend(ncol= 3, bbox_to_anchor=(0, -0.4), loc=2)

#ax1.xaxis.set_ticks_position('top')
#ax1.xaxis.set_label_position('top')
ax0.set_ylabel('Variance (m$^2$)')
ax0.set_ylim(0, y_max)
ax0.tick_params( bottom=True, labelbottom=False)


ax1.set_ylabel('Variance (m$^2$)')
ax1.set_ylim(0, y_max *1.8/3)

ax1.set_xlim(edge_pos[0], edge_pos[-4]-2)
ax0.set_xlim(edge_pos[0], edge_pos[-4]-2)



ax1.set_xlabel('Distance from Ice Edge (km)')

F.save_light(path= plot_path, name='B07_explvar_'+ID_name)
F.save_pup(path= plot_path  , name='B07_explvar_'+ID_name)

# %%
