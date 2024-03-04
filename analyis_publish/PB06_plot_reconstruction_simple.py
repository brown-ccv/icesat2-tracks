

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

#xr.set_options(display_style='text')()
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

ID_name, batch_key, ID_flag = 'SH_20190224_08800210', 'SH_publish', True
#ID_name, batch_key, ID_flag =  'SH_20190219_08070210', 'SH_publish', True




ID, _, hemis, batch = io.init_data(ID_name, batch_key, ID_flag, mconfig['paths']['work'],  )
#print(ID_name, batch_key, ID_flag)
hemis, batch = batch_key.split('_')


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
for b in all_beams:
    B07[b]= io.getATL07_beam(ATL07_path +ID['tracks']['ATL07']+'.h5', beam=b)

# print(Gk)
# print(Gx)



# %% check paths (again)
col.colormaps2(21)
col_dict= col.rels

#  define simple routines
def add_info(D, Dk, ylims):
    eta  = D.eta + D.x
    N_per_stancil, ksize = Dk.N_per_stancil.data , Dk.k.size
    plt.text(eta[0].data, ylims[-1], '  N='+numtostr(N_per_stancil)  + ' N/2M= '+ fltostr(N_per_stancil/2/ksize, 1))

# Single views
def plot_data_eta(D,  offset = 0  ,  **kargs ):
    eta_1  = D.eta + D.x
    y_data = D.y_model +offset
    plt.plot(eta_1,y_data , **kargs)
    return eta_1

def plot_model_eta(D, ax,  offset = 0,  **kargs ):
    eta  = D.eta + D.x
    y_data = D.y_model+offset
    plt.plot(eta ,y_data , **kargs)

    ax.axvline(eta[0].data, linewidth=0.1,  color=kargs['color'], alpha=0.5)
    ax.axvline(eta[-1].data, linewidth=0.1,  color=kargs['color'], alpha=0.5)

# %%
fltostr  = MT.float_to_str
numtostr = MT.num_to_str
font_for_print()

#for i in x_pos_sel[::2]:
#i =x_pos_sel[20]
#MT.mkdirs_r(plot_path+'B03_spectra/')


#~np.isnan(Gk.mean('beam').mean('k').gFT_PSD_data.data)
x_pos_sel =  np.arange(Gk.x.size)[(Gk.mean('beam').mean('k').gFT_PSD_data.data >0 )]
x_pos_max = Gk.mean('beam').mean('k').gFT_PSD_data[~np.isnan(Gk.mean('beam').mean('k').gFT_PSD_data)].argmax().data
xpp = x_pos_sel[ [int(i) for i in np.round(np.linspace(0, x_pos_sel.size-1, 4))]]
#xpp = np.insert(xpp, 0, x_pos_max)
xpp =x_pos_sel

# %%
i = 5
#i=6

k = all_beams[0]
#k = 'gt2l'

plot_path   = mconfig['paths']['plot'] + '/'+hemis+'/'+batch_key+'/publish/' + ID_name + '/B06/'
MT.mkdirs_r(plot_path)
font_for_print()


for i in xpp:

    fn = copy.copy(lstrings)

    F = M.figure_axis_xy(5.5, 6.5, container =True, view_scale= 0.8)

    plt.suptitle('ALT03 Decomposition\n'+ io.ID_to_str(ID_name), y = 0.93, x = 0.13, horizontalalignment ='left')
    #Photon height reconstruction | x='+str(Gk.x[i].data)+' \n' + ID_name, y = 0.95)
    gs = GridSpec(30,6,  wspace=0,  hspace=0.8)#figure=fig,

    ax0 = F.fig.add_subplot(gs[0:6, :])
    col_d = col.__dict__['rels']
    plt.title(' '+next(fn)+ 'Slope data and model', loc='left', y= 0.83)
    dx = Gx.eta.diff('eta').mean().data
    neven = True
    offs = 0
    Gx_1 = Gx.isel(x= i).sel(beam = k)
    Gk_1 = Gk.isel(x= i).sel(beam = k)

    k_thresh = Gk_1.k_lim.data
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

    B07_sel=B07[k][(T3_sel['delta_time'].min() < B07[k]['time']['delta_time']) & (B07[k]['time']['delta_time'] < T3_sel['delta_time'].max()) ]

    ### slope data
    T3 = B3[k]#.loc[mask_x_bin]
    dd      = np.copy(T3['heights_c_weighted_mean'])
    dd      = np.gradient(dd)
    dd, _   = spicke_remover.spicke_remover(dd, spreed=10, verbose=False)
    dd_nans = (np.isnan(dd) ) + (T3['N_photos'] <= 5)
    # dd_no_nans = dd[~dd_nans] # windowing is applied here
    # x_no_nans  = T3['dist'][~dd_nans]
    dd[dd_nans] = np.nan# windowing is applied here
    xx          = T3['dist']
    xx[dd_nans] = np.nan

    #plt.plot( xx , dd, color=col.green,alpha=0.8, linewidth =0.3)
    #B3[k]['dist']

    #plot_model_eta(Gx_1, ax0, offset= offs,  linestyle='-', color=col_d[k], linewidth=0.4, alpha=1, zorder=12 , label = 'GFT model')
    # ylims= -np.nanstd(Gx_1.y_data)*3, np.nanstd(Gx_1.y_data)*3
    # #add_info(Gx_1, Gk_1 , ylims )

    lead_color = col.cascade1#col_d[k]

    # oringial data
    eta_1= plot_data_eta(Gx_1 , offset= offs , linestyle= '-', c=col.gray,linewidth=2,  alpha =1, zorder=11, label = 'mean photon\nheight slope')

    # reconstruct slope model
    # introduce frequency filter:
    gFT_cos_coeff_sel = np.copy(Gk_1.gFT_cos_coeff)
    gFT_sin_coeff_sel = np.copy(Gk_1.gFT_sin_coeff)
    gFT_cos_coeff_sel[Gk_1.k > k_thresh] = 0
    gFT_sin_coeff_sel[Gk_1.k > k_thresh] = 0


    FT = gFT.generalized_Fourier(Gx_1.eta + Gx_1.x, None,Gk_1.k )
    _ = FT.get_H()
    FT.b_hat=np.concatenate([ gFT_cos_coeff_sel, gFT_sin_coeff_sel ])
    plt.plot(Gx_1.eta + Gx_1.x, FT.model()+offs ,'-', c=lead_color, linewidth=0.5, alpha=1,zorder= 12, label = 'GFT slope model')


    plt.legend(loc=1, ncol=2)

    ax1 = F.fig.add_subplot(gs[6:13, :])
    plt.title(' '+next(fn)+ 'Height data and model', loc='left', y= 0.83)

    ### height decomposition
    # plotting observed datazx
    # T3_sel['heights_c_weighted_mean']
    plt.plot( T3_sel['dist'] , T3_sel['heights_c_weighted_mean'], '-' , color =col_d[k], linewidth = 1, label = 'observed $h_c$ mean')

    if T2_sel['x_true'].iloc[0] > T2_sel['x_true'].iloc[-1]:
         T2_sel['dist'] = np.interp(T2_sel['x_true'][::-1], T3_sel['x_true'][::-1],  T3_sel['dist'][::-1] )[::-1]
    else:
        T2_sel['dist'] = np.interp(T2_sel['x_true'][::1], T3_sel['x_true'][::1],  T3_sel['dist'] )
    plt.scatter( T2_sel['dist'] , T2_sel['heights_c'], s= 1,  marker='o', color='black',   alpha =0.02, edgecolors= 'none' )

    try:
        if (T3_sel['delta_time'].iloc[0] > T3_sel['delta_time'].iloc[-1]) is (B07_sel['time']['delta_time'].iloc[0] > B07_sel['time']['delta_time'].iloc[-1]):
            B07_sel['dist'] = np.interp(B07_sel['time']['delta_time'], T3_sel['delta_time'],  T3_sel['dist'] )
        else:
            B07_sel['dist'] = np.interp(np.array(B07_sel['time']['delta_time']), np.array(T3_sel['delta_time'][::-1]),  np.array(T3_sel['dist'][::-1]) )
        B07_flag = True
    except:
        B07_flag = False

    # B07_sel['dist'] = scipy.interpolate.griddata(np.array(T3_sel['delta_time'][::-1]), np.array(T3_sel['dist'][::-1]) ,np.array(B07_sel['time']['delta_time']),  method= 'nearest' )
    # plt.plot(np.array(B07_sel['time']['delta_time']))
    # plt.plot(np.array(T3_sel['delta_time'][::-1]))
    # plt.grid()

    # plt.plot( B07_sel['dist']- B07_sel['dist_np'])
    #plt.plot( B07_sel['dist'] , B07_sel['heights']['height_segment_height'], color='black',  alpha =1, linewidth = 0.6 )
    #plt.scatter(B07_sel['dist'], B07_sel['heights']['height_segment_height'], s= 1,  marker='o', color='black',   alpha =0.7, edgecolors= 'none' )

    dist_nanmask = np.isnan(Gx_1.y_data)
    # height_data  = np.interp(dist_stencil, T3_sel['dist'],  T3_sel['heights_c_weighted_mean']) #[~np.isnan(Gx_1.y_data)]

    # def fit_offset(x, data,  model, nan_mask, deg):
    #
    #     #x, data,  model, nan_mask, deg = dist_stencil, height_data, height_model2, dist_nanmask, 1
    #     p_offset = np.polyfit(x[~nan_mask], data[~nan_mask] - model[~nan_mask], deg)
    #     p_offset[-1] = 0
    #     poly_offset = np.polyval(p_offset,x )
    #     return poly_offset
    #
    # poly_offset = fit_offset(dist_stencil, height_data, height_model2, dist_nanmask, 1)

    #plt.plot(dist_stencil, height_model2 ,'-', c='orange', linewidth=0.6, alpha=1,zorder= 12, label = 'spectral int model')
    #plt.plot(dist_stencil, poly_offset ,'-', c=col.gridcolor, linewidth=0.6, alpha=1,zorder= 12, label = 'offset')
    # plt.plot(dist_stencil, height_model2 ,'-', c=lead_color, linewidth=0.8, alpha=1,zorder= 12, label = 'GFT height model + correction')

    plt.plot( T3_sel['dist'] ,  T3_sel['heights_c_model'] ,'-', c=lead_color, linewidth=0.5, alpha=1,zorder= 12, label = '$h_c$ model')

    plt.legend(loc = 4, ncol =2)

    # ax1b = F.fig.add_subplot(gs[13:19, :])
    # plt.title(' '+next(fn)+ 'ATL07', loc='left', y= 0.83)
    # x_key= 'dist'
    # AT07_cat_offset= 0
    # AT07_bool_offset =0
    # htype_cmap = [col.orange, col.cascade3, col.cascade2, col.cascade1]
    # htype_list= ['cloud_covered','other', 'specular_lead', 'dark_lead' , 'other']
    # for htype, hcolor, htype_str in zip( [0, 1, (2, 5), (6, 9)] , htype_cmap , htype_list ):
    #     if type(htype) is tuple:
    #         imask = (B07_sel['heights']['height_segment_type'] >= htype[0]) & (B07_sel['heights']['height_segment_type'] <= htype[1])
    #     else:
    #         imask = B07_sel['heights']['height_segment_type'] == htype
    #     pdata = B07_sel[imask]
    #     plt.plot( pdata[x_key], pdata['heights']['height_segment_height'] + AT07_cat_offset, '.', color =hcolor, markersize=0.8,alpha=1, label=htype_str)
    #
    #
    # for htype, hcolor, htype_str, hsize in zip( [0, 1] , [col.gridcolor, col.red] , [None, 'ssh'] , [0.8, 5]):
    #
    #     pdata = B07_sel[B07_sel['heights']['height_segment_ssh_flag'] == htype]
    #     plt.plot( pdata[x_key], pdata['heights']['height_segment_height']*0+AT07_bool_offset, '.', color =hcolor, markersize=hsize,alpha=0.9, label=htype_str)
    #
    # plt.legend(ncol=3, loc=1)


    #plt.plot( B07_sel['dist_np'] , B07_sel['heights']['height_segment_height'], color='red',  alpha =1, linewidth = 0.6 )

    #     plt.plot( B07_sel['ref']['latitude'][0:100] , B07_sel['heights']['height_segment_height'][0:100])#, s= 1,  marker='o', color='black',   alpha =1, edgecolors= 'none' )
    #     plt.plot( T3_sel['lats'][0:100] , T3_sel['heights_c_model'][0:100])#, s= 1,  marker='o', color='black',   alpha =1, edgecolors= 'none' )
    #
    #     plt.plot( B07_sel['dist'][0:200] , B07_sel['heights']['height_segment_height'][0:200], 'r')#, s= 1,  marker='o', color='black',   alpha =1, edgecolors= 'none' )
    #     plt.plot( T3_sel['dist'][:] , T3_sel['heights_c_model'][:])#, s= 1,  marker='o', color='black',   alpha =1, edgecolors= 'none' )
    #
    # T2_sel
    # B07_sel.T
    # B07_sel['ref']['height_segment_id']
    # B07_sel['heights']['height_segment_ssh_flag'].plot()
    # B07_sel['heights']['height_segment_type'].plot()


    # T3_sel['heights_c_weighted_mean'].plot()
    # T2_sel['heights_c'].plot()
    #
    # T2_sel['dist'].iloc[0], T3_sel['dist'].iloc[0]
    # T2_sel['dist'].iloc[-1], T3_sel['dist'].iloc[-1]

    # reconstructued data by integration
    # height_model = np.cumsum(FT.model()) + T3_sel['heights_c_weighted_mean'].iloc[0]
    # plt.plot( Gx_1.eta + Gx_1.x, height_model, linewidth = 0.6  , color = 'red', label = 'real space integral')

    # FT_int = gFT.generalized_Fourier(Gx_1.eta + Gx_1.x, None,Gk_1.k )
    # _ = FT_int.get_H()
    # FT_int.b_hat = np.concatenate([ -gFT_sin_coeff_sel /Gk_1.k, gFT_cos_coeff_sel/Gk_1.k ])
    #
    # height_model2 = FT_int.model() /dx# + T3_sel['heights_c_weighted_mean'].iloc[0]



    ax2 = F.fig.add_subplot(gs[13:18, :])
    plt.title(' '+next(fn)+ 'Residual heights of ATL03', loc='left', y= 0.83)
    height_residual = T2_sel['heights_c_residual'] #T2_sel['heights_c'] - np.interp(T2_sel['dist'], dist_stencil, height_model2 +  poly_offset)
    plt.scatter(T2_sel['dist'], height_residual, s= 1,  marker='o', color='black',   alpha =0.02, edgecolors= 'none' )

    # heights_c_weighted_mean_stancil = np.interp(dist_stencil, T3_sel['dist'], T3_sel['heights_c_weighted_mean'] )
    # height_residual_mean = (heights_c_weighted_mean_stancil - height_model2) - poly_offset
    # height_residual_mean[dist_nanmask] = np.nan



    height_residual_mean = T3_sel['heights_c_residual']
    height_residual_mean_x = T3_sel['dist']
    plt.plot( height_residual_mean_x , height_residual_mean , color =col.rascade1, linewidth = 0.5, label = 'residual $h_c$')
    plt.fill_between(height_residual_mean_x , height_residual_mean, color= col.cascade2, edgecolor = None, alpha = 0.4, zorder= 0)
    plt.legend(loc = 1)


    # for pos, kgroup, lflag in zip([ gs[2, 0:2],  gs[2, 2:4], gs[2, 4:]],  [, ['gt2l', 'gt2r'], ['gt3l', 'gt3r']], [True, False, False] ):

    ax41 = F.fig.add_subplot(gs[3:6, 4:])

    #ax41.tick_params(labelleft=lflag)

    dd = Gk_1.gFT_PSD_data#.rolling(k=10, min_periods= 1, center=True).mean()
    plt.plot(Gk_1.k,  dd, color='gray', linewidth=.5 ,alpha= 0.5 )

    dd = Gk_1.gFT_PSD_data.rolling(k=10, min_periods= 1, center=True).mean()
    plt.plot(Gk_1.k,  dd, color=lead_color, linewidth=.8 )

    klim= Gk_1.k[0], Gk_1.k[-1]
    plt.xlim(klim)

    plt.ylabel('$(m/m)^2/k$')
    #plt.title('Spectra', loc ='left')s
    #plt.xlabel('k (2$\pi$ m$^{-1}$)')
    plt.ylim(dd.min(),np.nanmax(dd.data) * 1.5)

    ax41.axvline(k_thresh, linewidth=1,  color='black', alpha=1)
    ax41.axvspan(k_thresh , klim[-1],   color='black', alpha=0.5, zorder=12)
    ax41.set_facecolor((1.0, 1.00, 1.00, 0.8))

    #plt.show()

    #F.save_light(path=plot_path+'B03_spectra/', name = 'B03_freq_reconst_x'+str(i))
    #MT.json_save('B03_success', plot_path, {'time':'time.asctime( time.localtime(time.time()) )'})

    stencil_pos = spec.create_chunk_boundaries_unit_lengths(1000, (dist_stencil[0], dist_stencil[-1]), ov=0, iter_flag=False).T

    V0_list, V1_list, V2_list, V3_list = list(), list(), list(), list()
    no_nan_sum= list()
    for s in stencil_pos:
        V0_list.append( T2_sel['heights_c'].loc[M.cut_nparray( np.array(T2_sel['dist']), s[0], s[-1]) ].var() )
        V1_list.append( T3_sel['heights_c_weighted_mean'].loc[M.cut_nparray( np.array(T3_sel['dist']), s[0], s[-1]) ].var() )
        V2_list.append( np.nanvar( T3_sel['heights_c_model'].loc[M.cut_nparray( np.array(T3_sel['dist']), s[0], s[-1])]) )
        V3_list.append( np.nanvar(height_residual_mean[ M.cut_nparray( np.array(height_residual_mean_x), s[0], s[-1])]) )

        no_nan_sum.append( (~dist_nanmask[M.cut_nparray( dist_stencil, s[0], s[-1])].data).sum())


    ax3 = F.fig.add_subplot(gs[21:23, :])

    plt.title(next(fn) + 'Variance Decomposition', loc='left')
    V0_list, V1_list, V2_list = np.array(V0_list),np.array(V1_list),np.array(V2_list),
    no_nan_sum = np.array(no_nan_sum)
    no_nan_sum = no_nan_sum/no_nan_sum.max()

    edge_pos = np.insert(stencil_pos[:,0], stencil_pos[:,0].size, stencil_pos[:,-1][-1])
    plt.stairs(no_nan_sum * V0_list/V0_list,  edge_pos, baseline=0, fill=True, color= col.black, alpha=0.6, label = 'photon variance')
    plt.stairs(no_nan_sum * V1_list/V0_list,  edge_pos, baseline=0, fill=True, color= col_d[k] ,            label = 'mean photon variance')
    plt.stairs(no_nan_sum * V2_list/V0_list,  edge_pos, baseline=0, fill=True, color= lead_color,           label = 'wave variance')
    plt.stairs(no_nan_sum * (V3_list/V0_list+ V2_list/V0_list) ,  edge_pos, baseline=no_nan_sum * V2_list/V0_list, fill=True, color= col.green, label = 'residual variance')

    plt.legend(ncol= 4, bbox_to_anchor=(-0.02, 0), loc= 2)

    # residual
    #ax0.set_xticks(eta_ticks)
    #ax0.set_xticklabels(eta_ticks/1e3)
    #ax0.set_ylabel('Slope (m/m)')
    #ax1.spines['top'].set_visible(True)
    #ax1.spines['top'].set_linewidth(0.2)

    #ax1.xaxis.set_ticks_position('top')
    #ax1.xaxis.set_label_position('top')
    ax0.set_ylabel('Slope (m/m)')
    ax1.set_ylabel('Photon Height (m)')
    ax2.set_ylabel('Photon Height (m)')

    #ax2.spines['bottom'].set_visible(True)
    ax2.set_xlabel('Distance from the ice Edge (km)')

    eta_ticks =  np.arange(dist_stencil[0], dist_stencil[-1]+ 500, 500)
    eta_tick_labels, eta_ticks = MT.tick_formatter(eta_ticks[1::4]/1e3, interval= 3, expt_flag= False, shift=0)


    y_tick_labels, y_ticks = MT.tick_formatter(np.arange(-0.1, 0.1+ 0.05, 0.05), interval= 2, expt_flag= False, shift=0)
    ax0.set_yticks(y_ticks)
    ax0.set_yticklabels(y_tick_labels)
    ylim_slope= np.round(Gx_1.y_data.std().data*3 * 10)/10
    ax0.set_ylim(-1.3* ylim_slope ,ylim_slope*1.5)

    y_tick_labels, y_ticks = MT.tick_formatter(np.arange(-0.5, 3, 0.5), interval= 2, expt_flag= False, shift=1)
    ax1.set_yticks(y_ticks)
    ax1.set_yticklabels(y_tick_labels)
    ax1.set_ylim(-0.8, 1.5)

    # ax1b.set_yticks(y_ticks)
    # ax1b.set_yticklabels(y_tick_labels)
    # ax1b.set_ylim(-0.8, 0.8)

    ax2.set_xticks(eta_ticks*1e3)
    ax2.set_xticklabels(eta_tick_labels)

    ax2.set_yticks(y_ticks)
    ax2.set_yticklabels(y_tick_labels)
    ax2.set_ylim(0, 1.5)

    ax3.set_yticks(y_ticks)
    ax3.set_yticklabels(y_tick_labels)
    ax3.set_ylim(0, 1)

    xlims= eta_1[0].data+ 0 * dx, eta_1[-1].data- 1000 * dx
    #xlims= eta_1[0].data+ 0 * dx, eta_1[-1].data- 0 * dx

    for axx in [ax0, ax1, ax3]:
        axx.set_xlim(xlims )
        axx.axhline(0, linewidth =0.5, color=col.black)
        axx.spines['bottom'].set_visible(False)
        axx.tick_params(labelbottom=False, bottom=False)
    ax2.set_xlim(xlims )


    #F.save_light(path= plot_path, name='B03_decomposition_'+str(num_count).zfill(4))
    #F.save(path= plot_path, name='B06_decomposition_2_'+k+'_x'+str(i)+'_'+ID_name)

    F.save_light(path= plot_path, name='B06_decomposition_simple_'+k+'_x'+str(i)+'_'+ID_name)
    #F.save_pup(path= plot_path, name='B06_decomposition_simple_'+k+'_x'+str(i)+'_'+ID_name)


    #num_count +=1
# %%

V0_photon_var       = T2_sel['heights_c'].var()
V1_mean_photon_var  = T3_sel['heights_c_weighted_mean'].var()
V2_wave_model_var  = np.nanvar(height_residual_mean)

V0_photon_var/ V0_photon_var
V1_mean_photon_var/V0_photon_var
V2_wave_model_var/V0_photon_var
