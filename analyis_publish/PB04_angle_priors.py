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


import copy
import spicke_remover
import datetime
import concurrent.futures as futures

#from numba import jit
%matplotlib widget
#from ICEsat2_SI_tools import angle_optimizer
import ICEsat2_SI_tools.wave_tools as waves
import concurrent.futures as futures

import time
import ICEsat2_SI_tools.lanczos as lanczos

xr.set_options(display_style='text')
from contextlib import contextmanager
col.colormaps2(21)

@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout

col_dict = col.rels
#import s3fs
# %%
track_name, batch_key, test_flag = io.init_from_input(sys.argv) # loads standard experiment
#track_name, batch_key, test_flag = '20190605061807_10380310_004_01', 'SH_batch01', False
#track_name, batch_key, test_flag = '20190601094826_09790312_004_01', 'SH_batch01', False
#track_name, batch_key, test_flag = '20190207111114_06260210_004_01', 'SH_batch02', False
#track_name, batch_key, test_flag = '20190219073735_08070210_004_01', 'SH_batch02', False
#track_name, batch_key, test_flag = '20190215184558_07530210_004_01', 'SH_batch02', False

# good track
#track_name, batch_key, test_flag = '20190502021224_05160312_004_01', 'SH_batch02', False
#track_name, batch_key, test_flag = '20190502050734_05180310_004_01', 'SH_batch02', False
#track_name, batch_key, test_flag = '20190210143705_06740210_004_01', 'SH_batch02', False
#track_name, batch_key, test_flag = 'NH_20190301_09580203', 'NH_batch05', True

#track_name, batch_key, test_flag = 'SH_20190213_07190212', 'SH_publish', True
#track_name, batch_key, test_flag = 'SH_20190502_05180312', 'SH_publish', True
track_name, batch_key, test_flag = 'SH_20190502_05160312', 'SH_publish', True




#print(track_name, batch_key, test_flag)
hemis, batch = batch_key.split('_')
#track_name= '20190605061807_10380310_004_01'
ATlevel= 'ATL03'

save_path   = mconfig['paths']['work'] +batch_key+'/B04_angle/'
save_name   = 'B04_'+track_name

plot_path   = mconfig['paths']['plot'] + '/'+hemis+'/'+batch_key+'/publish/' + track_name + '/'
#plot_path   = mconfig['paths']['plot'] + '/'+hemis+'/'+batch_key+'/' + track_name + '/'
MT.mkdirs_r(plot_path)
MT.mkdirs_r(save_path)
bad_track_path =mconfig['paths']['work'] +'bad_tracks/'+ batch_key+'/'
# %%

all_beams   = mconfig['beams']['all_beams']
high_beams  = mconfig['beams']['high_beams']
low_beams   = mconfig['beams']['low_beams']
beam_groups = mconfig['beams']['groups']


#Gfilt   = io.load_pandas_table_dict(track_name + '_B01_regridded', load_path) # rhis is the rar photon data

load_path   = mconfig['paths']['work'] +batch_key+'/B01_regrid/'
#G_binned      = io.load_pandas_table_dict(track_name + '_B01_binned' , load_path)  #
G_binned_store = h5py.File(load_path +'/'+track_name + '_B01_binned.h5', 'r')
G_binned = dict()
for b in all_beams:

    G_binned[b]  = io.get_beam_hdf_store(G_binned_store[b])
G_binned_store.close()

load_path   = mconfig['paths']['work'] +batch_key+'/B02_spectra/'
Gx      = xr.load_dataset(load_path+ '/B02_'+track_name + '_gFT_x.nc' )  #
Gk      = xr.load_dataset(load_path+ '/B02_'+track_name + '_gFT_k.nc' )  #


# %% load prior information
load_path   = mconfig['paths']['work'] +batch_key+'/A02_prior/'
#track_name = '20190208104534_06410210_004_01'
try:
    Prior = MT.load_pandas_table_dict('/A02_'+track_name, load_path)['priors_hindcast']
except:
    print('Prior not founds exit')
    MT.json_save('B04_fail', plot_path,  {'time':time.asctime( time.localtime(time.time()) ) , 'reason': 'Prior not found'})
    print('exit()')
    exit()


#### Define Prior
# Use partitions
# Prior2              = Prior.loc[['ptp0','ptp1','ptp2','ptp3','ptp4','ptp5']]['mean']
# dominat_period      = Prior2[Prior2.max() ==Prior2]
# aa = Prior.loc[['pdp0','pdp1','pdp2','pdp3','pdp4','pdp5']]['mean'].astype('float')
# dominant_dir        = waves.get_ave_amp_angle(aa *0+1,aa  )[1]
# dominant_dir_spread = Prior.loc[['pspr0','pspr1','pspr2','pspr3','pspr4','pspr5']]['mean'].median()
#
# prior_sel= {'alpha': ( dominant_dir *np.pi/180 , dominant_dir_spread *np.pi/180) } # to radiens
#prior_sel= {'alpha': ( -60 *np.pi/180 , dominant_dir_spread *np.pi/180) } # to radiens

Pperiod     = Prior.loc[['ptp0','ptp1','ptp2','ptp3','ptp4','ptp5']]['mean']
Pdir        = Prior.loc[['pdp0','pdp1','pdp2','pdp3','pdp4','pdp5']]['mean'].astype('float')
Pspread     = Prior.loc[['pspr0','pspr1','pspr2','pspr3','pspr4','pspr5']]['mean']

Pperiod = Pperiod[  ~np.isnan(list(Pspread))]
Pdir    = Pdir[     ~np.isnan(list(Pspread))]
Pspread = Pspread[  ~np.isnan(list(Pspread))]


# reset dirs:
Pdir[Pdir > 180]    =  Pdir[Pdir > 180]  - 360
Pdir[Pdir < -180]   =  Pdir[Pdir < -180] + 360

# reorder dirs
dir_best = [0]
for dir in Pdir:
    ip = np.argmin([ abs(dir_best[-1] - dir), abs(dir_best[-1] - (dir - 360 )), abs(dir_best[-1] - (dir + 360 )) ] )
    new_dir = np.array([ dir, (dir - 360 ) , (dir + 360 ) ])[ip]
    dir_best.append(new_dir)
dir_best = np.array(dir_best[1:])

# %%

if len(Pperiod) == 0:
    print('constant peak wave number')
    kk              = Gk.k
    Pwavenumber     = kk*0 + (2 * np.pi / (1/ Prior.loc['fp']['mean'])  )**2 / 9.81
    dir_best        = kk*0 + Prior.loc['dp']['mean']
    #dir_interp      = np.interp(kk, Pwavenumber[Pwavenumber.argsort()] , dir_best[Pwavenumber.argsort()] )
    dir_interp_smth = dir_interp    = kk*0 + Prior.loc['dp']['mean']
    spread_smth     = spread_interp = kk*0 + Prior.loc['spr']['mean']
    #spread_interp   = np.interp(kk, Pwavenumber[Pwavenumber.argsort()] , Pspread[Pwavenumber.argsort()].astype('float')  )
    #spread_smth     = M.runningmean(spread_interp, 30, tailcopy= True)
    #spread_smth[-1] = spread_smth[-2]

else:
    Pwavenumber     = (2 * np.pi / Pperiod  )**2 / 9.81
    kk              = Gk.k
    dir_interp      = np.interp(kk, Pwavenumber[Pwavenumber.argsort()] , dir_best[Pwavenumber.argsort()] )
    dir_interp_smth = M.runningmean(dir_interp, 30, tailcopy= True)
    dir_interp_smth[-1] = dir_interp_smth[-2]

    spread_interp   = np.interp(kk, Pwavenumber[Pwavenumber.argsort()] , Pspread[Pwavenumber.argsort()].astype('float')  )
    spread_smth     = M.runningmean(spread_interp, 30, tailcopy= True)
    spread_smth[-1] = spread_smth[-2]


font_for_pres()

F = M.figure_axis_xy(5, 4.5, view_scale= 0.5)
plt.subplot(2, 1, 1)
plt.title('Prior angle smoothed\n'+ track_name, loc ='left')


plt.plot(  Pwavenumber , dir_best, '.r', markersize = 8)
plt.plot( kk , dir_interp, '-', color= 'red', linewidth = 0.8, zorder=11)
plt.plot( kk , dir_interp_smth  , color=col.green1)

plt.fill_between(kk, dir_interp_smth -spread_smth, dir_interp_smth +spread_smth, zorder= 1, color=col.green1, alpha = 0.2 )
plt.ylabel('Angle (deg)')
#plt.xlabel('wavenumber ($2 \pi/\lambda$)')

ax_final = plt.subplot(2, 1, 2)
plt.title('Prior angle adjusted ', loc ='left')

# adjust angle def:
dir_interp_smth[dir_interp_smth> 180] = dir_interp_smth[dir_interp_smth> 180]- 360
dir_interp_smth[dir_interp_smth< -180] = dir_interp_smth[dir_interp_smth< -180]+ 360

plt.fill_between(kk, dir_interp_smth -spread_smth, dir_interp_smth +spread_smth, zorder= 1, color=col.green1, alpha = 0.2 )
plt.plot( kk , dir_interp_smth , '.', markersize = 1 , color=col.green1)

ax_final.axhline(85, color='gray', linewidth= 2)
ax_final.axhline(-85, color='gray', linewidth= 2)

plt.ylabel('Angle (deg)')
plt.xlabel('wavenumber ($2 \pi/\lambda$)')

#F.save_light(path= plot_path, name = 'B04_prior_angle')


# save
dir_interp_smth = xr.DataArray(data=dir_interp_smth * np.pi/180 , dims='k', coords ={'k':kk}, name='Prior_direction')
spread_smth     = xr.DataArray(data=spread_smth* np.pi/180      , dims='k', coords ={'k':kk}, name='Prior_spread')
Prior_smth      = xr.merge([dir_interp_smth, spread_smth])


# Use fake
#prior_sel= {'alpha': ( 0.6 , dominant_dir_spread *np.pi/180) } # to radiens

# Use mean direction
#prior_sel= {'alpha': ( Prior.loc['dp']['mean'] *np.pi/180 , Prior.loc['spr']['mean'] *np.pi/180) }


# define paramater range
# params_dict = {'alpha': [  -0.85 * np.pi /2,     0.85 * np.pi /2,  5],
#                 'phase':[   0              , 2*np.pi            , 10]}
#
# alpha_dx        = 0.02
# max_wavenumbers = 25
#
# sample_flag     = True
# optimize_flag   = False
# brute_flag      = False
#
# plot_flag       = False
#
# Nworkers            = 6
# N_sample_chain      = 300
# N_sample_chain_burn = 30
#
max_x_pos = 8
x_pos_jump = 1


# isolate x positions with data
data_mask = Gk.gFT_PSD_data.mean('k')
data_mask.coords['beam_group'] = ('beam',  ['beam_group'+g[2] for g in data_mask.beam.data])
data_mask_group = data_mask.groupby('beam_group').mean(skipna=False)
# these stancils are actually used
data_sel_mask = data_mask_group.sum('beam_group') !=0

x_list  = data_sel_mask.x[data_sel_mask] # iterate over these x posistions
x_list_flag = ~np.isnan(data_mask_group.sel(x = x_list) )# flag that is False if there is no data

#### limit number of x coordinates

x_list = x_list#[::x_pos_jump]
#if len(x_list) > max_x_pos:
x_list = x_list[0:max_x_pos]
x_list_flag= x_list_flag.sel(x =x_list)

# plot
font_for_print()
F = M.figure_axis_xy(5.5, 3, view_scale= 0.8)
plt.suptitle(track_name)
ax1 =  plt.subplot(2, 1, 1)
plt.title('Data in Beam', loc= 'left')
plt.pcolormesh(data_mask.x/1e3, data_mask.beam, data_mask, cmap= plt.cm.OrRd)
for i in np.arange(1.5, 6, 2):
    ax1.axhline(i, color= 'black', linewidth =0.5)
plt.xlabel('Distance from Ice Edge')

ax_final = plt.subplot(2, 1, 2)
plt.title('Data in Group', loc= 'left')
plt.pcolormesh(data_mask.x/1e3, data_mask_group.beam_group, data_mask_group, cmap= plt.cm.OrRd)

for i in np.arange(0.5, 3, 1):
    ax_final.axhline(i, color= 'black', linewidth =0.5)

plt.plot( x_list/1e3, x_list*0 +0, '.', markersize= 2, color= col.cascade1 )
plt.plot( x_list/1e3, x_list*0 +1, '.', markersize= 2, color= col.cascade1 )
plt.plot( x_list/1e3, x_list*0 +2, '.', markersize= 2, color= col.cascade1 )

plt.xlabel('Distance from Ice Edge')

F.save_pup(path= plot_path, name = 'B04_data_avail')



# %% Load marginal distributions
MM = xr.open_dataset(save_path + save_name + '_marginals.nc')
LL = MT.load_pandas_table_dict(save_name+ '_res_table', save_path)['L_sample']

# plot

fn = copy.copy(lstrings)

font_for_print()
F = M.figure_axis_xy(fig_sizes['one_column_high'][0], fig_sizes['one_column_high'][1]*0.6, view_scale= 0.7, container = True)
gs = GridSpec(3,6,  wspace=0.5,  hspace=.8)#figure=fig,

ax1 = F.fig.add_subplot(gs[0:2 , 0:-2])

dd = MM.marginals
dd = dd.mean('beam_group').mean('x').rolling(k=20, min_periods= 1, center=True).mean()#.plot()

try:
    plt.pcolor(dd.angle *  180 /np.pi , dd.k , dd, cmap=col.cascade_r , zorder=0, vmin=0, vmax=5 )#np.arange(10, 37, 1) )
except:
    plt.pcolor(dd.angle *  180 /np.pi , dd.k , dd.T, cmap=col.cascade_r , zorder=0, vmin=0, vmax=5 )#np.arange(10, 37, 1) )

#klims = k_list.min()*0.2 , k_list.max()*1.2
klims = MM.k[~np.isnan(dd.mean('angle'))].min().data, MM.k[~np.isnan(dd.mean('angle'))].max().data# 0, LL['K_prime'].max()*1.2


dir_best[dir_best> 180] = dir_best[dir_best> 180] -360
plt.plot(dir_best,  Pwavenumber ,  '.k', markersize = 6)

dir_interp[dir_interp> 180] = dir_interp[dir_interp> 180] -360
plt.plot(dir_interp, Gk.k,  '-', color= 'k', linewidth = 1.5, zorder=1, alpha= 0.4)

# ax1.axvline(  best_guess * 180/ np.pi , color=col.blue, linewidth = 1.5, label ='best guess fitting')
# ax1.axvline(  (prior_sel['alpha'][0])  *  180 /np.pi, color='k', linewidth = 1.5, label ='prior')
# ax1.axvline(  (prior_sel['alpha'][0]- prior_sel['alpha'][1])  *  180 /np.pi, color='k', linewidth = 0.7, label ='prior uncertrainty')
# ax1.axvline(  (prior_sel['alpha'][0]+ prior_sel['alpha'][1]) *  180 /np.pi , color='k', linewidth = 0.7)

plt.fill_betweenx(Gk.k, (dir_interp_smth -spread_smth)* 180 /np.pi, (dir_interp_smth +spread_smth)* 180 /np.pi,  zorder= 1, color=col.orange, alpha = 0.2 )
plt.plot(dir_interp_smth * 180 /np.pi, Gk.k , '.', markersize = 1 , color=col.orange)

ax1.axvline(85, color='gray', linewidth= 1)
ax1.axvline(-85, color='gray', linewidth= 1)

#plt.legend()
plt.ylabel('wavenumber')
plt.xlabel('Angle (deg)')


plt.title(io.ID_to_str(track_name)+ '\n'+next(fn) +'marginal PDFs', loc='left')

#plt.xlim(- 170, 170)
#plt.xlim(- 90, 90)


#prior_angle_str =str(np.round( (prior_sel['alpha'][0])  *  180 /np.pi))
#plt.title(track_name + '\nprior=' + 'deg', loc= 'left' )
plt.xlim( min( [ -90,  np.nanmin(dir_best)] ) - 5,  max( [np.nanmax(dir_best), 90]) +5 )


ax0 = F.fig.add_subplot(gs[0:2 ,-2:])
ax0.tick_params(labelleft=False)



for g in MM.beam_group:
    MMi = MM.sel(beam_group= g)
    plt.plot(  MMi.weight.T,MMi.k,   '.', color= col_dict[str(g.data)], markersize= 2, alpha=0.5, linewidth = 0.8)

plt.title(next(fn) + 'weights', loc='left')
plt.xlabel('Power')
plt.ylim(klims)

F.save_pup(path= plot_path, name = 'B04_'+track_name+'_prior')
#MT.json_save('B04_success', plot_path, {'time':'time.asctime( time.localtime(time.time()) )'})

# %% plot


def derive_weights(weights):
    weights = (weights-weights.mean())/weights.std()
    weights = weights - weights.min()
    return weights


def weighted_means(data, weights, x_angle, color='k'):
    """
    weights should have nans when there is no data
    data should have zeros where there is no data
    """
    from scipy.ndimage.measurements import label
    # make wavenumber groups
    groups, Ngroups = label(weights.where(~np.isnan(weights), 0)  )

    for ng in np.arange(1, Ngroups+1):
        wi          = weights[groups == ng]
        weight_norm = weights.sum('k')
        k           = wi.k.data
        data_k      = data.sel(k=k).squeeze()
        data_weight = (data_k * wi)
        #plt.stairs(data_weight.sum('k')/ weight_norm , x_angle, linewidth=1 , color ='k')
        # if data_k.k.size > 1:
        #     for k in data_k.k.data:
        #         plt.stairs(data_weight.sel(k=k) / weight_norm, x_angle, color ='gray', alpha =0.5)
        # else:
        #     plt.stairs(data_weight.squeeze() / weight_norm, x_angle, color ='gray', alpha =0.5)


    data_weighted_mean = (data.where( (~np.isnan(data)) & (data != 0), np.nan) * weights ).sum('k')/weight_norm
    return data_weighted_mean


xticks_pi       =  np.arange(-np.pi/2, np.pi/2+np.pi/4, np.pi/4)
xtick_labels_pi = ['-$\pi/2$','-$\pi/4$','0','$\pi/4$','$\pi/2$',]

fn = copy.copy(lstrings)

font_for_print()
F = M.figure_axis_xy(fig_sizes['one_column_high'][0], fig_sizes['one_column_high'][1]*0.7, view_scale= 0.7, container = True)
gs = GridSpec(3,6,  wspace=0.5,  hspace=.6)#figure=fig,

ax1 = F.fig.add_subplot(gs[0:2 , :])

angle_mask = MM.angle *0 ==0
angle_mask[0], angle_mask[-1] = False, False
corrected_marginals = MM.marginals.isel(angle=angle_mask ) + MM.marginals.isel(angle=~angle_mask ).sum('angle')/sum(angle_mask).data


#xi = MM.x[3]
xi = MM.x[3]

dd = corrected_marginals.mean('beam_group').sel(x=xi).rolling(k=20, min_periods= 1, center=True).mean()#.plot()

try:
    plt.pcolor(dd.angle , dd.k , dd, cmap=col.cascade_r , zorder=0, vmin=0, vmax=5 )#np.arange(10, 37, 1) )
except:
    plt.pcolor(dd.angle , dd.k , dd.T, cmap=col.cascade_r , zorder=0, vmin=0, vmax=5 )#np.arange(10, 37, 1) )

#klims = k_list.min()*0.2 , k_list.max()*1.2
klims = MM.k[~np.isnan(dd.mean('angle'))].min().data * 0.8,  MM.k[~np.isnan(dd.mean('angle'))].max().data * 1.1# 0, LL['K_prime'].max()*1.2


dir_best[dir_best> 180] = dir_best[dir_best> 180] -360
plt.plot(dir_best * np.pi/180,  Pwavenumber ,  '.k', markersize = 6)

dir_interp[dir_interp> 180] = dir_interp[dir_interp> 180] -360
plt.plot(dir_interp * np.pi/180, Gk.k,  '-', color= 'k', linewidth = 1.5, zorder=1, alpha= 0.4)

# ax1.axvline(  best_guess * 180/ np.pi , color=col.blue, linewidth = 1.5, label ='best guess fitting')
# ax1.axvline(  (prior_sel['alpha'][0])  *  180 /np.pi, color='k', linewidth = 1.5, label ='prior')
# ax1.axvline(  (prior_sel['alpha'][0]- prior_sel['alpha'][1])  *  180 /np.pi, color='k', linewidth = 0.7, label ='prior uncertrainty')
# ax1.axvline(  (prior_sel['alpha'][0]+ prior_sel['alpha'][1]) *  180 /np.pi , color='k', linewidth = 0.7)

plt.fill_betweenx(Gk.k, (dir_interp_smth -spread_smth) , (dir_interp_smth +spread_smth) ,  zorder= 1, color=col.orange, alpha = 0.2 )
plt.plot(dir_interp_smth  , Gk.k , '.', markersize = 1 , color=col.orange)



x_str= str(int(xi.data/1e3))
plt.title( 'Example Marginal PDF' + '\n'  + io.ID_to_str(track_name)+ '\nat $X_i$='+ x_str +' km '+ '\n'+next(fn) +'mean marginal PDFs per wavenumber', loc='left')

#plt.xlim(- 170, 170)
#plt.xlim(- 90, 90)
plt.ylim(klims)

#prior_angle_str =str(np.round( (prior_sel['alpha'][0])  *  180 /np.pi))
#plt.title(track_name + '\nprior=' + 'deg', loc= 'left' )
plt.xlim( min( [ -np.pi/2,  np.nanmin(dir_best) * np.pi/180 ] ) - 5* np.pi/180,  max( [np.nanmax(dir_best) * np.pi/180 , np.pi/2]) +5* np.pi/180 )


ax_final = F.fig.add_subplot(gs[2: ,:])
#ax_final.tick_params(labelleft=False)


ax_list= dict()

Gweights = corrected_marginals.N_data
Gweights = Gweights/Gweights.max()
group_weight = Gweights.sel(x =xi)

data_collect = dict()
for group, gpos in zip(MM.beam_group.data, [ gs[0, 0], gs[1, 0], gs[2, 0]] ):

    data    = corrected_marginals.sel(x=xi).sel(beam_group= group)
    weights = derive_weights( MM.weight.sel(x=xi).sel(beam_group= group)  )
    weights = weights**2

    # derive angle axis
    x_angle = data.angle.data  #*  180 /np.pi
    d_angle = np.diff(x_angle)[0]
    x_angle = np.insert(x_angle, x_angle.size , x_angle[-1].data +  d_angle)

    if ( (~np.isnan(data)).sum().data == 0) | (( ~np.isnan(weights)).sum().data == 0):
        data_wmean = data.mean('k')
    else:
        data_wmean = weighted_means(data, weights, x_angle, color= col_dict[group] )
        #plt.stairs(data_wmean , x_angle, color =col_dict[group], alpha =1)

    data_collect[group] = data_wmean

data_collect = xr.concat(data_collect.values(), dim='beam_group')
final_data   = (group_weight * data_collect).sum('beam_group')  /  group_weight.sum('beam_group').data
final_data =final_data / final_data.sum() / np.diff(x_angle)[0]


#ax_final = F.fig.add_subplot(gs[-1, :])
plt.title(next(fn) + 'Final weighted marginal PDF', loc='left')

k_mask = corrected_marginals.mean('beam_group').mean('angle')

#priors_k = MM.Prior_direction[ ~np.isnan(k_mask.sel(x= xi))]
priors_k = dir_interp_smth[ ~np.isnan(k_mask.sel(x= xi))]
for pk in priors_k:
    ax_final.axvline(pk, color =col.orange, linewidth= 1, alpha = 0.5)

plt.stairs( final_data , x_angle, color = 'k', alpha =0.5, linewidth =0.8, zorder= 12)

final_data_smth = lanczos.lanczos_filter_1d(x_angle,final_data, 0.1)
#
# for group in MM.beam_group.data:
#     plt.stairs( data_collect.sel(beam_group= group) * group_weight.sel(beam_group= group) /group_weight.sum('beam_group').data, x_angle, color =col_dict[group], alpha =1)

plt.plot(x_angle[0:-1], final_data_smth, color = 'black', linewidth= 0.8)

ax_final.axvline( x_angle[0:-1][final_data_smth.argmax()] , color =col.cascade3, linewidth= 1.5, alpha = 1, zorder= 1)
ax_final.axvline( x_angle[0:-1][final_data_smth.argmax()] , color =col.black, linewidth= 3, alpha = 1, zorder= 0)

ax_final.set_xlabel('Angle')
ax_final.set_ylabel('Density')

ax_final.set_xlim(  - 1.1 * np.pi/2 , 1.3 *  np.pi/2)
ax1.set_xlim(       - 1.1 * np.pi/2 , 1.3 *  np.pi/2)
ax1.set_xticks(xticks_pi)


k_l = M.cut_nparray(dd.k.data, klims[0], klims[1] )

y_tickslabel, y_ticks = MT.tick_formatter(dd.k.data[k_l][::15]*10, interval= 3, rounder= 2, expt_flag= False)

y_ticks = y_ticks/10
ax1.set_yticks( y_ticks )
ax1.set_yticklabels(y_tickslabel)
#plt.ylim(klims)

ax1.set_xticklabels(xtick_labels_pi)
ax1.set_xticklabels([])

ax_final.set_xticks(xticks_pi)
ax_final.set_xticklabels(xtick_labels_pi)

ax1.axvline(85 * np.pi/180, color='gray', linewidth= 1)
ax1.axvline(-85 * np.pi/180, color='gray', linewidth= 1)


ax_final.axvline(85 * np.pi/180, color='gray', linewidth= 1)
ax_final.axvline(-85 * np.pi/180, color='gray', linewidth= 1)

#plt.legend()
ax1.set_ylabel('Wavenumber')
#ax1.set_xlabel('Angle (deg)')


dlim= 0, final_data_smth.max()* 1.2
density_labels, density_ticks  = MT.tick_formatter( np.arange(dlim[0], dlim[1], 0.5), interval= 2, rounder= 2, expt_flag= False)
ax_final.set_yticks(density_ticks)
ax_final.set_yticklabels(density_labels)
plt.ylim(dlim)
F.save_pup(path= plot_path, name = 'B04_'+track_name+'_prior_merged')

# %%
