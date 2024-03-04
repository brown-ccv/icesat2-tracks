
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

import imp
import copy
import spicke_remover
import datetime
import concurrent.futures as futures

from numba import jit

from ICEsat2_SI_tools import angle_optimizer
import ICEsat2_SI_tools.wave_tools as waves
import concurrent.futures as futures

import time

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


#track_name, batch_key, test_flag = '20190502021224_05160312_004_01', 'SH_batch02', False
track_name, batch_key, test_flag = '20190213133330_07190212_004_01', 'SH_batch02', False
#20190213133330_07190212_004_01

#print(track_name, batch_key, test_flag)
hemis, batch = batch_key.split('_')
#track_name= '20190605061807_10380310_004_01'
ATlevel= 'ATL03'

save_path   = mconfig['paths']['work'] + '/B04_angle_'+hemis+'/'
save_name   = 'B04_'+track_name

plot_path   = mconfig['paths']['plot'] + '/'+hemis+'/'+batch_key+'/publish/' + track_name + '/'
MT.mkdirs_r(plot_path)
MT.mkdirs_r(save_path)
bad_track_path =mconfig['paths']['work'] +'bad_tracks/'+ batch_key+'/'
# %%

all_beams   = mconfig['beams']['all_beams']
high_beams  = mconfig['beams']['high_beams']
low_beams   = mconfig['beams']['low_beams']
beam_groups = mconfig['beams']['groups']

#Gfilt   = io.load_pandas_table_dict(track_name + '_B01_regridded', load_path) # rhis is the rar photon data

load_path   = mconfig['paths']['work'] +'/B01_regrid_'+hemis+'/'
G_binned      = io.load_pandas_table_dict(track_name + '_B01_binned' , load_path)  #

load_path   = mconfig['paths']['work'] +'/B02_spectra_'+hemis+'/'
Gx      = xr.load_dataset(load_path+ '/B02_'+track_name + '_gFT_x.nc' )  #
Gk      = xr.load_dataset(load_path+ '/B02_'+track_name + '_gFT_k.nc' )  #


# %% load prior information
load_path   = mconfig['paths']['work'] +'/A02_prior_'+hemis+'/'
#track_name = '20190208104534_06410210_004_01'
try:
    Prior = MT.load_pandas_table_dict('/A02b_'+track_name, load_path)['priors_hindcast']
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

Pperiod = Pperiod[~np.isnan(list(Pspread))]
Pdir    = Pdir[~np.isnan(list(Pspread))]
Pspread = Pspread[~np.isnan(list(Pspread))]


# reset dirs:
Pdir[Pdir > 180]    =  Pdir[Pdir > 180] - 360
Pdir[Pdir < -180]   =  Pdir[Pdir < -180] + 360

# reorder dirs
dir_best = [0]
for dir in Pdir:
    ip = np.argmin([ abs(dir_best[-1] - dir), abs(dir_best[-1] - (dir - 360 )), abs(dir_best[-1] - (dir + 360 )) ] )
    new_dir = np.array([ dir, (dir - 360 ) , (dir + 360 ) ])[ip]
    dir_best.append(new_dir)
dir_best = np.array(dir_best[1:])

# %%

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

ax2 = plt.subplot(2, 1, 2)
plt.title('Prior angle adjusted ', loc ='left')

# adjust angle def:
dir_interp_smth[dir_interp_smth> 180] = dir_interp_smth[dir_interp_smth> 180]- 360
dir_interp_smth[dir_interp_smth< -180] = dir_interp_smth[dir_interp_smth< -180]+ 360

plt.fill_between(kk, dir_interp_smth -spread_smth, dir_interp_smth +spread_smth, zorder= 1, color=col.green1, alpha = 0.2 )
plt.plot( kk , dir_interp_smth , '.', markersize = 1 , color=col.green1)

ax2.axhline(85, color='gray', linewidth= 2)
ax2.axhline(-85, color='gray', linewidth= 2)

plt.ylabel('Angle (deg)')
plt.xlabel('wavenumber ($2 \pi/\lambda$)')

F.save_light(path= plot_path, name = 'B04_prior_angle')


# save
dir_interp_smth = xr.DataArray(data=dir_interp_smth * np.pi/180 , dims='k', coords ={'k':kk}, name='Prior_direction')
spread_smth     = xr.DataArray(data=spread_smth* np.pi/180      , dims='k', coords ={'k':kk}, name='Prior_spread')
Prior_smth      = xr.merge([dir_interp_smth, spread_smth])

# %%
prior_angle =Prior_smth.Prior_direction * 180/np.pi
if (abs(prior_angle) > 80).all():
    print('Prior angle is ', prior_angle.min().data,   prior_angle.max().data, '. quit.')
    dd_save = {'time' : time.asctime( time.localtime(time.time()) ),
     'angle': list([ float(prior_angle.min().data),   float(prior_angle.max().data), float(prior_angle.median()) ]) }
    MT.json_save('B04_fail', plot_path, dd_save)
    print('exit()')
    #exit()

# Use fake
#prior_sel= {'alpha': ( 0.6 , dominant_dir_spread *np.pi/180) } # to radiens

# Use mean direction
#prior_sel= {'alpha': ( Prior.loc['dp']['mean'] *np.pi/180 , Prior.loc['spr']['mean'] *np.pi/180) }


# define paramater range
params_dict = {'alpha': [  -0.85 * np.pi /2,     0.85 * np.pi /2,  5],
                'phase':[   0              , 2*np.pi            , 10]}

alpha_dx        = 0.04
max_wavenumbers = 25

sample_flag     = True
optimize_flag   = False
brute_flag      = False

plot_flag       = True

Nworkers            = 6
N_sample_chain      = 300
N_sample_chain_burn = 30

max_x_pos = 8
x_pos_jump = 2

def make_fake_data(xi,group ):
    ki= Gk.k[0:2]

    bins = np.arange(params_dict['alpha'][0], params_dict['alpha'][1]+alpha_dx,  alpha_dx)
    bins_pos = (bins[0:-1] + np.diff(bins)/2)
    marginal_stack = xr.DataArray( np.nan* np.vstack([bins_pos, bins_pos]).T, dims= ('angle', 'k'),  coords = {'angle':bins_pos, 'k':ki.data } )

    group_name  = str('group' + group[0].split('gt')[1].split('l')[0])
    marginal_stack.coords['beam_group'] = group_name
    marginal_stack.coords['x']          = xi
    marginal_stack.name                 = 'marginals'
    marginal_stack.expand_dims(dim = 'x', axis = 2).expand_dims(dim = 'beam_group', axis = 3)
    return marginal_stack

def define_wavenumber_weights_tot_var(dd, m = 3, variance_frac = 0.33, k_upper_lim= None, verbose=False):

    """
    return peaks of a power spectrum dd that in the format such that they can be used as weights for the frequencies based fitting

    inputs:
    dd             xarray with PSD as data amd coordindate wavenumber k
    m               running mean half-width in gridpoints
    variance_frac  (0 to 1) How much variance should be explained by the returned peaks
    verbose        if true it plots some stuff


    return:
    mask           size of dd. where True the data is identified as having significant amplitude
    k              wanumbers where mask is true
    dd_rm          smoothed version of dd
    positions      postions where of significant data in array
    """

    if len(dd.shape) == 2:
        dd_use = dd.mean('beam')

    if m is None:
        dd_rm   = dd_use.data#M.runningmean(dd, m, tailcopy=True)
    else:
        dd_rm   = M.runningmean(dd_use, m, tailcopy=True)

    k      =  dd_use.k[~np.isnan(dd_rm)].data
    dd_rm   =  dd_rm[~np.isnan(dd_rm)]

    orders = dd_rm.argsort()[::-1]
    var_mask = dd_rm[orders].cumsum()/dd_rm.sum() < variance_frac
    pos_cumsum = orders[var_mask]
    #k_list  = k[pos_cumsum]
    #dd_list = dd_rm[pos_cumsum]
    mask = var_mask[orders.argsort()]
    if k_upper_lim is not None:
        mask = (k < k_upper_lim) & mask

    if verbose:

        plt.plot(dd.k, dd, '-', color = col_dict[str(amp_data.beam[0].data)], markersize= 20, alpha = 0.6)
        plt.plot(k, dd_rm, '-k', markersize= 20)
        #print(k_list, dd_list)
        plt.plot(k[mask], dd_rm[mask], '.r', markersize= 10, zorder=12)
        if k_upper_lim is not None:
            plt.gca().axvline(k_upper_lim, color= 'black')

    return mask, k, dd_rm, pos_cumsum

def define_wavenumber_weights_threshold(dd, m = 3, Nstd= 2, verbose=False):

    if m is None:
        dd_rm   = dd#M.runningmean(dd, m, tailcopy=True)
    else:
        dd_rm   = M.runningmean(dd, m, tailcopy=True)

    k      =  dd.k[~np.isnan(dd_rm)]
    dd_rm   =  dd_rm[~np.isnan(dd_rm)]

    treshold    = np.nanmean(dd_rm)  + np.nanstd(dd_rm) *Nstd
    mask        = dd_rm > treshold


    if verbose:
        plt.plot(dd.k, dd, '-k', markersize= 20)
        plt.plot(k, dd_rm, '-b', markersize= 20)

        k_list      = k[mask]
        dd_list     = dd_rm[mask]
        #print(k_list, dd_list)
        plt.plot(k_list, dd_list, '.r', markersize= 10, zorder=12)

    return mask, k, dd_rm, np.arange(0, mask.size)[mask]



# isolate x positions with data
data_mask = Gk.gFT_PSD_data.mean('k')
data_mask.coords['beam_group'] = ('beam',  ['beam_group'+g[2] for g in data_mask.beam.data])
data_mask_group = data_mask.groupby('beam_group').mean(skipna=False)
# these stancils are actually used
data_sel_mask = data_mask_group.sum('beam_group') !=0

x_list  = data_sel_mask.x[data_sel_mask] # iterate over these x posistions
x_list_flag = ~np.isnan(data_mask_group.sel(x = x_list) )# flag that is False if there is no data

#### limit number of x coordinates

x_list = x_list[::x_pos_jump]
if len(x_list) > max_x_pos:
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

ax2 = plt.subplot(2, 1, 2)
plt.title('Data in Group', loc= 'left')
plt.pcolormesh(data_mask.x/1e3, data_mask_group.beam_group, data_mask_group, cmap= plt.cm.OrRd)

for i in np.arange(0.5, 3, 1):
    ax2.axhline(i, color= 'black', linewidth =0.5)

plt.plot( x_list/1e3, x_list*0 +0, '.', markersize= 2, color= col.cascade1 )
plt.plot( x_list/1e3, x_list*0 +1, '.', markersize= 2, color= col.cascade1 )
plt.plot( x_list/1e3, x_list*0 +2, '.', markersize= 2, color= col.cascade1 )

plt.xlabel('Distance from Ice Edge')

F.save_pup(path= plot_path, name = 'B04_data_avail')


# %%
Marginals = dict()
L_collect = dict()
marginal_stack = dict()

group_number = np.arange(len(beam_groups))
ggg, xxx = np.meshgrid(group_number , x_list.data[2:3]  )

for gi in zip(ggg.flatten(), xxx.flatten()):
    print(gi)

    group, xi = beam_groups[gi[0]], gi[1]

    if bool(x_list_flag.sel(x= xi).isel(beam_group= gi[0]).data) is False:
        print('no data, fill with dummy')
        ikey            = str(xi) +'_' +  '_'.join(group)
        Marginals[ikey] = make_fake_data(xi, group)
        #print(Marginals[ikey].angle.data[::20])
        continue

    GGx = Gx.sel(beam= group).sel(x = xi)
    GGk = Gk.sel(beam= group).sel(x = xi)

    ### define data
    # normalize data
    key = 'y_data'
    amp_Z =  (GGx[key] - GGx[key].mean(['eta'])) /GGx[key].std(['eta'])
    N = amp_Z.shape[0]

    # define x,y positions
    eta_2d = GGx.eta + GGx.x_coord - GGx.x_coord.mean()
    nu_2d = GGx.eta * 0 + GGx.y_coord - GGx.y_coord.mean()

    # repack as np arrays
    x_concat = eta_2d.data.T.flatten()
    y_concat = nu_2d.data.T.flatten()
    z_concat = amp_Z.data.flatten()

    x_concat= x_concat[~np.isnan(z_concat)]
    y_concat= y_concat[~np.isnan(z_concat)]
    z_concat= z_concat[~np.isnan(z_concat)]
    N_data = x_concat.size

    if np.isnan(z_concat).sum() != 0:
        raise ValueError('There are still nans')

    mean_dist    =  (nu_2d.isel(beam= 0) - nu_2d.isel(beam= 1)).mean().data
    k_upper_lim  =  2 *np.pi / ( mean_dist *1 )

    print('k_upper_lim ', k_upper_lim)
    # threshold method
    #mask, k, weights, positions =  define_wavenumber_weights_threshold( Gi.mean('dist_y')['gFT_PSD_data'], 3 , verbose= True)
    #plt.plot(k[mask], weights[mask], 'g*', markersize=20)
    # plt.show()

    #variance method
    amp_data = np.sqrt(GGk.gFT_cos_coeff**2 + GGk.gFT_sin_coeff**2)
    mask, k, weights, positions = define_wavenumber_weights_tot_var(amp_data, m= 1, k_upper_lim= k_upper_lim, variance_frac = 0.20 , verbose= False)
    #plt.xlim( k[mask].min()*0.8 ,max(k_upper_lim,  k[mask].max()*1.2) )
    #plt.xlim( k[mask].min()*0.8 ,k[mask].max()*1.4 )
    #plt.show()

    if (len(k[mask]) ==0):
        print('no good k found, fill with dummy')
        ikey            = str(xi) +'_' +  '_'.join(group)
        Marginals[ikey] = make_fake_data(xi, group)
        continue


    k_list, weight_list =  k[mask], weights[mask]
    print('# of wavenumber: ' , len(k_list))

    #### prepare loop
    #imp.reload(angle_optimizer)

    SM = angle_optimizer.sample_with_mcmc(params_dict)
    SM.set_objective_func(angle_optimizer.objective_func)

    SM.fitting_args = fitting_args = (x_concat, y_concat, z_concat)


    # test:
    k_prime_max= 0.02 #[mask][0] # chose a test wavenumber
    amp_Z= 1
    prior_sel= {'alpha': ( Prior_smth.sel(k =k_prime_max, method='nearest').Prior_direction.data,
                         Prior_smth.sel(k =k_prime_max, method='nearest').Prior_spread.data) }
    SM.fitting_kargs = fitting_kargs = {'prior': prior_sel , 'prior_weight' : 3 }
    # test if it works
    SM.params.add('K_prime', k_prime_max ,  vary=False  , min=k_prime_max*0.5, max=k_prime_max*1.5)
    SM.params.add('K_amp', amp_Z         ,  vary=False  , min=amp_Z*.0       , max=amp_Z*5)
    try:
        SM.test_objective_func()
    except:
        raise ValueError('Objective function test fails')

    #k_prime_max, Z_max = k_list[-1], weight_list[-1]
    pk= 0
    for k_prime_max, Z_max in zip(k_list, weight_list):


        brute_flag = True
        prior_sel= {'alpha': ( Prior_smth.sel(k =k_prime_max, method='nearest').Prior_direction.data,
                             Prior_smth.sel(k =k_prime_max, method='nearest').Prior_spread.data) }

        SM = angle_optimizer.sample_with_mcmc(params_dict)
        SM.set_objective_func(angle_optimizer.objective_func)

        SM.fitting_args = fitting_args = (x_concat, y_concat, z_concat)
        #print(prior_sel)
        SM.fitting_kargs = fitting_kargs = {'prior': prior_sel , 'prior_weight' : 3 }
        #SM.fitting_kargs = fitting_kargs = {'prior': None , 'prior_weight' : 1 }

        amp_Z = 1##z_concat.var()#Z_max#0.5 #amp_enhancement * abs(Z_max)**2 /N

        SM.params.add('K_prime', k_prime_max ,  vary=False  , min=k_prime_max*0.5, max=k_prime_max*1.5)
        SM.params.add('K_amp'  , amp_Z       ,  vary=False  , min=amp_Z*.0       , max=amp_Z*5)
        #print(SM.params.pretty_print())

        with suppress_stdout():
            SM.sample(verbose= False, steps=N_sample_chain,progress= False, workers= None)
        SM.optimize(verbose= False)
        SM.brute(verbose= False)

        y_hist, bins, bins_pos = SM.get_marginal_dist('alpha', alpha_dx, burn = N_sample_chain_burn, plot_flag= False)
        fitter = SM.fitter # MCMC results
        z_model = SM.objective_func(fitter.params, *fitting_args , test_flag= True)
        #cost = (fitter.residual**2).sum()/(z_concat**2).sum()
        #cost_list.append( (fitter.residual**2).sum()/(z_concat**2).sum() )
        marginal_stack_i = xr.DataArray( y_hist, dims= ('angle'),  coords = {'angle':bins_pos } )
        marginal_stack_i.coords['k']  = np.array(k_prime_max) #( ('k'), np.array(k_prime_max) )
        marginal_stack[k_prime_max] = marginal_stack_i
        #marginal_stack_i.coords['weight'] = Z_max

        # no prior:
        SM_nop = angle_optimizer.sample_with_mcmc(params_dict)
        SM_nop.set_objective_func(angle_optimizer.objective_func)

        SM_nop.fitting_args = fitting_args = (x_concat, y_concat, z_concat)

        brute_flag = True
        #print(prior_sel)
        SM_nop.fitting_kargs = {'prior': None , 'prior_weight' : 0 }
        #SM.fitting_kargs = fitting_kargs = {'prior': None , 'prior_weight' : 1 }
        amp_Z = 1##z_concat.var()#Z_max#0.5 #amp_enhancement * abs(Z_max)**2 /N

        SM_nop.params.add('K_prime', k_prime_max ,  vary=False  , min=k_prime_max*0.5, max=k_prime_max*1.5)
        SM_nop.params.add('K_amp'  , amp_Z       ,  vary=False  , min=amp_Z*.0       , max=amp_Z*5)
        #print(SM.params.pretty_print())
        try:
            SM_nop.test_objective_func()
        except:
            raise ValueError('Objective function test fails')

        with suppress_stdout():
            SM_nop.sample(verbose= False, steps=N_sample_chain,progress= False, workers= None)
        SM_nop.optimize(verbose= False)
        SM_nop.brute(verbose= False)

        y_hist_nop, bins_nop, bins_pos_nop = SM_nop.get_marginal_dist('alpha', alpha_dx, burn = N_sample_chain_burn, plot_flag= False)
        fitter_nop = SM_nop.fitter # MCMC results
        z_model_nop = SM.objective_func(fitter_nop.params, *fitting_args , test_flag= True)


        z_model, fargs , key  = z_model, fitting_args, 'y_data_normed'
        brute = brute_flag
        optimze= optimize_flag
        sample= sample_flag

        view_scale = 0.6

        # def plot_instance(z_model, fargs , key, SM, non_dim=False, title_str= None , brute=False, optimze= False, sample= False,  view_scale = 0.3):
        lstrings =iter([i+') ' for i in list(string.ascii_lowercase)])
        import copy

        brute_clevel = np.linspace(-3.05, 3, 30)
        font_for_print()
        fn = copy.copy(lstrings)
        x_concat, y_concat, z_concat = fargs
        F = M.figure_axis_xy(fig_sizes['23rd_width_high'][0],fig_sizes['23rd_width_high'][1], view_scale = view_scale, container = True)
        title_str =  'Incident Angle Sampling\nmodel wavelength='+ str(np.round(2 * np.pi/k_prime_max, 1)) + 'm'

        plt.suptitle(title_str, y = 0.95, ha='left', x=0.13)
        gs = GridSpec(12, 10,  wspace=0.3,  hspace=10)#figure=fig,
        F.gs = gs
        # y_offset= 0.5
        # plt.plot(Gm.eta,   Gm.y_model_normed+y_offset * Gm.dist_y/np.diff(Gm.dist_y), **model_plot_karg)
        # plt.plot(Gm.eta,   Gm.y_model_normed+y_offset * Gm.dist_y/np.diff(Gm.dist_y), **data_plot_karg)
        # plt.xlim(-1000, 1000)

        beam_list = list(set(y_concat))

        F.ax1 = F.fig.add_subplot(gs[0:2, :])
        y_pos, bcol = beam_list[0], col.rels['gt2l']
        #plt.title( str(y_pos) )
        plt.plot(x_concat[y_concat == y_pos]/1e3, z_concat[y_concat == y_pos] ,  c=bcol, linewidth = 1)
        plt.plot(x_concat[y_concat == y_pos]/1e3, z_model[y_concat == y_pos] , '-', c='black', linewidth= 0.6)
        #plt.xlim(x_concat[y_concat == y_pos][0]/1e3, x_concat[y_concat == y_pos][-1])
        #plt.xlim(-1000/1e3, 1000/1e3)
        plt.ylabel('Slope (m/m)')
        plt.title(next(fn) +'Beam Pair', loc='left')

        F.ax2 = F.fig.add_subplot(gs[1:3, :])
        y_pos, bcol = beam_list[1], col.rels['gt2r']
        #plt.title( str(y_pos) )
        plt.plot(x_concat[y_concat == y_pos]/1e3, z_concat[y_concat == y_pos] ,  c=bcol, linewidth = 1)
        plt.plot(x_concat[y_concat == y_pos]/1e3, z_model[y_concat == y_pos] , '-', c='black', linewidth= 0.6)
        #plt.xlim(x_concat[y_concat == y_pos][0]/1e3, x_concat[y_concat == y_pos][-1])


        #F.ax1.spines['left'].set_position(('outward', 5))
        #F.ax2.spines['right'].set_position(('outward', 5))
        #F.ax2.set_ylabel('Slope (m/m)')

        y_ticks =  MT.tick_formatter(   np.arange(-10, 12, 1),  interval=2, rounder=1, expt_flag=False, shift=0 )
        F.ax1.set_yticks(y_ticks[1])
        F.ax1.set_yticklabels(y_ticks[0])
        F.ax2.set_yticks(y_ticks[1])
        F.ax2.set_yticklabels(y_ticks[0])
        F.ax1.set_ylim(-3, 3)
        F.ax2.set_ylim(-3, 3)

        F.ax1.tick_params(bottom=False, labelbottom= False)
        F.ax1.spines['bottom'].set_visible(False)
        #F.ax2.xaxis.set_ticks_position('top')
        #F.ax2.xaxis.set_label_position('top')
        F.ax1.spines['bottom'].set_linewidth(0.5)
        F.ax1.spines['left'].set_linewidth(0.5)
        #F.ax1.xaxis.set_ticks_position('bottom')
        #F.ax1.xaxis.set_label_position('bottom')


        F.ax2.tick_params(left= False, labelleft=False, right= True, labelright= True)
        F.ax2.spines['right'].set_visible(True)
        F.ax2.spines['left'].set_visible(False)
        F.ax2.spines['bottom'].set_visible(True)
        F.ax2.set_facecolor((1.0, 1.00, 1.00, 0))

        F.ax1.axhline(0, color='gray', linewidth = 0.5, alpha = 1)
        F.ax2.axhline(0, color='gray', linewidth = 0.5, alpha = 1)

        x_ticks =  MT.tick_formatter(   np.arange(-10, 10, 0.5),  interval=2, rounder=1, expt_flag=False, shift=0 )
        F.ax1.set_xticks(x_ticks[1])
        F.ax1.set_xticklabels(x_ticks[0])
        F.ax1.set_xlim(-2, 2)

        F.ax2.set_xticks(x_ticks[1])
        F.ax2.set_xticklabels(x_ticks[0])
        F.ax2.set_xlim(-1.5, 1.5)
        plt.xlabel('km')


        # with prior
        F.ax5 = F.fig.add_subplot(gs[3:8, 0:-2])
        F.ax5.tick_params(bottom= False, labelbottom=False)
        F.ax5.spines['bottom'].set_visible(False)

        if brute is True:
            plt.title(next(fn) +'Sample Visualization', loc='left')
            #plt.title('with Prior', loc='left')
            SM.plot_brute(clevel =  brute_clevel , marker= '.', color ='blue', markersize=15, label= 'Brute', zorder=10)
            plt.plot(SM.fitter_brute.brute_x0[1], SM.fitter_brute.brute_x0[0], '.',  color ='red', markersize=5, zorder=10, label='best fit')

            # plt.colorbar(orientation='horizontal')
        if optimze is True:
            SM.plot_optimze(color= 'r', markersize=10, zorder=12, label= 'Dual Annealing')

        if sample is True:
            SM.plot_sample(markersize= 2, linewidth= 0.8, alpha= 0.2, color= 'black', zorder=8)


        if (fitting_kargs['prior'] is not None):
            F.ax5.axhline(prior_sel['alpha'][0], color='orange', linewidth = 2, label ='Prior')
            F.ax5.axhspan(prior_sel['alpha'][0]- prior_sel['alpha'][1], prior_sel['alpha'][0]+ prior_sel['alpha'][1], color='orange', alpha=0.2)
            F.ax5.axhline(prior_sel['alpha'][0]- prior_sel['alpha'][1], color='orange', linewidth = 0.7)
            F.ax5.axhline(prior_sel['alpha'][0]+ prior_sel['alpha'][1], color='orange', linewidth = 0.7)

        F.ax5.axhline(fitter.params['alpha'].min, color='black', linewidth = 0.6, alpha = 1)
        F.ax5.axhline(fitter.params['alpha'].max, color='black', linewidth = 0.6, alpha = 1)

        F.ax5.set_yticks([-np.pi/2,   0, np.pi/2])
        F.ax5.set_yticklabels(['$-\pi$' ,'0', '$\pi$'])

        # F.ax5.set_yticks(deg_ticks)
        # F.ax5.set_yticklabels(deg_tick_label)

        plt.sca(F.ax5)
        plt.legend(loc= 1)
        plt.xlabel('')
        plt.ylabel('Angle of Incidence')
        plt.xlim(0, np.pi*2)

        F.ax51 = F.fig.add_subplot(gs[3:8, -2:])
        F.ax51.tick_params(left= False, labelleft=False, labelbottom= False, bottom= False)#, right= True, labelright= True)
        F.ax51.spines['bottom'].set_visible(False)

        plt.title(next(fn) +'Marginal', loc= 'left')
        #plt.xlabel('Density')
        plt.stairs(y_hist, bins, orientation='horizontal', color=col.rels['group'+str(gi[0]+1)])

        F.ax51.axhline(fitter.params['alpha'].min, color='black', linewidth = 0.6, alpha = 1)
        F.ax51.axhline(fitter.params['alpha'].max, color='black', linewidth = 0.6, alpha = 1)

        F.ax5.set_ylim(-np.pi/1.5, np.pi/1.5)
        F.ax51.set_ylim(-np.pi/1.5, np.pi/1.5)

        # F.ax5.set_ylim(min( -np.pi /2, prior_sel['alpha'][0]- 0.2 ) , max(np.pi /2, prior_sel['alpha'][0] + 0.2 ) )
        # F.ax51.set_ylim(min( -np.pi /2, prior_sel['alpha'][0]- 0.2 )  , max(np.pi /2, prior_sel['alpha'][0]+ 0.2 )  )
        plt.xlim(0, 20)
        #marginal_stack_xr  = xr.concat(marginal_stack.values(), dim='k'  ).sortby('k')

        # no prior
        F.ax3 = F.fig.add_subplot(gs[8:, 0:-2])
        #F.ax3.tick_params(bottom= False, labelbottom=False)
        # F.ax3.spines['bottom'].set_visible(False)
        if brute is True:
            #plt.title(next(fn) +'Beam Pair', loc='left')
            plt.title(next(fn) +'Sample Visualization without prior', loc='left')
            #plt.title('no Prior', loc='left')
            SM_nop.plot_brute(clevel =  brute_clevel)
            # plt.colorbar(orientation='horizontal')
            plt.plot(SM_nop.fitter_brute.brute_x0[1], SM_nop.fitter_brute.brute_x0[0], '.',  color ='red', markersize=5, label= 'best fit', zorder=10)
        if optimze is True:
            SM_nop.plot_optimze(color= 'r', markersize=10, zorder=12, label= 'Dual Annealing')

        if sample is True:
            SM_nop.plot_sample(markersize= 2, linewidth= 0.8, alpha= 0.2, color= 'black', zorder=8)

        F.ax4 = F.fig.add_subplot(gs[8:, -2:])
        F.ax4.tick_params(left= False, labelleft=False)#,labelbottom= False)
        #return F

        # if (fitting_kargs['prior'] is not None):
        #     F.ax3.axhline(prior_sel['alpha'][0], color='green', linewidth = 2, label ='Prior')
        #     #F.ax3.axhspan(prior_sel['alpha'][0]- prior_sel['alpha'][1], prior_sel['alpha'][0]+ prior_sel['alpha'][1], color='gray', alpha=0.3)
        #     F.ax3.axhline(prior_sel['alpha'][0]- prior_sel['alpha'][1], color='green', linewidth = 0.7)
        #     F.ax3.axhline(prior_sel['alpha'][0]+ prior_sel['alpha'][1], color='green', linewidth = 0.7)

        F.ax3.axhline(fitter.params['alpha'].min, color='black', linewidth = 0.6, alpha = 1)
        F.ax3.axhline(fitter.params['alpha'].max, color='black', linewidth = 0.6, alpha = 1)

        F.ax3.set_yticks([-np.pi/2,   0, np.pi/2])
        F.ax3.set_yticklabels(['$-\pi$' ,'0', '$\pi$'])
        F.ax3.set_xticks([0, np.pi/2, np.pi, np.pi* 1.5, 2 *np.pi])
        F.ax3.set_xticklabels(['0', '', '$\pi$', '', '$2\pi$'])


        plt.sca(F.ax3)
        #plt.legend(loc= 1)
        plt.xlabel('')
        plt.ylabel('Angle of Incidence')
        plt.xlim(0, np.pi*2)
        plt.xlabel('Wave Phase')

        plt.sca(F.ax4)
        plt.title(next(fn) +'Marginal', loc= 'left')
        #plt.xlabel('Density')
        plt.stairs(y_hist_nop, bins_nop, orientation='horizontal', color='k')

        # deg_ticks=np.arange(-180, 360+60, 60)
        # deg_tick_label=[str(l)+'$^\circ$' for l in deg_ticks[:]]
        # deg_ticks=deg_ticks * np.pi/180

        #y_ticks =  MT.tick_formatter(   np.arange(, 12, 1),  interval=2, rounder=1, expt_flag=False, shift=0 )
        F.ax3.set_yticks([-np.pi/2,   0, np.pi/2])
        F.ax3.set_yticklabels(['$-\pi$' ,'0', '$\pi$'])

        F.ax3.set_xticks([0, np.pi/2, np.pi, np.pi* 1.5, 2 *np.pi])
        F.ax3.set_xticklabels(['0', '', '$\pi$', '', '$2\pi$'])

        F.ax4.axhline(fitter.params['alpha'].min, color='black', linewidth = 0.6, alpha = 1)
        F.ax4.axhline(fitter.params['alpha'].max, color='black', linewidth = 0.6, alpha = 1)

        # F.ax3.set_ylim(min( -np.pi /2, prior_sel['alpha'][0]- 0.2 ) , max(np.pi /2, prior_sel['alpha'][0] + 0.2 ) )
        # F.ax4.set_ylim(min( -np.pi /2, prior_sel['alpha'][0]- 0.2 )  , max(np.pi /2, prior_sel['alpha'][0]+ 0.2 )  )

        plt.xlim(0, 20)
        F.ax3.set_ylim(-np.pi/2, np.pi/2)
        F.ax3.set_xlim(0, 2 *np.pi)
        F.ax4.set_ylim(-np.pi/2, np.pi/2)
        F.ax4.set_xlabel('Density')
        #plt.colorbar()

        #marginal_stack.mean('k').plot()
        # F.ax52 = F.fig.add_subplot(gs[7:, -1])
        # F.ax52.tick_params(left= False, labelleft=False)#, right= True, labelright= True)
        #
        # plt.title('Sum', loc= 'left')
        # plt.xlabel('Density')
        #
        # marginal_mean = (marginal_stack_xr * marginal_stack_xr.weight).sum('k') / marginal_stack_xr.weight.sum()
        #
        # plt.stairs(marginal_mean,bins, orientation='horizontal', color=col.rels['group'+str(gi[0]+1)], linewidth= 1.5)
        #
        # F.ax52.axhline(fitter.params['alpha'].min, color='black', linewidth = 0.6, alpha = 1)
        # F.ax52.axhline(fitter.params['alpha'].max, color='black', linewidth = 0.6, alpha = 1)
        #
        # #F.ax52.set_ylim(min( -np.pi /2, prior_sel['alpha'][0]- 0.2 )  , max(np.pi /2, prior_sel['alpha'][0]+ 0.2 )  )
        #
        # F.ax52.set_ylim(-np.pi/1.2, np.pi/1.2)
        # plt.xlim(0, 5)

        cbaxes = F.fig.add_subplot(gs[6:8, 5:-2])
        cbaxes.axis('off')
        cbpos  = cbaxes.get_position()
        #cbaxes2 = F.fig.add_axes([cbpos.x0,cbpos.y0,cbpos.width/5,cbpos.height])
        cbaxes2 = F.fig.add_axes([cbpos.x0,cbpos.y0+ 2*cbpos.height/6,cbpos.width,cbpos.height/6])
        cb = plt.colorbar(cax =cbaxes2, orientation='horizontal')
        cb.set_ticks([-3,0, 3])
        cb.set_label('Anomalous Cost')

        plt.show()
        F.save_light(path= plot_path, name =  'MCMC_fit_' + group[0]+'_'+group[1]+'_'+str(int(xi)) +'_'+ str(pk).zfill(3) )
        F.save_pup(path= plot_path, name =  'MCMC_fit_' + group[0]+'_'+group[1]+'_'+str(int(xi)) +'_'+ str(pk).zfill(3) )

        pk+=1
#
#
# # %%
#
# # %%
# # # A= dict()
# # # for k_pair in zip(k_list, weight_list):
# # #     kk, I = get_instance(k_pair)
# # #     A[kk] = I
# #
# # with futures.ProcessPoolExecutor(max_workers=Nworkers) as executor:
# #     A = dict( executor.map(get_instance, zip(k_list, weight_list)   ))
# #
# # cost_stack  = dict()
# # marginal_stack =dict()
# # #fitting_kargs = {'size' :1}
# # L_sample    = pd.DataFrame(index=['alpha', 'group_phase', 'K_prime', 'K_amp'] )
# # L_optimize  = pd.DataFrame(index=['alpha', 'group_phase', 'K_prime', 'K_amp'] )
# # L_brute     = pd.DataFrame(index=['alpha', 'group_phase', 'K_prime', 'K_amp'] )
# #
# # for kk,I in A.items():
# #     L_sample[kk]   = I['L_sample_i']
# #     L_optimize[kk] = I['L_optimize_i']
# #     L_brute[kk]    = I['L_brute_i']
# #
# #     marginal_stack[kk] = I['marginal_stack_i']
# #     cost_stack[kk]     = I['cost']
# #
# # # ## add beam_group dimension
# # marginal_stack  = xr.concat(marginal_stack.values(), dim='k'  ).sortby('k')
# # L_sample        = L_sample.T.sort_values('K_prime')
# # L_optimize      = L_optimize.T.sort_values('K_prime')
# # L_brute         = L_brute.T.sort_values('K_prime')
# #
# # #print(marginal_stack.angle.data[::20])
# #
# # print('done with ',  group, xi/1e3)
# #
# # # % collect
# # ikey        = str(xi) +'_' +  '_'.join(group)
# #
# # #marginal_stack.coords['cost']        = (('k'), np.expand_dims(np.expand_dims(list(cost_stack.values()), 1), 2) )
# # marginal_stack.name           = 'marginals'
# # marginal_stack                = marginal_stack.to_dataset()
# # marginal_stack['cost']        = (('k'), list(cost_stack.values()) )
# # marginal_stack['weight']      = (('k'), weight_list )
# #
# # group_name  = str('group' + group[0].split('gt')[1].split('l')[0])
# # marginal_stack.coords['beam_group'] = group_name
# # marginal_stack.coords['x']          = xi
# #
# # Marginals[ikey]                     = marginal_stack.expand_dims(dim = 'x', axis = 0).expand_dims(dim = 'beam_group', axis = 1)
# # Marginals[ikey].coords['N_data']    = ( ('x', 'beam_group'), np.expand_dims(np.expand_dims(N_data, 0), 1) )
# # # L_brute
# # # L_optimize
# #
# # L_sample['cost'] = cost_stack
# # L_sample['weight'] = weight_list
# # L_collect[group_name, str(int(xi))] = L_sample#pd.concat(L_collect_per_xi)
# #
#
# # %%
# #list(Marginals.values())[0]
# MM = xr.merge( Marginals.values())
# MM =xr.merge([ MM, Prior_smth])
# #MM.to_netcdf(save_path + save_name + '_marginals.nc')
#
# LL = pd.concat(L_collect)
# #MT.save_pandas_table({'L_sample':LL} ,save_name+ '_res_table', save_path)
#
# # %% plot
# font_for_print()
# F = M.figure_axis_xy(6, 5.5, view_scale= 0.7, container = True)
#
# gs = GridSpec(4,6,  wspace=0.2,  hspace=.8)#figure=fig,
#
# ax0 = F.fig.add_subplot(gs[0:2, -1])
# ax0.tick_params(labelleft=False)
#
# #klims = k_list.min()*0.2 , k_list.max()*1.2
#
# klims = 0, LL['K_prime'].max()*1.2
#
#
# for g in MM.beam_group:
#     MMi = MM.sel(beam_group= g)
#     plt.plot(  MMi.weight.T,MMi.k,   '.', color= col_dict[str(g.data)], markersize= 3, linewidth = 0.8)
#
# plt.xlabel('Power')
# plt.ylim(klims)
#
# ax1 = F.fig.add_subplot(gs[0:2 , 0:-1])
#
# for g in MM.beam_group:
#     Li = LL.loc[str(g.data)]
#
#     angle_list = np.array(Li['alpha']) *  180 /np.pi
#     kk_list = np.array(Li['K_prime'])
#     weight_list_i = np.array(Li['weight'])
#
#     plt.scatter( angle_list, kk_list, s= (weight_list_i*8e1)**2 , c=col_dict[str(g.data)], label ='mode ' + str(g.data) )
#
#
# # lflag= 'paritions ww3'
# # for i in np.arange(6):
# #     i_dir, i_period = Prior.loc['pdp'+ str(i)]['mean'], Prior.loc['ptp'+ str(i)]['mean']
# #     i_k = (2 * np.pi/ i_period)**2 / 9.81
# #     i_dir = [i_dir -360 if i_dir > 180 else i_dir][0]
# #     i_dir = [i_dir +360 if i_dir < -180 else i_dir][0]
# #
# #     plt.plot(i_dir, i_k,  '.', markersize = 6, color= col.red, label= lflag)
# #     plt.plot(i_dir, i_k,  '-', linewidth = 0.8, color= col.red)
# #
# #     lflag = None
#
# dir_best[dir_best> 180] = dir_best[dir_best> 180] -360
# plt.plot(dir_best,  Pwavenumber ,  '.r', markersize = 6)
#
# dir_interp[dir_interp> 180] = dir_interp[dir_interp> 180] -360
# plt.plot(dir_interp, Gk.k,  '-', color= 'red', linewidth = 0.3, zorder=11)
#
#
# #ax1.axvline(  best_guess * 180/ np.pi , color=col.blue, linewidth = 1.5, label ='best guess fitting')
#
# # ax1.axvline(  (prior_sel['alpha'][0])  *  180 /np.pi, color='k', linewidth = 1.5, label ='prior')
# # ax1.axvline(  (prior_sel['alpha'][0]- prior_sel['alpha'][1])  *  180 /np.pi, color='k', linewidth = 0.7, label ='prior uncertrainty')
# # ax1.axvline(  (prior_sel['alpha'][0]+ prior_sel['alpha'][1]) *  180 /np.pi , color='k', linewidth = 0.7)
#
# plt.fill_betweenx(Gk.k, (dir_interp_smth -spread_smth)* 180 /np.pi, (dir_interp_smth +spread_smth)* 180 /np.pi,  zorder= 1, color=col.green1, alpha = 0.2 )
# plt.plot(dir_interp_smth * 180 /np.pi, Gk.k , '.', markersize = 1 , color=col.green1)
#
# ax1.axvline(85, color='gray', linewidth= 2)
# ax1.axvline(-85, color='gray', linewidth= 2)
#
#
# plt.legend()
# plt.ylabel('wavenumber (deg)')
# plt.xlabel('Angle (deg)')
#
# #plt.xlim(- 170, 170)
# #plt.xlim(- 90, 90)
# plt.ylim(klims)
#
# prior_angle_str =str(np.round( (prior_sel['alpha'][0])  *  180 /np.pi))
# plt.title(track_name + '\nprior=' + prior_angle_str + 'deg', loc= 'left' )
#
# plt.xlim( min( [ -90,  np.nanmin(dir_best)] ),  max( [np.nanmax(dir_best), 90]) )
#
#
# ax3 = F.fig.add_subplot(gs[2 , 0:-1])
#
# for g in MM.beam_group:
#     MMi = MM.sel(beam_group= g)
#     wegihted_margins = ( (MMi.marginals * MMi.weight).sum(['x','k'] )/MMi.weight.sum(['x', 'k']) )
#     plt.plot(  MMi.angle * 180/ np.pi, wegihted_margins , '.', color= col_dict[str(g.data)], markersize= 2, linewidth = 0.8)
#
# plt.ylabel('Density')
# plt.title('weight margins', loc='left')
#
# #plt.plot(marginal_stack.angle *  180 /np.pi, marginal_stack.T ,  c=col.gray, label ='weighted mean BF')
#
# #plt.plot(cost_wmean.angle *  180 /np.pi, cost_wmean ,  c=col.rascade3, label ='weighted mean BF')
# plt.xlim(- 90, 90)
# #plt.xlim(- 125, 125)
#
# ax3 = F.fig.add_subplot(gs[-1 , 0:-1])
#
# for g in MM.beam_group:
#     MMi = MM.sel(beam_group= g)
#     wegihted_margins = MMi.marginals.mean(['x','k'] )# ( (MMi.marginals * MMi.weight).sum(['x','k'] )/MMi.weight.sum(['x', 'k']) )
#     plt.plot(  MMi.angle * 180/ np.pi, wegihted_margins , '.', color= col_dict[str(g.data)], markersize= 2, linewidth = 0.8)
#
# plt.ylabel('Density')
# plt.xlabel('Angle (deg)')
# plt.title('unweighted margins', loc='left')
#
# #plt.plot(marginal_stack.angle *  180 /np.pi, marginal_stack.T ,  c=col.gray, label ='weighted mean BF')
#
# #plt.plot(cost_wmean.angle *  180 /np.pi, cost_wmean ,  c=col.rascade3, label ='weighted mean BF')
# plt.xlim(- 90, 90)
# #plt.xlim(- 125, 125)
#
# #F.save_pup(path= plot_path, name = 'B04_marginal_distributions')
#
# #MT.json_save('B04_success', plot_path, {'time':'time.asctime( time.localtime(time.time()) )'})
