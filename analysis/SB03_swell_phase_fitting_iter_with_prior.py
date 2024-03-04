
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

col.colormaps2(21)

#import s3fs
# %%
track_name, batch_key, test_flag = io.init_from_input(sys.argv) # loads standard experiment
#track_name, batch_key, test_flag = '20190605061807_10380310_004_01', 'SH_batch01', False
#track_name, batch_key, test_flag = '20190601094826_09790312_004_01', 'SH_batch01', False
#track_name, batch_key, test_flag = '20190207111114_06260210_004_01', 'SH_batch02', False
#track_name, batch_key, test_flag = '20190219073735_08070210_004_01', 'SH_batch02', False
#track_name, batch_key, test_flag = '20190601093502_09790310_004_01', 'SH_batch02', False
track_name, batch_key, test_flag = '20190502021224_05160312_004_01', 'SH_batch02', False


#print(track_name, batch_key, test_flag)
hemis, batch = batch_key.split('_')
#track_name= '20190605061807_10380310_004_01'
ATlevel= 'ATL03'

save_path   = mconfig['paths']['work'] + '/B03_spectra_'+hemis+'/'
save_name   = 'B03_'+track_name

plot_path   = mconfig['paths']['plot'] + '/'+hemis+'/'+batch_key+'/' + track_name + '/B_spectra/'
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
Prior = MT.load_pandas_table_dict('/A02b_'+track_name, load_path)['priors_hindcast']

#
#     import numpy as np
#
#     G = np.vstack([ np.cos(np.outer(XX, ks) + np.outer(YY, ls) ).T ,  np.sin(np.outer(XX, ks) + np.outer(YY, ls) ).T ] ).T
#
#     #phase1 = np.random.rand(1, amp_list.size) *  np.pi*2
#     #phase = np.arange(0, amp_list.size) *  np.pi/2
#
#     b = np.hstack([ np.cos(group_phase)*amps, np.sin(group_phase) *amps]).squeeze() * amp_height
#     z_model = (G @ b)
#
#     return z_model


# %% select data
for b in Gx.beam:
    B = Gx.sel(beam= b)
    plt.plot(B['x_coord'], B['y_coord'], '.')

plt.grid()
# %%

# for group in beam_groups:
#     B1 = Gd.sel(beam =  group[0])
#     B2 = Gd.sel(beam =  group[1])
#
#     dist = np.sqrt( (B1['x_coord']-B2['x_coord'])**2 + (B1['y_coord']-B2['y_coord'])**2 )
#     print(group, dist.data)

#     B1.coords['dist_y'] = 0
#     B2.coords['dist_y'] = dist

# isolate case
xi = 2
group = beam_groups[0]
GGx = Gx.sel(beam= group).isel(x = 2)
GGk = Gk.sel(beam= group).isel(x = 2)

# %%
GGk.gFT_PSD_model.mean('beam').plot()

# %%
data_plot_karg = {'linewidth':2, 'color':'k', 'alpha' :0.5}
model_plot_karg = {'linewidth':1, 'color':'r', 'alpha' :1}

G_x = GGx.isel(beam = 0)
y_offset= 0.5
plt.plot(G_x.eta,G_x.y_model , **model_plot_karg)
plt.plot(G_x.eta,G_x.y_data, **data_plot_karg)

plt.xlim(-500, 1000)

# %%

# dd = GGk['gFT_PSD_data']
# m = 3
# variance_frac = 0.5

def define_wavenumber_weights_tot_var(dd, m = 3, variance_frac = 0.33, verbose=False):

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
    dd_rm   = M.runningmean(dd, m, tailcopy=True)
    k      =  dd.k[~np.isnan(dd_rm)].data
    dd_rm   =  dd_rm[~np.isnan(dd_rm)]

    orders = dd_rm.argsort()[::-1]
    var_mask = dd_rm[orders].cumsum()/dd_rm.sum() < variance_frac


    pos_cumsum = orders[var_mask]
    k_list  = k[pos_cumsum]
    dd_list = dd_rm[pos_cumsum]

    if verbose:
        plt.plot(dd.k, dd, '-k', markersize= 20)
        plt.plot(k, dd_rm, '-b', markersize= 20)

        #print(k_list, dd_list)
        plt.plot(k_list, dd_list, '.r', markersize= 10, zorder=12)

    return var_mask[orders.argsort()], k, dd_rm, pos_cumsum


def define_wavenumber_weights_threshold(dd, m = 3, Nstd= 2, verbose=False):

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

#mask, k, weights, positions =  define_wavenumber_weights_threshold( Gi.mean('dist_y')['gFT_PSD_data'], 3 , verbose= True)
#plt.plot(k[mask], weights[mask], 'g*', markersize=20)
# plt.show()

mask, k, weights, positions = define_wavenumber_weights_tot_var( GGk.mean('beam')['gFT_PSD_data'], m= 3,  variance_frac = 0.33 , verbose= True)
#plt.xlim(k_list.min()*.9, k_list.max()*1.1)

# group wavenumbers
import scipy.ndimage
k_group_mask =scipy.ndimage.measurements.label(mask )



# %%

def get_wavenumbers_polar( amp, angle_rad):
    """
    inputs:

    amp     length of peak k vector in radial coordinates
    angle_rad   angle of peak k vector in radians between - pi/2 to  + pi/2

    returns:
    wavenumber k,l
    """
    import numpy as np
    k0 = amp * np.cos(angle_rad)
    l0 = amp * np.sin(angle_rad)

    return k0, l0

def wavemodel(XX, YY, ks, ls, amps, group_phase = 0):

    import numpy as np

    G = np.vstack([ np.cos(np.outer(XX, ks) + np.outer(YY, ls) ).T ,  np.sin(np.outer(XX, ks) + np.outer(YY, ls) ).T ] ).T

    #phase1 = np.random.rand(1, amp_list.size) *  np.pi*2
    #phase = np.arange(0, amp_list.size) *  np.pi/2

    b = np.hstack([ np.cos(group_phase) * amps , np.sin(group_phase) * amps ]).squeeze()
    z_model = (G @ b)

    return z_model


def get_z_model(x_positions, y_position, K_prime, K_amp,  alpha_rad, group_phase):

    K_abs = K_prime   / np.cos(alpha_rad)

    k = K_abs * np.cos(alpha_rad)
    l = K_abs * np.sin(alpha_rad)

    return wavemodel( x_positions,y_position, k, l, np.array(K_amp ), group_phase= group_phase)




def plot_instance(z_model, fargs , key, non_dim=False, title_str= None ,fitter=None, view_scale = 0.3):

    x_concat, y_concat, z_concat = fargs


    F = M.figure_axis_xy(5,6, view_scale = view_scale, container = True)
    plt.suptitle(title_str)
    gs = GridSpec(4, 2,  wspace=0.3,  hspace=1.2)#figure=fig,
    # y_offset= 0.5
    # plt.plot(Gm.eta,   Gm.y_model_normed+y_offset * Gm.dist_y/np.diff(Gm.dist_y), **model_plot_karg)
    # plt.plot(Gm.eta,   Gm.y_model_normed+y_offset * Gm.dist_y/np.diff(Gm.dist_y), **data_plot_karg)
    # plt.xlim(-1000, 1000)


    import itertools
    col_list = itertools.cycle([col.cascade2, col.rascade2, col.cascade1, col.rascade1, col.cascade3, col.rascade3])

    beam_list = list(set(y_concat))
    for y_pos, pos in zip(beam_list, [ gs[0, :] , gs[1, :] ] ):

        F.ax2 = F.fig.add_subplot(pos)

        plt.title( str(y_pos) )
        #plt.plot(GG['x_prime'], GG['z_model'], '.' )


        plt.plot(x_concat[y_concat == y_pos], z_concat[y_concat == y_pos] ,  c=col.gray, linewidth = 1)
        plt.plot(x_concat[y_concat == y_pos], z_model[y_concat == y_pos] , '-', c=next(col_list))
        plt.xlim(x_concat[y_concat == y_pos][0], x_concat[y_concat == y_pos][-1])


    plt.xlabel('meter')
    F.ax3 = F.fig.add_subplot(gs[2:, :])
    if fitter is not None:
        plt.title('Brute-force costs', loc='left')
        plot_brute_force(fitter)

    return F


def plot_brute_force(fitter_brute):

    clevel = np.linspace(-2.2, 2.2, 30)
    dd = (fitter_brute.brute_Jout- fitter_brute.brute_Jout.mean())/fitter_brute.brute_Jout.std()
    plt.contourf(fitter_brute.brute_grid[1,:,:], fitter_brute.brute_grid[0,:,:], dd , clevel, cmap= plt.cm.YlGnBu_r )

    plt.colorbar()
    #plt.scatter(fitter_brute.brute_grid[1,:,:], fitter_brute.brute_grid[0,:,:], s=0.2, alpha= 0.4, color='black')
    plt.plot(fitter_brute.brute_x0[1], fitter_brute.brute_x0[0], '+r', markersize=20, label= 'Brute force')

    plt.xlabel('Phase (rad)')
    plt.ylabel('Angle (rad)')

    plt.legend()



# %% normalize data
key = 'y_data'
amp_Z = G_x[key +'_normed']=  (GGx[key] - GGx[key].mean(['eta']) )/GGx[key].std(['eta'])
N = G_x[key +'_normed'].shape[0]

eta_2d = GGx.eta + GGx.x_coord - GGx.x_coord.mean()
nu_2d = GGx.eta * 0 + GGx.y_coord - GGx.y_coord.mean()

amp_Z.shape

# repack as np arrays
x_concat = eta_2d.data.T.flatten()
y_concat = nu_2d.data.T.flatten()
z_concat = amp_Z.data.flatten()

# repack as np arrays
# x_concat = np.concatenate([amp_Z.eta, amp_Z.eta])
# y_concat = np.concatenate([amp_Z.isel(dist_y= 0).dist_y+ amp_Z.eta*0 , amp_Z.isel(dist_y= 1).dist_y+ amp_Z.eta*0 ] )
# z_concat = np.concatenate([amp_Z.isel(dist_y= 0).data, amp_Z.isel(dist_y= 1).data])

# test
#plt.plot(x_concat, 1*y_concat +z_concat)
# plt.plot(G_x_model.eta, G_x_model.y_data/G_x_model.y_data.std()  +y_offset * G_x_model.dist_y/np.diff(G_x_model.dist_y), **data_plot_karg)
# plt.xlim(-1000, 1000)

# y_offset = 0.2
# plt.plot(G_x_model.eta,G_x_model.y_model+y_offset *G_x_model.dist_y/np.diff(G_x_model.dist_y), **model_plot_karg)
# plt.plot(G_x_model.eta,G_x_model.y_data+y_offset * G_x_model.dist_y/np.diff(G_x_model.dist_y), **data_plot_karg)
# plt.xlim(-1000, 1000)
# test: this should be 0
x_concat= x_concat[~np.isnan(z_concat)]
y_concat= y_concat[~np.isnan(z_concat)]
z_concat= z_concat[~np.isnan(z_concat)]

if np.isnan(z_concat).sum() != 0:
    raise ValueError('There are still nans')

# %%


def objective_func(pars, x, y, z, test_flag= False , prior= None ):

    z_model = get_z_model(x, y, pars['K_prime'], pars['K_amp'], pars['alpha'],pars['group_phase'])
    if prior is not None:
        a_0, a_std = prior['alpha']
        penalties = np.array([  (abs(a_0 - pars['alpha'] )**2  / a_std)  ])
    else:
        penalties =  np.array([0])

    cost =( abs(z - z_model) )**2  /z.std()**2
    if test_flag:
        return z_model
    else:
        return np.concatenate([cost , 2 * penalties])


import lmfit as LM
params0 = LM.Parameters()

d_alpha, d_phase= np.pi/90, 2 *np.pi/90
alpha_grid = np.arange(-0.9 * np.pi /2, 0.9 * np.pi /2+ d_alpha, d_alpha)
phase_grid = np.arange(0              , 2*np.pi+d_phase, d_phase)

params0.add('alpha', 0       ,  vary=True  , min=alpha_grid[0], max=alpha_grid[-1], brute_step= d_alpha)
params0.add('group_phase', 0 ,  vary=True  , min=phase_grid[0], max= phase_grid[-1]+d_phase, brute_step= d_phase)

fitting_args = (x_concat, y_concat, z_concat)


Prior2              = Prior.loc[['ptp0','ptp1','ptp2','ptp3','ptp4','ptp5']]['mean']
# dominat_period      = Prior2[Prior2.max() ==Prior2]
# dominant_dir        = Prior.loc[list(dominat_period.index)[0].replace('ptp', 'pdp' )]['mean']
# dominant_dir_spread = Prior.loc[list(dominat_period.index)[0].replace('ptp', 'pspr' )]['mean']
#
# prior_sel= {'alpha': ( dominant_dir *np.pi/180 , dominant_dir_spread *np.pi/180) } # to radiens
#
import ICEsat2_SI_tools.wave_tools as waves


dominat_period      = Prior2[Prior2.max() ==Prior2]
aa = Prior.loc[['pdp0','pdp1','pdp2','pdp3','pdp4','pdp5']]['mean'].astype('float')
dominant_dir        = waves.get_ave_amp_angle(aa *0+1,aa  )[1]
dominant_dir_spread = Prior.loc[['pspr0','pspr1','pspr2','pspr3','pspr4','pspr5']]['mean'].median()

#prior_sel= {'alpha': ( dominant_dir *np.pi/180 , dominant_dir_spread *np.pi/180) } # to radiens
prior_sel= {'alpha': ( 1 , dominant_dir_spread *np.pi/180) } # to radiens

#prior_sel= {'alpha': ( Prior.loc['dp']['mean'] *np.pi/180 , Prior.loc['spr']['mean'] *np.pi/180) }

#fitting_kargs = {'prior': None}
fitting_kargs = {'prior': prior_sel  }
angle_list  = list()
cost_list  = list()
cost_stack = dict()
#fitting_kargs = {'size' :1}
L = pd.DataFrame(index=['alpha', 'group_phase', 'K_prime', 'K_amp'] )


k_list, weight_list =  k[mask], weights[mask]


N_grid = 30
N_data = x_concat.size

#k_prime_max, Z_max = k_list[0], weight_list[0]
#k_list[:4]
for k_prime_max,Z_max in zip(k_list[::4], weight_list[::4]):

    print('#-------------------------------------------------------------#')
    print(k_prime_max)
    amp_enhancement = 1
    amp_Z = 0.5 #amp_enhancement * abs(Z_max)**2 /N

    params = params0.copy()
    params.add('K_prime', k_prime_max ,  vary=False  , min=k_prime_max*0.5, max=k_prime_max*1.5)
    params.add('K_amp', amp_Z         ,  vary=False  , min=amp_Z*.0       , max=amp_Z*5)

    #fitter = LM.minimize(objective_func, params, args=fitting_args, method='dual_annealing',max_nfev=None)
    fitter_brute = LM.minimize(objective_func, params, \
                    args=fitting_args, kws=fitting_kargs ,  method='brute', workers=3 )
    print(LM.report_fit(fitter_brute))

    params['K_amp'].vary = False
    fitter = LM.minimize(objective_func, params, \
                    args=fitting_args, kws=fitting_kargs ,  method='dual_annealing',max_nfev=None)

    print(LM.report_fit(fitter))

    z_model = objective_func(fitter.params, *fitting_args , test_flag= True)
    angle_list.append(fitter.params['alpha'].value)
    cost_list.append( (fitter.residual**2).sum()/(z_concat**2).sum() )
    # add prior:
    #fitting_kargs = {'prior': None}
    #fitter=None, F = plot_instance(GG, 'slopes', 'z_model', fitter = fitter_brute, view_scale = 0.5, non_dim=True )
    F = plot_instance(z_model, fitting_args, 'y_data_normed' , fitter = fitter_brute ,title_str =  str(k_prime_max),  view_scale = 0.6)
    F.ax3.axhline(prior_sel['alpha'][0], color='k', linewidth = 1.5)
    #F.ax3.axhspan(prior_sel['alpha'][0]- prior_sel['alpha'][1], prior_sel['alpha'][0]+ prior_sel['alpha'][1], color='gray', alpha=0.3)
    F.ax3.axhline(prior_sel['alpha'][0]- prior_sel['alpha'][1], color='k', linewidth = 0.7)
    F.ax3.axhline(prior_sel['alpha'][0]+ prior_sel['alpha'][1], color='k', linewidth = 0.7)
    plt.plot(fitter.params['group_phase'].value, fitter.params['alpha'].value, '.r', markersize=20)

    print(fitting_kargs)
    print(fitter.params.pretty_print())
    plt.show()

    #cost_stack[k_prime_max] = xr.DataArray( fitter_brute.brute_Jout/N_data, dims= ('angle', 'phase'),  coords = {'angle':np.linspace(-np.pi/2, np.pi/2, N_grid), 'phase':np.linspace(0, 2* np.pi, N_grid) } )
    cost_stack[k_prime_max] = xr.DataArray( fitter_brute.brute_Jout/N_data, dims= ('angle', 'phase'),  coords = {'angle':alpha_grid, 'phase':phase_grid } )

    cost_stack[k_prime_max].coords['k']  = np.array(k_prime_max) #( ('k'), np.array(k_prime_max) )


    L[k_prime_max]          = fitter.params.valuesdict().values()

    #F.save_light(path= plot_path, name = key_name + '_fit_k' + str(k_prime_max))

cost_stack = xr.concat(cost_stack.values(), dim='k'  ).sortby('k')
L = L.T.sort_values('K_prime')


# %%

cost_stack2 = cost_stack #/ cost_stack.mean('angle').mean('phase').mean()
cost_stack_rolled = xr.DataArray(coords = cost_stack.coords)

shift_list = list()

for kindex in L.index:
    #.sel( phase=  L[i]['group_phase'], method ='nearest').plot()
    ii = abs(cost_stack.k - kindex).argmin().data

    shift = int( -abs(cost_stack.phase - L['group_phase'][kindex]).argmin().data     + cost_stack.phase.size/2 )
    #shift = int(- abs(cost_stack.phase - L['group_phase'][kindex]).argmin().data  )
    shift_list.append(shift)
    cost_stack_rolled[ii, :,:]= cost_stack.sel(k= kindex).roll(phase = shift, roll_coords='phase').data

    # M.figure_axis_xy(7, 3, view_scale= 0.5)
    # plt.subplot(1, 2,1)
    # cost_stack.sel(k= kindex).plot(cmap =plt.cm.Blues_r)
    # plt.plot(L.T[kindex]['group_phase'], L.T[kindex]['alpha'], '.r', markersize=20)
    #
    # plt.subplot(1, 2,2)
    # cost_stack_rolled[ii, :,:].plot(cmap =plt.cm.Blues_r)
    # plt.show()

cost_stack_rolled['phase'] = cost_stack_rolled['phase'] - np.pi

# %%
#weights = xr.DataArray(Z_max_list_gauss, dims ='k', coords = {'k': k_list_gauss})
weights_costs = xr.DataArray(weight_list, dims ='k', coords = {'k': k_list})
weights_costs = weights_costs/weights_costs.max()
weights_costs = weights_costs.sel(k= cost_stack_rolled.k)

weight_k = (1/weights_costs.k)
weight_k = weight_k/weight_k.max()

weights_sum= weights_costs * weight_k.data**2
plt.plot(weights_sum)


data_normed  = cost_stack_rolled#/cost_stack_rolled.std(['angle', 'phase'])
cost_wmean  = (weights_sum * data_normed /weights_sum.sum() ).interp(phase= 0)#.sum('k')

#cost_wmean  = data_normed.sel(phase = 0 , method='nearest')
#plt.plot( cost_wmean.T )

best_guess = cost_wmean.angle[cost_wmean.mean('k').argmin()].data

best_guess * 180/ np.pi



# %% 2nd step optimization
F = M.figure_axis_xy(5, 3, view_scale= 0.7, container = True)

gs = GridSpec(1,5,  wspace=0.25,  hspace=.2)#figure=fig,

ax0 = F.fig.add_subplot(gs[0, -1])
ax0.tick_params(labelleft=False)


klims = k_list.min()*0.8 , k_list.max()*1.2

col_dict = col.rels
for g in group:
    plt.plot(  GGk.sel(beam=g)['gFT_PSD_data'].data,GGk.k,   '-k', color= col_dict[g], markersize= 20, linewidth = 0.8)

plt.plot(weights, k,  '-b', linewidth=0.9)
plt.plot(weight_list, k_list,  '.r', markersize= 5, zorder=12)

plt.ylim(klims)

ax1 = F.fig.add_subplot(gs[0 , 0:-1])

plt.scatter( np.array(angle_list)*  180 /np.pi, np.array(k_list), s= (np.array(weight_list)*4e1)**3 , c=col.rascade1, label ='min per freq.')

cost_dd = cost_wmean.mean('k')
plt.plot(cost_wmean.angle *  180 /np.pi, k_list.mean() * cost_dd/cost_dd.mean() ,  c=col.rascade3, label ='weighted mean BF')

lflag= 'paritions ww3'
for i in np.arange(6):
    i_dir, i_period = Prior.loc['pdp'+ str(i)]['mean'], Prior.loc['ptp'+ str(i)]['mean']
    i_k = (2 * np.pi/ i_period)**2 / 9.81
    i_dir = [i_dir -360 if i_dir > 180 else i_dir][0]
    i_dir = [i_dir +360 if i_dir < -180 else i_dir][0]

    plt.plot(i_dir, i_k,  '.', markersize = 10, color= col.green, label= lflag)
    lflag = None

ax1.axvline(  best_guess * 180/ np.pi , color=col.blue, linewidth = 1.5, label ='best guess fitting')

ax1.axvline(  (prior_sel['alpha'][0])  *  180 /np.pi, color='k', linewidth = 1.5, label ='prior')
ax1.axvline(  (prior_sel['alpha'][0]- prior_sel['alpha'][1])  *  180 /np.pi, color='k', linewidth = 0.7, label ='prior uncertrainty')
ax1.axvline(  (prior_sel['alpha'][0]+ prior_sel['alpha'][1]) *  180 /np.pi , color='k', linewidth = 0.7)


plt.legend()
plt.xlabel('Angle (deg)')
plt.ylabel('wavenumber (deg)')
plt.xlim(- 125, 125)
plt.ylim(klims)


# _%%%%%%%%%%%%%%%%%%%%%%%%%% upto  here
 # %%
#plt.plot( k_list, cost_list, '.')

## %% cross correlation

# A1 = GG[GG['y_prime'] == 0]
# A2 = GG[GG['y_prime'] == 300]

# def autocrosscorr_func_1d(dd1, dd2):
#     "takes data s as xarray"
#
#     #print(dim)
#     xp1=(dd1-dd1.mean())
#     xp2=(dd2-dd2.mean())
#     var=xp2.var()
#     corr=np.correlate(xp1,xp2,'full')[len(xp1)-1:]/var/len(xp1)
#     return corr
#
# ymean = abs(A1['y_prime'].mean() - A2['y_prime'].mean())
# angle_rad = np.concatenate([ -np.arctan2(A2['x_prime'], ymean)[::-1], np.arctan2(A2['x_prime'], ymean) ])
# cross_corr = np.concatenate([autocrosscorr_func_1d(A1['slopes'][::-1], A2['slopes'][::-1])[::-1], autocrosscorr_func_1d(A1['slopes'], A2['slopes']) ])

#plt.plot( angle_rad, cross_corr )


# %%

F = M.figure_axis_xy(6, 6.5, view_scale= 0.5)
ax = plt.subplot(3, 1, 1)
plt.title( 'Normalized weighted mean cost at maximum phase', loc='left')
dn_unweighted = data_normed.mean('k')

plt.plot(data_normed.angle* 180/np.pi, M.normalize(dn_unweighted.sel(phase=0, method = 'nearest')  ) , label='mean')
plt.plot(cost_wmean.angle* 180/np.pi , M.normalize(cost_wmean.sel(phase=0, method = 'nearest') )    , label='weighted mean' )

def get_min_values(field2d, x, y):
    min_pos = np.unravel_index(field2d.argmin(), field2d.shape)
    return x[min_pos[0]] , y[min_pos[1]]

angle_mean, phase_mean  = get_min_values(cost_wmean,cost_wmean.angle, cost_wmean.phase )

phase_mean, angle_mean
angle_mean.data * 180 /np.pi

#ax.axvline(angle_true* 180 /np.pi, linewidth = 1, color = 'black', label ='True')

ax.axvline(  (prior_sel['alpha'][0])  *  180 /np.pi, color='k', linewidth = 1.5)
ax.axvline(  (prior_sel['alpha'][0]- prior_sel['alpha'][1])  *  180 /np.pi, color='k', linewidth = 0.7)
ax.axvline(  (prior_sel['alpha'][0]+ prior_sel['alpha'][1]) *  180 /np.pi , color='k', linewidth = 0.7)

plt.legend()
plt.xlabel('Angle (deg)')
plt.ylabel('normalized cost')
plt.xlim(-80, 80)

# %%







F = M.figure_axis_xy(5, 4.5, view_scale = 0.5)
# L.T['alpha'].hist(bins = 60)
# F.ax.axvline(alpha_true, linewidth = 0.8, color = 'black')
xlims= [L.T.index.min()* 0.9, L.T.index.max()* 1.1]

ax0 = plt.subplot(2, 1, 1)
plt.plot( L.T.index,  L.T['alpha'], '.r', markersize = 10)
#ax0.axhline(alpha_true, linewidth = 0.8, color = 'black')
plt.xlim(xlims[0], xlims[1])

plt.subplot(2, 1, 2)

plt.plot(k_fft, Z_fft_rm )
#plt.plot(k_fft_max, Z_fft_rm[Z_max_pos], '.', markersize= 20)
plt.plot(k_fft_max_list,Z_max_list, '.r', markersize= 10)

plt.xlim(xlims[0], xlims[1])

# %% construct spectrum

L
amp_Z = 2* abs(Z_max)**2 /N

GG['z_model'] = get_z_model(GG['dist'],GG['dist_y'], L.T['K_prime'], L.T['K_amp']/amp_enhancement, L.T['alpha'],L.T['group_phase'])
plot_instance(GG, 'slopes', 'z_model',  view_scale = 0.8)

# %%
params_big = LM.Parameters()
L2 = L.T.sort_values('K_prime')

N_wavenumbers = L2.shape[0]

L2['num'] = np.arange(0, L2.shape[0])

for p in L2.T.items():
    numi= str(int(p[1]['num']))
    ki = p[1]['K_prime']
    params_big.add('K_prime_' +  numi  , ki     ,  vary=False , min=ki*0.5       , max=ki*1.5)

for p in L2.T.items():
    numi= str(int(p[1]['num']))
    ampi = p[1]['K_amp']/amp_enhancement
    params_big.add('K_amp_' + numi   , ampi      ,  vary=False , min=ampi*.5      , max=ampi*5)

for p in L2.T.items():
    numi= str(int(p[1]['num']))
    grpi = p[1]['group_phase']
    params_big.add('group_phase_' + numi  , grpi         ,  vary=False  , min=0           , max= 2*np.pi)

for p in L2.T.items():
    numi= str(int(p[1]['num']))
    grpi = p[1]['alpha']
    params_big.add('alpha_' + numi  , grpi                ,  vary=False  , min=-0.95 * np.pi /2, max=0.95 * np.pi /2)

# def get_max_angle(a):
#     hist = np.histogram(a, np.linspace(- np.pi/2, np.pi/2 , int(np.round(a.size/2)) ) )
#     return hist[1][hist[0].argmax()]
#
#
params_big.add('angle_shift', 0 ,  vary=False  )#, min=-0.95 * np.pi /2, max=0.95 * np.pi /2)

# %%
#params_big

def objective_func_big(params_big, x, y, z, test_flag= False ):

    pardict_val         = np.array(list(params_big.valuesdict().values()))
    K_prime_array       = pardict_val[0:N_wavenumbers]
    K_amp_array         = pardict_val[N_wavenumbers:N_wavenumbers*2]
    group_phase_array   = pardict_val[N_wavenumbers*2:N_wavenumbers*3]
    #alpha     = pardict_val[-1]
    alpha               = pardict_val[N_wavenumbers*3:N_wavenumbers*4] + pardict_val[-1]

    #print(pardict_val[-1])
    #print(alpha)
    z_model =   get_z_model(GG['dist'],GG['dist_y'], K_prime_array, K_amp_array, alpha, group_phase_array)
    cost    =   ( abs(z - z_model) )**2  /z.std()**2
    #alpha_cost  =  (alpha - alpha.mean())**2 / (2* np.pi/180)**2
    if test_flag:
        return z_model
    else:
        return cost

def turn_vary_key_on(pard, key):

    for pi in pard.keys():
        if key in pi:
            pard[pi].vary = True
        else:
            pard[pi].vary = False
    return pard


GG['z_model'] = objective_func_big(params_big, GG['dist'], GG['dist_y'], GG['slopes'], test_flag= True)
plot_instance(GG, 'slopes', 'z_model',  view_scale = 0.6)

# %%
#help(LM.minimize)
# params_big = turn_vary_key_on(params_big, 'group_phase')
# fitter = LM.minimize(objective_func_big, params_big, args=fitting_args, method='ampgo',max_nfev=10000)
#fitter

params2 = turn_vary_key_on(params_big, 'K_prime')
fitter = LM.minimize(objective_func_big, params2, args=fitting_args, method='least_squares',max_nfev=10000)

params2 = turn_vary_key_on(fitter.params, 'K_amp')
fitter = LM.minimize(objective_func_big, params2, args=fitting_args, method='least_squares',max_nfev=10000)

#print('angle_shift')
# params2 = turn_vary_key_on(fitter.params, 'angle_shift')
# fitter = LM.minimize(objective_func_big, params2, args=fitting_args, method='least_squares',max_nfev=10000)
#fitter = LM.minimize(objective_func_big, fitter.params, args=fitting_args, method='emcee')

# %%
GG['z_model'] = objective_func_big(fitter.params, GG['dist'],GG['dist_y'], GG['slopes'], test_flag= True)
plot_instance(GG, 'slopes', 'z_model',  view_scale = 0.6)

# %%
z_mdoel_init = objective_func_big(params_big, GG['dist'],GG['dist_y'], GG['slopes'], test_flag= True)
objective_func_big(params_big, GG['dist'],GG['dist_y'], GG['slopes'], test_flag= False).sum()
z_mdoel_fit = objective_func_big(fitter.params, GG['dist'],GG['dist_y'], GG['slopes'], test_flag= True)
objective_func_big(fitter.params, GG['dist'],GG['dist_y'], GG['slopes'], test_flag= False).sum()
plt.plot(z_mdoel_init)
plt.plot(z_mdoel_fit)


# %%
#fitter.params.valuesdict()

def params_to_Dataframe(pard):

    pardict_val         = np.array(list(pard.valuesdict().values()))

    pardict = dict()
    pardict['K_prime_array']       = pardict_val[0:N_wavenumbers]
    pardict['K_amp_array']         = pardict_val[N_wavenumbers:N_wavenumbers*2]
    pardict['group_phase_array']   = pardict_val[N_wavenumbers*2:N_wavenumbers*3]
    pardict['alpha']               = pardict_val[N_wavenumbers*3:N_wavenumbers*4]

    return pd.DataFrame.from_dict(pardict)

Fits = params_to_Dataframe(fitter.params)
plt.plot( k_fft_max_list, abs( (Fits['alpha']) ), '.' )

Fits['alpha']

plt.hist(fitter.flatchain, 30)
