
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
#import s3fs
# %%
track_name, batch_key, test_flag = io.init_from_input(sys.argv) # loads standard experiment
#track_name, batch_key, test_flag = '20190605061807_10380310_004_01', 'SH_batch01', False
#track_name, batch_key, test_flag = '20190601094826_09790312_004_01', 'SH_batch01', False
#track_name, batch_key, test_flag = '20190207111114_06260210_004_01', 'SH_batch02', False
#track_name, batch_key, test_flag = '20190219073735_08070210_004_01', 'SH_batch02', False
track_name, batch_key, test_flag = '20190601093502_09790310_004_01', 'SH_batch02', False



#print(track_name, batch_key, test_flag)
hemis, batch = batch_key.split('_')
#track_name= '20190605061807_10380310_004_01'
ATlevel= 'ATL03'

save_path   = mconfig['paths']['work'] + '/B03_spectra_'+hemis+'/'
save_name   = 'B03_'+track_name

#plot_path   = mconfig['paths']['plot'] + '/'+hemis+'/'+batch_key+'/' + track_name + '/B_spectra/'
plot_path   = mconfig['paths']['plot'] + '/phase_fitting_fake/'
MT.mkdirs_r(plot_path)
MT.mkdirs_r(save_path)
bad_track_path =mconfig['paths']['work'] +'bad_tracks/'+ batch_key+'/'
# %%

all_beams   = mconfig['beams']['all_beams']
high_beams  = mconfig['beams']['high_beams']
low_beams   = mconfig['beams']['low_beams']
#Gfilt   = io.load_pandas_table_dict(track_name + '_B01_regridded', load_path) # rhis is the rar photon data

load_path   = mconfig['paths']['work'] +'/B01_regrid_'+hemis+'/'
Gd      = io.load_pandas_table_dict(track_name + '_B01_binned' , load_path)  #

col.colormaps2(21)
# %%


T_max = 40 #sec
k_0 = (2 * np.pi/ T_max)**2 / 9.81
x= np.array(Gd['gt1r']['dist'])
dx = np.diff(x).mean()
xlims   = x[0], x[-1]
min_datapoint =  1/k_0/dx

Lpoints = int(np.round(min_datapoint) * 20) * 0.5
Lmeters =Lpoints  * dx

xg = np.arange(Lpoints) *dx
yg = np.array([0, 300])
XX, YY = np.meshgrid(xg, yg)
Nx, Ny = xg.size, yg.size

# %%
sys.path.append(mconfig['paths']['analysis'])
import SB04_2d_wavefield_emulator as WaveEmulator

# test wave model
T_max = 14 #sec
k_abs = (2 * np.pi/ T_max)**2 / 9.81
print(k_abs)
angle  =angle_true = 30 * np.pi/180
imp.reload(WaveEmulator)

#kk, ll, amps, mesh_shape = WaveEmulator.get_stancils_polar(k_abs, angle, size=20, dk= 0.005 , plot_flag=False, random = True)
kk, ll, amps, mesh_shape = WaveEmulator.get_stancils_polar(k_abs, angle, size=200, dk= 0.003 , plot_flag=False, random = True)
#kk, ll, amps, mesh_shape = WaveEmulator.get_stancils_polar(k_abs, angle, size=2, dk= 0.005 , plot_flag=False, random = True)

k_noise, l_noise, amp_noise, stancil_shape = WaveEmulator.get_stancils_polar(0.8, 0 * np.pi/180, size=2, dk = 1, mesh = True , plot_flag= False,  random = True)
#k_noise, l_noise, amp_noise, stancil_shape = WaveEmulator.get_stancils_polar(0.8, 0 * np.pi/180, size=20, dk = 0.9, mesh = True , plot_flag= False, random = True)
amp_noise = (amp_noise *0+1) * 0

k_all = np.concatenate([kk, k_noise])
l_all = np.concatenate([ll, l_noise])
amp_all = np.concatenate([amps, amp_noise])

phase = (np.random.random(amp_all.size)  ) *2*np.pi
G = np.vstack([ np.cos(np.outer(XX, k_all) + np.outer(YY, l_all)).T ,  np.sin(np.outer(XX, k_all) + np.outer(YY, l_all)).T ] ).T

b = np.hstack([ np.cos(phase)*amp_all, np.sin(phase) *amp_all]).squeeze()
z_model = (G @ b)
z_model += np.random.normal(0, 1,z_model.size) * z_model.std()/2
z_model = z_model.reshape(Ny, Nx)

GG0 = pd.DataFrame(columns = ['x', 'y', 'slopes'])
GG0['x_prime'] = XX.reshape(Nx*Ny)
GG0['y_prime'] = YY.reshape(Nx*Ny)
GG0['slopes'] = z_model.reshape(Nx*Ny)
GG0['dist'] = np.sqrt( (GG0['x_prime']) **2 + (GG0['y_prime'])**2)
GG= GG0
#plt.plot(GG['x_prime'], GG['slopes']  , '-', markersize = 0.8)
plt.plot(GG['x_prime'], GG['slopes'] + GG0['y_prime']/10  , '-', markersize = 0.8)
plt.grid()
key_name = 'Lp'+ str(int(Lpoints)) + '_size' + str(int(kk.size)) + '_angle' + str(int(np.round(angle *180/np.pi, 0)))

M.save_anyfig(plt.gcf(), name= key_name+ '_realization', path= plot_path )
plt.show()
# plt.plot(GG['x_prime']/1e3, GG['y_prime']/1e3)
# plt.show()

# plt.axis('equal')

# %% get peak wavenumber, power and spreed
def get_fft(y , dx):
    ys = np.copy(y)
    ys[np.isnan(ys)] =0
    k_fft = np.fft.rfftfreq(ys.size, d=dx) * 2* np.pi
    return np.fft.rfft(ys), k_fft


B1_fft, k_fft = get_fft(GG0[ GG0['y_prime'] == 0 ]['slopes'], dx)
B2_fft, k_fft = get_fft(GG0[ GG0['y_prime'] == 300 ]['slopes'], dx)

Z_fft      = (B1_fft+ B2_fft)/2
#Z_fft      = Z_fft[1:]
Z_fft_rm   = M.runningmean(abs(Z_fft), 2, tailcopy=True)

k_fft      = k_fft[~np.isnan(Z_fft_rm)]
Z_fft      = Z_fft[~np.isnan(Z_fft_rm)]
Z_fft_rm   = Z_fft_rm[~np.isnan(Z_fft_rm)]

Z_max_pos = Z_fft_rm.argmax()
k_fft_max  = k_fft[Z_max_pos]
Z_max      = Z_fft_rm[Z_max_pos]


# Z_sort = Z_fft_rm.argsort()[::-1]
# k_fft[Z_sort].shape
# K_fft_sel = k_fft[Z_sort][np.cumsum(Z_fft_rm[Z_sort])/np.sum(Z_fft_rm) <0.2]
# Z_fft_sel = Z_fft_rm[Z_sort][np.cumsum(Z_fft_rm[Z_sort])/np.sum(Z_fft_rm) <0.2]
#
# plt.plot(K_fft_sel, Z_fft_sel, '.')
#plt.plot(k_fft, Z_fft_rm )

# %
font_for_pres()
plt.plot(k_fft, abs(Z_fft) )
plt.plot(k_fft, Z_fft_rm )
treshold = np.nanmean(Z_fft_rm)  + np.nanstd(Z_fft_rm) *3
# k_fft_max_list  = k_fft[Z_fft_rm > treshold]
# Z_max_list      = Z_fft_rm[Z_fft_rm > treshold]

dk = np.diff(k_fft).mean()
k_interp = np.arange(k_fft[0], k_fft[-1], dk/4)
Z_interp = np.interp( k_interp, k_fft, abs(Z_fft_rm)  )
mask_interp = abs(Z_interp) > treshold
plt.plot(k_interp[mask_interp], Z_interp[mask_interp], '.')

k_list_interp , Z_list_interp = k_interp[mask_interp], Z_interp[mask_interp]

mask = abs(Z_fft) > treshold
k_fft_max_list  = k_fft[mask]
Z_max_list      = abs(Z_fft)[mask]

plt.plot(k_fft_max_list, Z_max_list, '.r', markersize= 20)


def gaus(x, x_0, amp, sigma_g ):
    return amp* np.exp(-0.5 * (  (x-x_0)/sigma_g)**2)

k_sigma =0.003

plt.plot(k_fft, gaus(k_fft,  k_fft_max, Z_max , k_sigma ), 'r', linewidth = 2 )


k_list_gauss = np.arange(k_fft_max - 1.5* k_sigma, k_fft_max + 1.5* k_sigma+ np.diff(k_fft).mean(), np.diff(k_fft).mean()/2)
Z_max_list_gauss =gaus(k_list_gauss,  k_fft_max, Z_max , k_sigma )
plt.plot(k_list_gauss, Z_max_list_gauss, 'b.', linewidth = 2 )
plt.xlim(0, 0.09)
M.save_anyfig(plt.gcf(), name= key_name+ '_fft', path= plot_path )

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

# %%

def get_z_model(x_positions, y_position, K_prime, K_amp,  alpha_rad, group_phase):

    K_abs = K_prime   / np.cos(alpha_rad)

    k = K_abs * np.cos(alpha_rad)
    l = K_abs * np.sin(alpha_rad)

    return wavemodel( x_positions,y_position, k, l, np.array(K_amp ), group_phase= group_phase)



def objective_func(pars, x, y, z, test_flag= False , alpha_prior= None ):

    z_model = get_z_model(x, y, pars['K_prime'], pars['K_amp'], pars['alpha'],pars['group_phase'])
    if alpha_prior is not None:
        penalties = np.array([ abs(alpha_prior - pars['alpha']) / (5 * np.pi/ 180) ])
    else:
        penalties =  np.array([0])

    cost =( abs(z - z_model) )**2  /z.std()**2
    if test_flag:
        return z_model
    else:
        return np.concatenate([cost , 2 * penalties])

#objective_func(fitter.params, *fitting_args , test_flag= False, alpha_prior = -np.pi/4)

def plot_brute_force(fitter_brute):

    clevel = np.linspace(0.4, 1.2, 30)
    plt.contourf(fitter_brute.brute_grid[1,:,:], fitter_brute.brute_grid[0,:,:], fitter_brute.brute_Jout/fitter_brute.brute_Jout.mean() , clevel, cmap= plt.cm.YlGnBu_r )

    plt.colorbar()
    plt.scatter(fitter_brute.brute_grid[1,:,:], fitter_brute.brute_grid[0,:,:], s=0.2, alpha= 0.4, color='black')
    plt.plot(fitter_brute.brute_x0[1], fitter_brute.brute_x0[0], '+r', markersize=20, label= 'Brute force')

    plt.xlabel('Phase (rad)')
    plt.ylabel('Angle (rad)')

    plt.legend()

def plot_instance(GG3, data_key, model_key, non_dim=False , view_scale = 0.3, fitter=None, title_str= None):

    import itertools
    F = M.figure_axis_xy(5,6, view_scale = view_scale, container = True)
    plt.suptitle(title_str)
    gs = GridSpec(4, 2,  wspace=0.3,  hspace=1.2)#figure=fig,

    #beam_list = list(set(GG3['beam']))
    beam_list = [0, 300]

    col_list = itertools.cycle([col.cascade2, col.rascade2, col.cascade1, col.rascade1, col.cascade3, col.rascade3])


    subz = len(beam_list)
    for beam, pos in zip(beam_list, [ gs[0, :] , gs[1, :] ] ):

        F.ax2 = F.fig.add_subplot(pos)
        plt.title( 'y=' + str(beam)  , loc='left')
        #plt.plot(GG['x_prime'], GG['z_model'], '.' )

        GGsel =GG3[GG3['y_prime'] == beam]
        plt.plot(GGsel['dist'], GGsel[model_key],  c=col.gray, linewidth = 1)

        if non_dim:
            # normalized version
            plt.plot(GGsel['dist'], GGsel['slopes']/GGsel['slopes'].std() , '-',  linewidth = 0.5, c=next(col_list))
        else:
            # diumensional version
            plt.plot(GGsel['dist'], GGsel[data_key] , '-', c=next(col_list), linewidth = 0.5)
        plt.ylabel('slope (m/m)')

    plt.xlabel('meter')

    F.ax3 = F.fig.add_subplot(gs[2:, :])
    if fitter is not None:
        plt.title('Brute-force costs', loc='left')
        plot_brute_force(fitter)

    return F

#plot_instance(GG, 'slopes', 'z_model' , view_scale= 0.7, fitter= fitter_brute, title_str='sdgjfds')


# %%
# dimensional version
N = GG.shape[0]
amp_Z = 2* abs(Z_max)**2 /N

import lmfit as LM
params0 = LM.Parameters()
params0.add('alpha', 0       ,  vary=True  , min=-0.95 * np.pi /2, max=0.95 * np.pi /2)
params0.add('group_phase', 0       ,  vary=True  , min=0, max= 2*np.pi)

# test: this should be 0
np.isnan(GG['slopes']).sum()
fitting_args = (GG['x_prime'],GG['y_prime'], GG['slopes']/GG['slopes'].std(), )
fitting_kargs = {'alpha_prior': None}

N_data = GG['x_prime'].size
angle_list  = list()
#fitting_kargs = {'size' :1}
L = pd.DataFrame(index=['alpha', 'group_phase', 'K_prime', 'K_amp'] )

Z_indexes = np.argsort(Z_max_list_gauss)[::-1]
cost_stack = dict()

#for k_prime_max,Z_max in zip(k_list_gauss[Z_indexes],Z_max_list_gauss[Z_indexes] ):
N_grid = 90
#for k_prime_max,Z_max in zip(k_fft_max_list,Z_max_list):
for k_prime_max,Z_max in zip(k_list_interp , Z_list_interp):

    print(k_prime_max)
    amp_enhancement = 1
    amp_Z = 1 #amp_enhancement * abs(Z_max)**2 /N

    params = params0.copy()
    params.add('K_prime', k_prime_max ,  vary=False  , min=k_prime_max*0.5, max=k_prime_max*1.5)
    params.add('K_amp', amp_Z         ,  vary=False  , min=amp_Z*.5       , max=amp_Z*5)

    #fitter = LM.minimize(objective_func, params, args=fitting_args, method='dual_annealing',max_nfev=None)
    fitter_brute = LM.minimize(objective_func, params, \
                    args=fitting_args,kws=fitting_kargs ,  method='brute', Ns=N_grid, )
    fitter = LM.minimize(objective_func, fitter_brute.params, \
                    args=fitting_args,kws=fitting_kargs ,  method='differential_evolution',max_nfev=None)

    GG['z_model'] = objective_func(fitter.params, *fitting_args , test_flag= True)
    angle_list.append(fitter.params['alpha'].value)
    #fitting_kargs = {'alpha_prior':  np.mean(angle_list )}
    fitting_kargs = {'alpha_prior':  None}
    F = plot_instance(GG, 'slopes', 'z_model', fitter = fitter_brute, view_scale = 0.5, non_dim=True )

    plt.plot(fitter.params['group_phase'].value, fitter.params['alpha'].value, '.r', markersize=20)
    print(fitting_kargs)
    print(fitter.params.pretty_print())
    plt.show()

    cost_stack[k_prime_max] = xr.DataArray( fitter_brute.brute_Jout/N_data, dims= ('angle', 'phase'),  coords = {'angle':np.linspace(-np.pi/2, np.pi/2, N_grid), 'phase':np.linspace(0, 2* np.pi, N_grid) } )
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

#weights = xr.DataArray(Z_max_list_gauss, dims ='k', coords = {'k': k_list_gauss})
weights = xr.DataArray(Z_max_list, dims ='k', coords = {'k': k_fft_max_list})
weights2 = 1/ cost_stack_rolled.sel(phase=0, method = 'nearest').min('angle')
#plt.plot(weights)
data_normed  = cost_stack_rolled #/cost_stack_rolled.std(['angle', 'phase'])
cost_wmean  = (weights * data_normed /weights.sum() ).sum('k')
# %% cross correlation

A1 = GG[GG['y_prime'] == 0]
A2 = GG[GG['y_prime'] == 300]

def autocrosscorr_func_1d(dd1, dd2):
    "takes data s as xarray"

    #print(dim)
    xp1=(dd1-dd1.mean())
    xp2=(dd2-dd2.mean())
    var=xp2.var()
    corr=np.correlate(xp1,xp2,'full')[len(xp1)-1:]/var/len(xp1)
    return corr

ymean = abs(A1['y_prime'].mean() - A2['y_prime'].mean())
angle_rad = np.concatenate([ -np.arctan2(A2['x_prime'], ymean)[::-1], np.arctan2(A2['x_prime'], ymean) ])
cross_corr = np.concatenate([autocrosscorr_func_1d(A1['slopes'][::-1], A2['slopes'][::-1])[::-1], autocrosscorr_func_1d(A1['slopes'], A2['slopes']) ])

#plt.plot( angle_rad, cross_corr )


# %%

F = M.figure_axis_xy(6, 6.5, view_scale= 0.5)
ax = plt.subplot(3, 1, 1)
plt.title( 'Normalized weighted mean cost at maximum phase', loc='left')
dn_unweighted = data_normed.mean('k')

plt.plot(data_normed.angle* 180/np.pi, M.normalize(dn_unweighted.sel(phase=0, method = 'nearest')  ) , label='mean')
plt.plot(cost_wmean.angle* 180/np.pi , M.normalize( cost_wmean.sel(phase=0, method = 'nearest') )    , label='weighted mean' )

def get_min_values(field2d, x, y):
    min_pos = np.unravel_index(field2d.argmin(), field2d.shape)
    return x[min_pos[0]] , y[min_pos[1]]

angle_mean, phase_mean  = get_min_values(cost_wmean,cost_wmean.angle, cost_wmean.phase )

phase_mean, angle_mean
angle_mean.data * 180 /np.pi

ax.axvline(angle_true* 180 /np.pi, linewidth = 1, color = 'black', label ='True')

plt.legend()
plt.xlabel('Angle (deg)')
plt.ylabel('normalized cost')
plt.xlim(-80, 80)


ax = plt.subplot(3, 1, 2)
plt.title( 'minus Lagged cross-correlation', loc='left')
plt.plot( angle_rad* 180 /np.pi,  - cross_corr )

ax.axvline(angle_true* 180 /np.pi, linewidth = 1, color = 'black', label ='True')

plt.legend()
plt.xlabel('Angle (deg)')
plt.ylabel('corr')
plt.xlim(-80, 80)


ax = plt.subplot(3, 1, 3)

plt.plot( angle_rad* 180 /np.pi,  1/np.cos(angle_rad) , c='k' )
ax.axvline(angle_true* 180 /np.pi, linewidth = 1, color = 'black', label ='True')

plt.title( 'Angle correction factor for wavenumber (1/cos($\\alpha$))', loc='left')
plt.xlabel('Angle (deg)')
plt.ylabel('k')
plt.grid()
plt.xlim(-80, 80)
plt.ylim(0.9, 3)

F.save_light(path= plot_path, name = key_name + '_rolled_pahse_weighted_mean')

# %%

F = M.figure_axis_xy(5, 4.5, view_scale = 0.5)
# L.T['alpha'].hist(bins = 60)
# F.ax.axvline(alpha_true, linewidth = 0.8, color = 'black')
xlims= [L.index.min()* 0.7, L.index.max()* 1.6]

ax0 = plt.subplot(2, 1, 1)
plt.plot( L.index,  L['alpha'], '.r', markersize = 10 , zorder=12)

plt.text( np.mean(xlims), angle_true + 0.08, 'True $\\alpha$=' + str(np.round(angle_true, 2)))
ax0.axhline(angle_true, linewidth = 1, color = 'black')

ax0.axhline(angle_mean, linewidth = 1, color = 'blue')
plt.text( np.mean(xlims)/1.5, angle_mean.data - 0.2, 'mean $\\alpha$=' + str(np.round(angle_mean.data , 2)))

for ang, aamp  in zip(np.arctan2(ll, kk), kk ):
    #ax0.axhline(ang, linewidth = 0.8,alpha=0.3,  color = 'black')
    plt.plot(aamp, ang, '.k', markersize= 9)

plt.xlim(xlims[0], xlims[1])
plt.ylim(-np.pi/2, np.pi/2)

plt.subplot(2, 1, 2)

plt.plot(k_fft, abs(Z_fft)/abs(Z_fft).mean() , label = 'normalized Power')
plt.plot(k_fft, Z_fft_rm/Z_fft_rm.mean() , label = 'smoothed')
plt.plot(k_list_interp,k_list_interp * 0 + k_list_interp.mean(), '.r', markersize= 10, zorder=12)
#weights2 = 1/ cost_stack_rolled.sel(phase=0, method = 'nearest').min('angle')
(weights/Z_fft_rm.mean()).plot(label = 'weights')
#plt.plot(k_list_gauss,k_list_gauss * 0 +Z_max_list_gauss.mean(), '.r', markersize= 10)
#plt.plot(k_fft_max_list,Z_max_list, '.r', markersize= 10)

plt.legend()
plt.xlim(xlims[0], xlims[1])

F.save_light(path= plot_path, name = key_name + 'single_results')
