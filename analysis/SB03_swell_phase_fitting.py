
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
track_name, batch_key, test_flag = '20190219073735_08070210_004_01', 'SH_batch02', False




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

Lpoints = int(np.round(min_datapoint) * 20)
Lmeters =Lpoints  * dx

def get_stacil_data(stancil, G3, key = 'dist' ):

    mask = (G3[key] >= stancil[0]) & (G3[key] < stancil[2])
    return G3[np.array(mask)]

def make_slopes(G3, key = 'heights_c_weighted_mean', spreed =10, verbose= False, photon_min =5):

    dd      = np.copy(G3['heights_c_weighted_mean'])
    dd      = np.gradient(dd)
    dd, _   = spicke_remover.spicke_remover(dd, spreed=spreed, verbose=verbose)
    dd_nans = (np.isnan(dd) ) + (G3['N_photos'] <= photon_min)

    return dd, dd_nans

stancil_iter = spec.create_chunk_boundaries_unit_lengths(Lmeters, xlims, ov= None , iter_flag=False)

# %%
sys.path.append(mconfig['paths']['analysis'])
import SB04_2d_wavefield_emulator as WaveEmulator

# test wave model
k_abs = 2 * np.pi/100
imp.reload(WaveEmulator)
kk, ll, amps, mesh_shape = WaveEmulator.get_stancils_polar(k_abs,0, size=1, dk= k_abs/10, amp_std=k_abs/20  )

x=np.arange(-250, 250, 0.5)
y=np.arange(-200, 200, 0.5)

def wavemodel(XX, YY, ks, ls, amps, group_phase = 0, amp_height= 1):

    import numpy as np

    G = np.vstack([ np.cos(np.outer(XX, ks) + np.outer(YY, ls) ).T ,  np.sin(np.outer(XX, ks) + np.outer(YY, ls) ).T ] ).T

    #phase1 = np.random.rand(1, amp_list.size) *  np.pi*2
    #phase = np.arange(0, amp_list.size) *  np.pi/2

    b = np.hstack([ np.cos(group_phase)*amps, np.sin(group_phase) *amps]).squeeze() * amp_height
    z_model = (G @ b)

    return z_model

Nx, Ny = x.size, y.size
XX, YY = np.meshgrid(x, y)
XX, YY = XX.reshape(XX.size), YY.reshape(YY.size)

z = wavemodel(XX, YY, kk, ll, amps).reshape(Ny, Nx)
plt.contourf(z)

# %% select data
Gd.keys()

def get_beam(G3, key):
    G1 = G3[key]
    G1['slopes'], _ = make_slopes(G1)
    G1['beam']= key
    return G1

B1= get_beam(Gd, 'gt3r')
B2= get_beam(Gd, 'gt3l')

dist_y_mean = np.sqrt( (B1['x']-B2['x'])**2 + (B1['y']-B2['y'])**2 ).mean()

B1['dist_y'] = 0
B2['dist_y'] = dist_y_mean

GG = pd.concat( [B1, B2] )
#GG = get_beam(Gd, 'gt3r')

stancil = stancil_iter[:, 30]
GG = get_stacil_data(stancil, GG).sort_values(['beam','dist'])

# %%
GG['x_prime'] = GG['x'] - GG['x'].min()
GG['y_prime'] = GG['y'] - GG['y'].max()
# plt.plot(GG['x_prime'], GG['y_prime'] , '.', markersize = 0.5)
# plt.axis('equal')

# %% get peak wavenumber, power and spreed

G4 = GG
def get_fft(G4, beam, dx):
    G4[G4['beam'] == beam]
    ys = np.copy(G4['slopes'])
    ys[np.isnan(ys)] =0
    k_fft = np.fft.rfftfreq(ys.size, d=dx) * 2* np.pi
    return np.fft.rfft(ys), k_fft

B1_fft, k_fft = get_fft(GG, 'gt3r', dx)
B2_fft, k_fft = get_fft(GG, 'gt3l', dx)

Z_fft      = (B1_fft+ B2_fft)/2
Z_fft_rm   = M.runningmean(abs(Z_fft), 2, tailcopy=True)
Z_max_pos = Z_fft_rm[~np.isnan(Z_fft_rm)].argmax()
k_fft_max  = k_fft[Z_max_pos]
Z_max      = Z_fft[Z_max_pos]

font_for_pres()
plt.plot(k_fft, abs(Z_fft) )
plt.plot(k_fft, Z_fft_rm )
plt.plot(k_fft_max, Z_fft_rm[Z_max_pos], '.', markersize= 20)

def gaus(x, x_0, amp, sigma_g ):
    return amp* np.exp(-0.5 * (  (x-x_0)/sigma_g)**2)

dk =0.002
plt.plot(k_fft, gaus(k_fft,  k_fft_max, Z_max , dk ), 'r', linewidth = 2 )
plt.xlim(0, 0.1)

# %%
k_prime_abs = k_fft_max#2 * np.pi/ 200
imp.reload(WaveEmulator)

alpha = 60
for group_phase in np.linspace(0, 2* np.pi,4):

    akpha_rad= alpha *np.pi/180
    k_abs = k_prime_abs   / np.cos(akpha_rad)
    print(k_abs, k_prime_abs)
    kk, ll, amps, mesh_shape = WaveEmulator.get_stancils_polar(k_abs,  akpha_rad , size=0 , dk= dk , plot_flag= False)

    print('k mean ', kk.mean(), 'l mean ', ll.mean() )
    # x2 = np.arange( GG['x_prime'].min(), GG['x_prime'].max(), 1)
    # y2 = x2 * 0
    # x2, y2 = np.concatenate([x2, x2]),     np.concatenate([y2, y2+60])
    # #z_model = wavemodel( x2, y2 , kk, ll, amps)

    N = GG.shape[0]
    #abs(Z_max)**2/ N

    # normalized version
    #amp_Z = abs(Z_max)/np.sqrt(2)/2
    # dimensional version
    amp_Z = 2* abs(Z_max)**2 /N
    GG['z_model'] = wavemodel( GG['dist'],GG['dist_y'], kk, ll, amps, group_phase= group_phase, amp_height= amp_Z)

    M.figure_axis_xy(6, 5.5, view_scale = 0.5)

    beam = 'gt3l'
    plt.subplot(2, 1, 1)
    plt.title(beam +'  ' + str(alpha) )
    #plt.plot(GG['x_prime'], GG['z_model'], '.' )

    GGsel =GG[GG['beam'] == beam]
    plt.plot(GGsel['dist'], GGsel['z_model'],  c=col.gray)

    # diumensional version
    plt.plot(GGsel['dist'], GGsel['slopes'] , '-', c=col.cascade2)

    # normalized version
    #plt.plot(GGsel['dist'], GGsel['slopes']/GGsel['slopes'].std() , '-', c=col.cascade2)

    beam = 'gt3r'
    plt.subplot(2, 1, 2)
    plt.title(beam +'  ' + str(alpha) )
    GGsel =GG[GG['beam'] == beam]
    plt.plot(GGsel['dist'], GGsel['z_model'],  c=col.gray)
    plt.plot(GGsel['dist'], GGsel['slopes'] , '-', c=col.rascade2)

    plt.show()

# %%


#abs(Z_max)**2/ N

# normalized version
#amp_Z = abs(Z_max)/np.sqrt(2)/2

# dimensional version
amp_Z = 2* abs(Z_max)**2 /N

def get_z_model(x_positions, y_position, K_prime, K_amp,  alpha_rad, group_phase, dk, size =0):

    K_abs = K_prime   / np.cos(alpha_rad)
    kk, ll, amps, mesh_shape = WaveEmulator.get_stancils_polar(K_abs,  alpha_rad , size=size , dk= dk , plot_flag= False)

    return wavemodel( x_positions,y_position, kk, ll, amps, group_phase= group_phase, amp_height= K_amp)



def objective_func(pars, x, y, z, test_flag= False ):

    z_model = get_z_model(x, y, pars['K_prime'], pars['K_amp'],pars['alpha'],pars['group_phase'], pars['dk'], size=pars['model_size'])
    cost =( abs(z - z_model) )**2  /z.std()**2
    if test_flag:
        return z_model
    else:
        return cost


def plot_instance(GG3, data_key, model_key, non_dim=False ):

    import itertools
    F = M.figure_axis_xy(8, 5, view_scale = 0.5)

    beam_list = list(set(GG3['beam']))


    col_list = itertools.cycle([col.cascade2, col.rascade2, col.cascade1, col.rascade1, col.cascade3, col.rascade3])
    subz = len(beam_list)
    for beam, i in zip(beam_list, np.arange(1, subz+1, 1)):

        plt.subplot(subz, 1, i)
        plt.title(beam  )
        #plt.plot(GG['x_prime'], GG['z_model'], '.' )

        GGsel =GG3[GG3['beam'] == beam]
        plt.plot(GGsel['dist'], GGsel[model_key],  c=col.gray)

        if non_dim:
            # normalized version
            plt.plot(GGsel['dist'], GGsel['slopes']/GGsel['slopes'].std() , '-', c=next(col_list))
        else:
            # diumensional version
            plt.plot(GGsel['dist'], GGsel[data_key] , '-', c=next(col_list))

    return F



#GG = pd.concat( [B1, B2] )
GG = pd.concat( [B1, B2] )

stancil = stancil_iter[:, 30]
GG = get_stacil_data(stancil, GG).sort_values(['beam','dist'])
#GG['slopes'] = zm
slopes1 = GG[GG['beam'] == 'gt3r']['slopes']
GG['slopes'] = np.concatenate([slopes1, slopes1])



# %%


import lmfit as LM
params = LM.Parameters()
params.add('K_prime', k_prime_abs ,  vary=False  , min=k_prime_abs*0.5, max=k_prime_abs*1.5)
params.add('K_amp', amp_Z         ,  vary=False  , min=amp_Z*.5, max=amp_Z*5)
params.add('alpha', np.pi/4       ,  vary=True  , min=-0.95 * np.pi /2, max=0.95 * np.pi /2)
params.add('group_phase', 0       ,  vary=True  , min=0, max= 2*np.pi)
params.add('dk', dk               ,  vary=False  , min=1e-4, max= 0.05)
params.add('model_size', 0        ,  vary=False  , min=0, max= 2)


# test: this should be 0
zm   =get_z_model( GG['dist'],GG['dist_y'], k_prime_abs, amp_Z,  np.pi/4, 0 , dk)
objective_func(params, GG['dist'],GG['dist_y'], zm)

objective_func(params, GG['dist'],GG['dist_y'], GG['slopes'])

fitting_args = (GG['dist'],GG['dist_y'], GG['slopes'])
#fitting_kargs = {'size' :1}

fitter = LM.minimize(objective_func, params, args=fitting_args, method='brute', Ns=100, workers=4, max_nfev=None)
GG['z_model'] = objective_func(fitter.params, GG['dist'],GG['dist_y'], GG['slopes'] , test_flag= True)
plot_instance(GG, 'slopes', 'z_model' )
fitter


# %%

params2 = fitter.params.copy()
params2['model_size'].value = 0
params2['K_prime'].vary = True
#params2['dk'].vary      = False
#params2['dk'].value = params2['K_prime']/20

#params2['alpha'].vary       = False
params2['group_phase'].vary = True
#params2['K_amp'].vary = True


fitter2 = LM.minimize(objective_func, params2, args=fitting_args, method='dual_annealing',  max_nfev=None)
#fitter3 = LM.minimize(objective_func, params2, args=fitting_args, method='emcee', nwalkers=100, steps=1500)
GG['z_model'] = objective_func(fitter2.params, GG['dist'],GG['dist_y'], GG['slopes'] , test_flag= True)
plot_instance(GG, 'slopes', 'z_model' )
fitter2


# %% rotation test
#GG = pd.concat( [B1, B2] )
GG = pd.concat( [B1, B2] )

stancil = stancil_iter[:, 30]
GG = get_stacil_data(stancil, GG).sort_values(['beam','dist'])
#GG['slopes'] = zm
slopes1 = GG[GG['beam'] == 'gt3r']['slopes']
GG['slopes'] = np.concatenate([slopes1, np.roll(slopes1, int( (0 *np.pi/4) * (2*np.pi/ k_prime_abs)/dx ))])

import lmfit as LM
params = LM.Parameters()
params.add('K_prime', k_prime_abs ,  vary=False  , min=k_prime_abs*0.5, max=k_prime_abs*1.5)
params.add('K_amp', amp_Z         ,  vary=False  , min=amp_Z*.5, max=amp_Z*5)
params.add('alpha', np.pi/4       ,  vary=True  , min=-0.95 * np.pi /2, max=0.95 * np.pi /2)
params.add('group_phase', 0       ,  vary=True  , min=0, max= 2*np.pi)
params.add('dk', dk               ,  vary=False  , min=1e-4, max= 0.05)
params.add('model_size', 0        ,  vary=False  , min=0, max= 2)


# test: this should be 0
zm   =get_z_model( GG['dist'],GG['dist_y'], k_prime_abs, amp_Z,  np.pi/4, 0 , dk)
objective_func(params, GG['dist'],GG['dist_y'], zm).sum()

objective_func(params, GG['dist'],GG['dist_y'], GG['slopes']).sum()

fitting_args = (GG['dist'],GG['dist_y'], GG['slopes'])
#fitting_kargs = {'size' :1}
fitter = LM.minimize(objective_func, params, args=fitting_args, method='dual_annealing',  max_nfev=None)
GG['z_model'] = objective_func(fitter.params, GG['dist'],GG['dist_y'], GG['slopes'] , test_flag= True)
plot_instance(GG, 'slopes', 'z_model' )
fitter
# %%

params2 = fitter.params.copy()
params2['K_prime'].vary = True
GG['slopes_smth'] =  M.runningmean_wrap_around(np.array(GG['slopes']), 2)
fitter2 = LM.minimize(objective_func, params, args=fitting_args, method='dual_annealing',  max_nfev=None)
#fitter3 = LM.minimize(objective_func, params2, args=fitting_args, method='emcee', nwalkers=100, steps=1500)
GG['z_model'] = objective_func(fitter2.params, GG['dist'],GG['dist_y'], GG['slopes_smth'] , test_flag= True)
plot_instance(GG, 'slopes_smth', 'z_model' )
fitter2
# %%
