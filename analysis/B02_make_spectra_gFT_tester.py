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

import imp
import copy
import spicke_remover
import datetime
import generalized_FT as gFT
from scipy.ndimage.measurements import label

# %%

def recreate_fft_weights(GG_xi, k):
    GG_xi.y_data[np.isnan(GG_xi.y_data)] = 0
    y_gridded = GG_xi.y_data

    # take FFT to get peaj parameters
    k_fft = np.fft.rfftfreq(GG_xi.eta.size, d=dx) * 2* np.pi
    f_weight= np.sqrt(9.81 * k_fft) / (2 *np.pi)
    data_fft = spec.Z_to_power(np.fft.rfft(y_gridded), np.diff(f_weight).mean(), GG_xi.eta.size)


    # create JONSWAP weight
    Spec_fft = gFT.get_prior_spec(f_weight, data_fft )
    f= np.sqrt(9.81 * k) / (2 *np.pi)
    weight = Spec_fft.create_weight(freq = f, plot_flag= False, max_nfev=None)

    return k_fft, data_fft, k, weight

#import s3fs
# %%
#track_name, batch_key, test_flag = io.init_from_input(sys.argv) # loads standard experiment
#track_name, batch_key, test_flag = '20190605061807_10380310_004_01', 'SH_batch01', False
#track_name, batch_key, test_flag = '20190601094826_09790312_004_01', 'SH_batch01', False
#track_name, batch_key, test_flag = '20190207111114_06260210_004_01', 'SH_batch02', False
#track_name, batch_key, test_flag = '20190208152826_06440210_004_01', 'SH_batch01', False
#track_name, batch_key, test_flag = '20190215184558_07530210_004_01', 'SH_batch01', False


ID_name, batch_key, ID_flag = 'SH_20190502_05160312', 'SH_publish', True
#ID_name, batch_key, ID_flag = 'SH_20190224_08800210', 'SH_publish', True
ID, _, hemis, batch = io.init_data(ID_name, batch_key, ID_flag, mconfig['paths']['work'],  )

#print(track_name, batch_key, test_flag)
hemis, batch = batch_key.split('_')
#track_name= '20190605061807_10380310_004_01'
ATlevel= 'ATL03'

load_path   = mconfig['paths']['work'] + batch_key +'/B01_regrid/'
load_file   = load_path + 'processed_' + ATlevel + '_' + ID_name + '.h5'

save_path   = mconfig['paths']['work'] + '/B02_spectra_'+hemis+'/'
save_name   = 'B02_'+ID_name

plot_path   = mconfig['paths']['plot'] + '/'+hemis+'/'+batch_key+'/' + ID_name + '/B_spectra/'
MT.mkdirs_r(plot_path)
MT.mkdirs_r(save_path)
bad_track_path =mconfig['paths']['work'] +'bad_tracks/'+ batch_key+'/'
# %%

all_beams   = mconfig['beams']['all_beams']
high_beams  = mconfig['beams']['high_beams']
low_beams   = mconfig['beams']['low_beams']
#Gfilt   = io.load_pandas_table_dict(ID_name + '_B01_regridded', load_path) # rhis is the rar photon data
Gd      = io.load_pandas_table_dict(ID_name + '_B01_binned_' , load_path)  #
Gd = h5py.File(load_path +'/'+ID_name + '_B01_binned.h5', 'r')

# %% test amount of nans in the data
nan_fraction= list()
for I in Gd.values():
    nan_fraction.append( np.sum(np.isnan(I['heights_c_std'])) / I['heights_c_std'].shape[0] )


if np.array(nan_fraction).mean() > 0.95:
    print('nan fraction > 95%, pass this track, add to bad tracks')
    MT.json_save(ID_name, bad_track_path, {'nan_fraction': np.array(nan_fraction).mean(), 'date': str(datetime.date.today()) })
    print('exit.')
    exit()

# %% test LS with an even grid where missing values are set to 0
imp.reload(spec)
print(Gd.keys())
Gi =Gd[ list(Gd.keys())[0] ] # to select a test  beam

# derive spectal limits
# Longest deserved period:
T_max       = 40 #sec
k_0         = (2 * np.pi/ T_max)**2 / 9.81
x           = np.array(Gi['dist'])
dx          =  np.diff(x).mean()
min_datapoint =  2*np.pi/k_0/dx

Lpoints     = int(np.round(min_datapoint) * 10 )
Lmeters     = Lpoints  * dx

#plt.plot(np.diff(np.array(Gi['dist'])))
print('L number of gridpoint:', Lpoints)
print('L length in km:', Lmeters/1e3)
print('approx number windows', 2* Gi['dist'][:][-1] /Lmeters-1   )

T_min       = 5
lambda_min  = 9.81 * T_min**2/ (2 *np.pi)
flim        = 1/T_min

oversample  = 2
dlambda     = Lmeters * oversample
dk          = 2 * np.pi/ dlambda
kk          =np.arange(0, 1/lambda_min,  1/dlambda) * 2*np.pi
kk          = kk[k_0<=kk]
#dk = np.diff(kk).mean()
print('2 M = ',  kk.size *2 )

dk *150
# %%

G_gFT= dict()
G_gFT_x = dict()
G_rar_fft= dict()
Pars_optm = dict()
imp.reload(spec)
imp.reload(gFT)

k=all_beams[1]
#for k in all_beams:

# -------------------------------  use gridded data
hkey= 'heights_c_weighted_mean'
x       = Gd[k]['dist'][:]
xlims   = x[0], x[-1]
dd      = np.copy(Gd[k][hkey])

dd_error = np.copy(Gd[k]['heights_c_std'])
dd_error[np.isnan(dd_error)] = 100
#plt.hist(1/dd_weight, bins=40)

F = M.figure_axis_xy(6, 3)
plt.subplot(2, 1, 1)
#plt.plot(x, dd, 'gray', label='displacement (m) ')

# compute slope spectra !!
dd      = np.gradient(dd)
dd, _   = spicke_remover.spicke_remover(dd, spreed=10, verbose=False)
dd_nans = (np.isnan(dd) ) + (Gd[k]['N_photos'][:] <= 5)

dd_filled = np.copy(dd)
dd_filled[dd_nans] = 0
#win = create_weighted_window(dd_filled)

# using gappy data
Nsel= 10000
dd_no_nans = dd[~dd_nans][0:Nsel] # windowing is applied here
x_no_nans  = x[~dd_nans][0:Nsel]
dd_error_no_nans = dd_error[~dd_nans][0:Nsel]

plt.plot(x_no_nans, dd_no_nans, '.', 'black', label='slope (m/m)')
plt.legend()
plt.show()



# %%

xlims = xlims[0], xlims[0] + (xlims[1] -xlims[0])/2

print('gFT')
font_for_print()
#S_pwelch_k2 = np.arange(S_pwelch_k[1], S_pwelch_k[-1], S_pwelch_dk*2 )
imp.reload(gFT)
S = gFT.wavenumber_spectrogram_gFT( np.array(x_no_nans), np.array(dd_no_nans), Lmeters, dx, kk, data_error = dd_error_no_nans,  ov=None)
GG, GG_x, Params = S.cal_spectrogram(xlims= xlims, max_nfev = None, plot_flag = True)


# %%

xsel= 0
GG_xi = GG_x.isel(x=xsel)
GGi = GG.isel(x=xsel)

fft_k, fft_data, k, weight = recreate_fft_weights(GG_xi, GGi.k)

font_for_pres()
F= M.figure_axis_xy(6, 3, view_scale= 0.8)
GGi.gFT_PSD_data.plot(color ='red', lw= 0.8, label= 'GFT')
GGi.weight.plot(color = 'green', label= 'weight')

plt.plot(fft_k, fft_data, 'gray', lw=0.5, label= 'FFT', zorder= 3)
plt.plot(k, weight, 'k--', label= 'fitten PM model')

plt.legend(ncol =2)
plt.xlim(k[0], k[-1])
plt.ylim(0, np.quantile(fft_data, 0.98))
plt.ylabel('PSD ( (m/m)$^2$/k)')
plt.xlabel('wavenumber (k)')


plt.title('x='+ str(np.round(GG_xi.x.data/1e3,1))+ 'km, N=' + str(int(GGi.N_per_stancil)) )

# %%


# np.nanmean( (GG.gFT_PSD_model.sum('k') *dk) )
# GG.gFT_PSD_model.median('x').plot()
# GG.gFT_PSD_data.median('x').rolling(k= 30).mean().plot()
# GG.gFT_PSD_model.median('x').rolling(k= 30).mean().plot()
# plt.plot(Params.T['alpha'], '.')
# GG.gFT_PSD_data.rolling(x=6, k =20, min_periods=5).median().plot()
# plt.plot(GG.k ,  GG.gFT_PSD_model.rolling(x=10, k =140, min_periods=5).mean() , 'k', alpha= 0.5)
# plt.ylim(0, .6)
#
# (GG.gFT_PSD_data).rolling(k= 100).median().plot()#levels= np.arange(0, 0.5, 0.05) )
# plt.ylim(0, 0.1)
#
# np.log(GG.gFT_PSD_data).rolling(k= 100).median().plot()#levels = np.arange(-4, 1, 0.25))
#
# #np.log(GG.gFT_PSD_data).plot()
#
# plt.plot(Params.T['alpha'], '.')
# %%

def linear_gap_fill(F, key_lead, key_int):

    """
    F pd.DataFrame
    key_lead   key in F that determined the independent coordindate
    key_int     key in F that determined the dependent data
    """
    y_g = np.array(F[key_int])

    nans, x2= np.isnan(y_g), lambda z: z.nonzero()[0]
    y_g[nans]= np.interp(x2(nans), x2(~nans), y_g[~nans])

    return y_g



plot_data_model=True
if plot_data_model:
    for i in np.arange(30,60,2):
        c1= 'blue'
        c2= 'red'

        GGi = GG.isel(x= i)

        xi_1=GG_x.x[i]
        xi_2=GG_x.x[i+1]
        #if k%2 ==0:

        F = M.figure_axis_xy(16, 2)
        eta  = GG_x.eta

        # gFT model
        y_model = GG_x.y_model[:, i]
        plt.plot(eta +xi_1, y_model ,'-', c=c1, linewidth=0.8, alpha=1, zorder=12)
        y_model = GG_x.y_model[:, i+1]
        plt.plot(eta +xi_2, y_model,'-', c=c2, linewidth=0.8, alpha=1, zorder=12)

        # iterpolated model in gaps
        FT = gFT.generalized_Fourier(eta +xi_1, None,GG.k )
        _ = FT.get_H()
        FT.b_hat=np.concatenate([ GGi.gFT_cos_coeff, GGi.gFT_sin_coeff ])
        plt.plot(eta +xi_1, FT.model() ,'-', c='orange', linewidth=0.8, alpha=1,zorder= 2)

        FT = gFT.generalized_Fourier(eta +xi_2, None,GG.k )
        _ = FT.get_H()
        FT.b_hat=np.concatenate([ GGi.gFT_cos_coeff, GGi.gFT_sin_coeff ])
        plt.plot(eta +xi_2, FT.model() ,'-', c='orange', linewidth=0.8, alpha=1,zorder= 2)


        # oringial data
        #plt.plot(x, dd, '-', c='k',linewidth=3,  alpha =0.6, zorder=11)
        #plt.plot(x, dd, '.', c='k',markersize=3,  alpha =0.5)

        # oringal data from model_output
        y_data = GG_x.y_data[:, i]
        plt.plot(eta +xi_1, y_data ,'-', c='k',linewidth=3,  alpha =0.3, zorder=11)
        y_data = GG_x.y_data[:, i+1]
        plt.plot(eta +xi_2, y_data,'-', c='k',linewidth=3,  alpha =0.3, zorder=11)


        #plt.plot(x[~dd_nans], dd_error[~dd_nans], '.', c='green',linewidth=1,  alpha =0.5)

        F.ax.axvline(xi_1 + eta[0].data , linewidth=4,  color=c1, alpha=0.5)
        F.ax.axvline(xi_1 + eta[-1].data, linewidth=4,  color=c1, alpha=0.5)
        F.ax.axvline(xi_2 + eta[0].data , linewidth=4,  color=c2, alpha=0.5)
        F.ax.axvline(xi_2 + eta[-1].data, linewidth=4,  color=c2, alpha=0.5)

        ylims= -np.nanstd(dd)*2, np.nanstd(dd)*2
        plt.text(xi_1 + eta[0].data, ylims[-1], '  N='+ str(GG.sel(x=xi_1, method='nearest').N_per_stancil.data) + ' N/2M= '+ str(GG.sel(x=xi_1, method='nearest').N_per_stancil.data/2/kk.size) )
        plt.text(xi_2 + eta[0].data, ylims[-1], '  N='+ str(GG.sel(x=xi_2, method='nearest').N_per_stancil.data) + ' N/2M= '+ str(GG.sel(x=xi_2, method='nearest').N_per_stancil.data/2/kk.size) )
        plt.xlim(xi_1 + eta[0].data*1.2, xi_2 + eta[-1].data*1.2 )
        #plt.xlim(xi_1, xi_2 )


        plt.ylim(ylims[0], ylims[-1])
        plt.show()

# %
#S.mean_spectral_error() # add x-mean spectal error estimate to xarray
S.parceval(add_attrs= True, weight_data=False)

# assign beam coordinate
GG.coords['beam'], GG_x.coords['beam']  = str(k), str(k)
GG, GG_x                                = GG.expand_dims(dim = 'beam', axis = 1), GG_x.expand_dims(dim = 'beam', axis = 1)
# repack such that all coords are associated with beam
GG.coords['N_per_stancil']              = (('x', 'beam' ), np.expand_dims(GG['N_per_stancil'], 1))

# add more coodindates to the Dataset
x_coord_no_gaps = linear_gap_fill( Gd[k], 'dist', 'x' )
y_coord_no_gaps = linear_gap_fill( Gd[k], 'dist', 'y' )
mapped_coords   = spec.sub_sample_coords(Gd[k]['dist'], x_coord_no_gaps, y_coord_no_gaps, S.stancil_iter , map_func = None )

GG.coords['x_coord'] = GG_x.coords['x_coord'] = (('x', 'beam' ), np.expand_dims(mapped_coords[:,1], 1) )
GG.coords['y_coord'] = GG_x.coords['y_coord'] = (('x', 'beam' ), np.expand_dims(mapped_coords[:,2], 1) )

# if data staarts with nans replace coords with nans again
if (GG.coords['N_per_stancil'] == 0).squeeze()[0].data:
    nlabel = label( (GG.coords['N_per_stancil'] == 0).squeeze())[0]
    nan_mask= nlabel ==nlabel[0]
    GG.coords['x_coord'][nan_mask] =np.nan
    GG.coords['y_coord'][nan_mask] =np.nan

lons_no_gaps  = linear_gap_fill( Gd[k], 'dist', 'lons' )
lats_no_gaps  = linear_gap_fill( Gd[k], 'dist', 'lats' )
mapped_coords = spec.sub_sample_coords(Gd[k]['dist'], lons_no_gaps, lats_no_gaps, S.stancil_iter , map_func = None )

GG.coords['lon'] = GG_x.coords['lon'] = (('x', 'beam' ), np.expand_dims(mapped_coords[:,1], 1) )
GG.coords['lat'] = GG_x.coords['lat'] = (('x', 'beam' ), np.expand_dims(mapped_coords[:,2], 1) )

# plt.plot(Gd[k]['x'], Gd[k]['y']  , '.')
# plt.plot(GG.coords['x_coord'], GG.coords['y_coord'] , '.r')
#
# plt.plot(Gd[k]['lons'], Gd[k]['lats']  , '.')
# plt.plot(GG.coords['lon'], GG.coords['lat'] , '.r')
# spectral errors are cacualted within S and now repacked to main DataSet G
#G.coords['mean_El'] = (('k', 'beam' ), np.expand_dims(S.G['mean_El'], 1))
#G.coords['mean_Eu'] = (('k', 'beam' ), np.expand_dims(S.G['mean_Eu'], 1))

# calculate number data points
def get_stancil_nans(stancil):
    x_mask = (stancil[0] < x) & (x <= stancil[-1])
    idata  = Gd[k]['N_photos'][x_mask]
    return stancil[1], idata.sum()

photon_list = np.array(list(dict(map(  get_stancil_nans,  copy.copy(S.stancil_iter) )).values()))
GG.coords['N_photons'] = (('x', 'beam' ), np.expand_dims(photon_list, 1))

# Save to dict
G_gFT[k]    = GG
G_gFT_x[k]  = GG_x
Pars_optm[k] = Params

# plot
plt.subplot(2, 1, 2)
G_gFT_power = GG.gFT_PSD_data.squeeze()
plt.plot(G_gFT_power.k, np.nanmean(G_gFT_power,1), 'gray', label='mean gFT power data ')
G_gFT_power = GG.gFT_PSD_model.squeeze()
plt.plot(GG.k, np.nanmean(S.G, 1), 'k', label='mean LS power model')

# standard FFT
print('FFT')
dd[dd_nans]    = 0
S = spec.wavenumber_spectrogram(x, dd, Lpoints)
G = S.cal_spectrogram()
S.mean_spectral_error() # add x-mean spectal error estimate to xarray
S.parceval(add_attrs= True)

# assign beam coordinate
G.coords['beam']      = str(k)#(('beam'), str(k))
G                     = G.expand_dims(dim = 'beam', axis = 2)
G.coords['mean_El']   = (('k', 'beam' ), np.expand_dims(G['mean_El'], 1))
G.coords['mean_Eu']   = (('k', 'beam' ), np.expand_dims(G['mean_Eu'], 1))
G.coords['x']         = G.coords['x'] * dx # adjust x-coodinate definition

stancil_iter = spec.create_chunk_boundaries(int(Lpoints), dd_nans.size)
def get_stancil_nans(stancil):
    idata = dd_nans[stancil[0]:stancil[-1]]
    return stancil[1], idata.size - idata.sum()

N_list = np.array(list(dict(map(  get_stancil_nans,  stancil_iter )).values()))

# repack such that all coords are associated with beam
G.coords['N_per_stancil'] = (('x', 'beam' ), np.expand_dims(N_list, 1))

#save to dict
G_rar_fft[k] = G

# for plotting
G_rar_fft_p =G.squeeze()
plt.plot(G.k, G_rar_fft_p[:, G_rar_fft_p['N_per_stancil'] > 10 ].mean('x'), 'darkblue', label='mean FFT')
#plt.plot(G.k, GG.mean('x'), 'lightblue', label='mean FFT')
plt.legend()
plt.show()

F.save_light(path=plot_path, name = 'B02_control_'+k+'_' + track_name)
#print('saved as '+'B02_control_'+k+'_' + track_name)
#print(np.isinf(G).sum().data)



# %%

def dict_weighted_mean(Gdict, weight_key):
    """
    returns the weighted meean of a dict of xarray, data_arrays
    weight_key must be in the xr.DataArrays
    """
    #Gdict = G_LS
    # weight_key='N_per_stancil'

    akey = list( Gdict.keys() )[0]
    GSUM = Gdict[akey].copy()
    GSUM.data     = np.zeros(GSUM.shape)
    N_per_stancil = GSUM.N_per_stancil * 0
    N_photons     = np.zeros(GSUM.N_per_stancil.size)

    counter= 0
    for k,I in Gdict.items():
        I =I.squeeze()
        GSUM                += I.where( ~np.isnan(I), 0) * I[weight_key] #.sel(x=GSUM.x)
        N_per_stancil       += I[weight_key]
        if 'N_photons' in GSUM.coords:
            N_photons    += I['N_photons']
        counter+=1

    GSUM             = GSUM  / N_per_stancil /counter

    if 'N_photons' in GSUM.coords:
        GSUM.coords['N_photons'] = (('x', 'beam'), np.expand_dims(N_photons, 1) )

    GSUM['beam'] = ['weighted_mean']
    GSUM.name='power_spec'

    return GSUM

G_gFT_sel = {}
for k,I in G_gFT.items():
    G_gFT_sel[k] = I['gFT_PSD_model']

G_gFT_wmean = dict_weighted_mean(G_gFT_sel, 'N_per_stancil')
G_fft_wmean = dict_weighted_mean(G_rar_fft, 'N_per_stancil')


# %% plot
def plot_wavenumber_spectrogram(ax, Gi, clev, title= None, plot_photon_density=True ):

    if Gi.k[0] ==0:
        Gi= Gi.sel(k=Gi.k[1:])
    x_lambda= 1/Gi.k
    plt.pcolormesh(Gi.x/1e3, x_lambda , Gi, cmap=plt.cm.ocean_r , vmin = clev[0], vmax = clev[-1])

    ax.set_yscale('log')
    # plt.colorbar(orientation='vertical', pad=0.06, label='Spectral Power (m^2/m)')

    if plot_photon_density:

        plt.plot(Gi.x/1e3, x_lambda[-1] + (Gi.N_per_stancil/Gi.N_per_stancil.max() ) * 10 , c='black', linewidth= 0.8, label='NAN-density' )
        plt.fill_between(Gi.x/1e3, x_lambda[-1] + (Gi.N_per_stancil/Gi.N_per_stancil.max() ) * 10,  0, color='gray', alpha = 0.3)
        ax.axhline(30, color='black', linewidth=0.3)

    #plt.xlabel('Distance from the Ice Edge (km)')
    plt.ylim(x_lambda[-1], x_lambda[0])
    plt.title(title, loc='left')

#Gplot = G.rolling(x=5, min_periods= 1, center=True).mean()
#Gmean = G_gFT_wmean.rolling(x=2, min_periods= 1, center=True).mean()
Gmean = G_gFT_wmean.rolling(k=5, center=True).mean()
#Gmean = Gmean.where(~np.isnan(Gmean), 0)
k_max_range = Gmean.k[Gmean.mean('x').argmax().data].data* 0.75, Gmean.k[Gmean.mean('x').argmax().data].data* 1, Gmean.k[Gmean.mean('x').argmax().data].data* 1.25

# %%
font_for_print()
F = M.figure_axis_xy(6.5, 5.6, container= True, view_scale =1)

plt.suptitle('LS and FFT Slope Spectrograms\n' + track_name, y = 0.98)
gs = GridSpec(3,3,  wspace=0.2,  hspace=.5)#figure=fig,
#clev=np.arange(0, 6, 0.1)*3

#%matplotlib inline

clev = M.clevels( [Gmean.quantile(0.8).data * 1e4, Gmean.quantile(0.99).data * 1e4], 31)/ 1e4

xlims= Gmean.x[0]/1e3, Gmean.x[-1]/1e3

for pos, k, pflag in zip([gs[0, 0],gs[0, 1],gs[0, 2] ], high_beams, [True, False, False] ):
    ax0 = F.fig.add_subplot(pos)
    Gplot = G_gFT_sel[k].squeeze()#.rolling(k=10, x=2, min_periods= 1, center=True).mean()
    #Gplot.mean('x').plot()

    # #plt.pcolormesh(G.x/1e3, 1/G.k , G, norm=colors.LogNorm(vmin=G.min()*1e6, vmax=G.max()), cmap='PuBu')#, vmin = clev[0], vmax = clev[-1])
    #
    #np.log(Gplot).plot()
    plot_wavenumber_spectrogram(ax0, Gplot,  clev, title =k,  plot_photon_density=True )
    plt.xlim(xlims)
    #
    if pflag:
        plt.ylabel('Wave length\n(meters)')
        plt.legend()

for pos, k, pflag in zip([gs[1, 0],gs[1, 1],gs[1, 2] ], low_beams, [True, False, False] ):
    ax0 = F.fig.add_subplot(pos)
    Gplot = G_gFT_sel[k].squeeze()#.rolling(k=10, x=2, min_periods= 1, center=True).mean()
    #Gplot.mean('x').plot()

    # #plt.pcolormesh(G.x/1e3, 1/G.k , G, norm=colors.LogNorm(vmin=G.min()*1e6, vmax=G.max()), cmap='PuBu')#, vmin = clev[0], vmax = clev[-1])
    #
    plot_wavenumber_spectrogram(ax0, Gplot,  clev, title =k,  plot_photon_density=True )
    plt.xlim(xlims)
    #
    if pflag:
        plt.ylabel('Wave length\n(meters)')
        plt.legend()

pos, k, pflag = gs[2, 0], 'Power(weighted mean) \n10 $\log_{10}( (m/m)^2 m )$', True
ax0 = F.fig.add_subplot(pos)
Gplot = G_gFT_wmean.squeeze().rolling(k=5, min_periods= 1, center=True).median().rolling(x=3, min_periods= 1, center=True).median()

dd = 10 * np.log10(Gplot)
dd= dd.where(~np.isinf(dd), np.nan )
clev_log = M.clevels( [dd.quantile(0.01).data, dd.quantile(0.98).data * 1.2], 31)* 1
plot_wavenumber_spectrogram(ax0, dd, clev_log  , title =k, plot_photon_density= True)
plt.xlim(xlims)

# plt.plot(Gplot.x/1e3, 10* nan_list +20 , c='black', label='NAN-density' )
# ax0.axhline(30, color='black', linewidth=0.5)

ax0.axhline(1/k_max_range[0], color='red', linestyle= '--', linewidth= 0.5)
ax0.axhline(1/k_max_range[1], color='red', linestyle= '-', linewidth= 0.5)
ax0.axhline(1/k_max_range[2], color='red', linestyle= '--', linewidth= 0.5)

if pflag:
    plt.ylabel('Wave length\n(meters)')
    plt.legend()

pos = gs[2, 1]
ax0 = F.fig.add_subplot(pos)
plt.title('Photons density ($m^{-1}$)', loc='left')

for k,I in G_gFT.items():
    plt.plot(Gplot.x/1e3, I.N_photons/Lmeters, label=k, linewidth=0.8)
plt.plot(Gplot.x/1e3, G_gFT_wmean.N_photons/3/Lmeters , c='black', label='ave Photons' , linewidth=0.8)
plt.xlim(xlims)
plt.xlabel('Distance from the Ice Edge (km)')

pos = gs[2, 2]

ax0 = F.fig.add_subplot(pos)
ax0.set_yscale('log')

plt.title('Peak Spectal Power', loc='left')

for k,I in G_gFT_sel.items():
    plt.scatter(I.x.data/1e3, I.sel(k=slice(k_max_range[0], k_max_range[2])).integrate('k').data *1e3 ,  s=0.5, marker='.', color='red', alpha= 0.3)

for k,I in G_rar_fft.items():
    I= I.squeeze()
    I= I[:,I.N_per_stancil >=  I.N_per_stancil.max().data*0.9]
    plt.scatter(I.x/1e3, I.sel(k=slice(k_max_range[0], k_max_range[2])).integrate('k') *1e3 ,  s=0.5, marker='.', c='blue', alpha= 0.3)


Gplot= G_fft_wmean.squeeze()
Gplot = Gplot[:,Gplot.N_per_stancil >=  Gplot.N_per_stancil.max().data*0.9]
plt.plot(Gplot.x/1e3, Gplot.sel(k=slice(k_max_range[0], k_max_range[2])).integrate('k') *1e3, '.', markersize=1.5 , c='blue', label= 'FFT')

Gplot= G_gFT_wmean.squeeze()
plt.plot(Gplot.x/1e3, Gplot.sel(k=slice(k_max_range[0], k_max_range[2])).integrate('k') *1e3 , '.' , markersize=1.5, c='red', label= 'gFT')

plt.ylabel('1e-3 $(m)^2~m$')
plt.legend()
plt.ylim(Gplot.min()*1.4, Gplot.max()*1.4 )
plt.xlim(xlims)

F.save_light(path=plot_path, name = 'B02_specs_' + track_name +'_L'+str(Lmeters))

# %% save fitting parameters
MT.save_pandas_table(Pars_optm, save_name+'_params', save_path )

# %% repack data
def repack_attributes(DD):
    #DD = G_LS
    attr_dim_list = list(DD.keys())
    for k in attr_dim_list:
        for ka in list(DD[k].attrs.keys()):
            I = DD[k]
            I.coords[ka] = ( 'beam', np.expand_dims(I.attrs[ka], 0) )
    return DD

G_gFT[G_gFT_wmean.beam.data[0]]     =G_gFT_wmean
G_rar_fft[G_fft_wmean.beam.data[0]] =G_fft_wmean

G_gFT        = repack_attributes(G_gFT)
G_gFT_x      = repack_attributes(G_gFT_x)
G_rar_fft    = repack_attributes(G_rar_fft)

# %% save results
# G_gFT_DS         = xr.merge(G_gFT.values())
# G_gFT_DS['Z_hat_imag'] = G_gFT_DS.Z_hat.imag
# G_gFT_DS['Z_hat_real'] = G_gFT_DS.Z_hat.real
# G_gFT_DS                     = G_gFT_DS.drop('Z_hat')
# G_gFT_DS.attrs['name'] = 'gFT_estimates'
# G_gFT_DS.to_netcdf(save_path+save_name+'_gFT_k.nc')

# G_gFT_x_DS         = xr.merge(G_gFT_x.values())
# G_gFT_x_DS.attrs['name'] = 'gFT_estimates_real_space'
# G_gFT_x_DS.to_netcdf(save_path+save_name+'_gFT_x.nc')

# G_fft_DS        = xr.merge(G_rar_fft.values())
# G_fft_DS.attrs['name']= 'FFT_power_spectra'
# G_fft_DS.to_netcdf(save_path+save_name+'_FFT.nc')
