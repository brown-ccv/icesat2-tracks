
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

imp.reload(M_color)
col=M_color.color(path=mconfig['paths']['analysis']+'../config/', name='color_def')

xr.set_options(display_style='text')
#import s3fs
# %%
track_name, batch_key, test_flag = io.init_from_input(sys.argv) # loads standard experiment
#track_name, batch_key, test_flag = '20190605061807_10380310_004_01', 'SH_batch01', False
#track_name, batch_key, test_flag = '20190601094826_09790312_004_01', 'SH_batch01', False
#track_name, batch_key, test_flag = '20190207111114_06260210_004_01', 'SH_batch02', False
#track_name, batch_key, test_flag = '20190208152826_06440210_004_01', 'SH_batch01', False
#track_name, batch_key, test_flag = '20190213133330_07190212_004_01', 'SH_batch02', False
#track_name, batch_key, test_flag = '20190207002436_06190212_004_01', 'SH_batch02', False
#track_name, batch_key, test_flag = '20190206022433_06050212_004_01', 'SH_batch02', False


#track_name, batch_key, test_flag = '20190502050734_05180310_004_01', 'SH_batch02', False

# local track
#track_name, batch_key, test_flag = '20190219073735_08070210_004_01', 'SH_batch02', False

track_name, batch_key, test_flag = 'SH_20190219_08070210', 'SH_publish', True

x_pos= 3

#print(track_name, batch_key, test_flag)
hemis, batch = batch_key.split('_')

load_path   = mconfig['paths']['work'] +batch_key +'/B02_spectra/'
load_file   = load_path + 'B02_' + track_name #+ '.nc'
#plot_path   = mconfig['paths']['plot'] + '/'+hemis+'/'+batch_key+'/' + track_name + '/'
plot_path   = mconfig['paths']['plot'] + '/'+hemis+'/'+batch_key+'/publish/' + track_name + '/'
MT.mkdirs_r(plot_path)

Gk = xr.open_dataset(load_file+'_gFT_k.nc')
Gx = xr.open_dataset(load_file+'_gFT_x.nc')

Gfft = xr.open_dataset(load_file+'_FFT.nc')


# %%
all_beams   = mconfig['beams']['all_beams']
high_beams  = mconfig['beams']['high_beams']
low_beams   = mconfig['beams']['low_beams']
#Gfilt   = io.load_pandas_table_dict(track_name + '_B01_regridded', load_path) # rhis is the rar photon data
#Gd      = io.load_pandas_table_dict(track_name + '_B01_binned' , load_path)  #

col.colormaps2(21)

# %% check paths (again)
# G_gFT_wmean = (Gk['gFT_PSD_model'].where( ~np.isnan(Gk['gFT_PSD_model']), 0) * Gk['N_per_stancil']).sum('beam')/ Gk['N_per_stancil'].sum('beam')
# G_gFT_wmean['N_per_stancil'] = Gk['N_per_stancil'].sum('beam')

# G_fft_wmean = (Gfft.where( ~np.isnan(Gfft), 0) * Gfft['N_per_stancil']).sum('beam')/ Gfft['N_per_stancil'].sum('beam')
# G_fft_wmean['N_per_stancil'] = Gfft['N_per_stancil'].sum('beam')


# %%  define simple routines
def add_info(D, Dk, ylims):
    eta  = D.eta + D.x
    N_per_stancil, ksize = Dk.N_per_stancil.data , Dk.k.size
    plt.text(eta[0].data, ylims[-1], '  N='+numtostr(N_per_stancil)  + ' N/2M= '+ fltostr(N_per_stancil/2/ksize, 1) )


# %% Single views

def plot_data_eta(D,  offset = 0  ,  **kargs ):
    eta_1  = D.eta# + D.x
    y_data = D.y_model +offset
    plt.plot(eta_1,y_data , **kargs)
    return eta_1

def plot_model_eta(D, ax,  offset = 0,  **kargs ):
    eta  = D.eta #+ D.x
    y_data = D.y_model+offset
    plt.plot(eta ,y_data , **kargs)

    # ax.axvline(eta[0].data, linewidth=0.1,  color=kargs['color'], alpha=0.5)
    # ax.axvline(eta[-1].data, linewidth=0.1,  color=kargs['color'], alpha=0.5)

if ('y_data' in Gx.sel(beam = 'gt3r').keys()):
    print('ydata is ', ('y_data' in Gx.sel(beam = 'gt3r').keys()) )
else:
    print('ydata is ', ('y_data' in Gx.sel(beam = 'gt3r').keys()) )
    MT.json_save('B03_fail', plot_path, {'reason':'no y_data'})
    print('failed, exit')
    exit()



# %%

# derive spectral errors:
Lpoints=  Gk.Lpoints.mean('beam').data
N_per_stancil = Gk.N_per_stancil.mean('beam').data#[0:-2]

G_error_model =dict()
G_error_data =dict()

for bb in Gk.beam.data:
    I = Gk.sel(beam= bb)
    b_bat_error =  np.concatenate([ I.model_error_k_cos.data , I.model_error_k_sin.data ])
    Z_error = gFT.complex_represenation(b_bat_error, Gk.k.size, Lpoints)
    PSD_error_data, PSD_error_model = gFT.Z_to_power_gFT(Z_error, np.diff(Gk.k)[0],N_per_stancil  , Lpoints )

    #np.expand_dims(PSD_error_model, axis =)
    G_error_model[bb] =  xr.DataArray(data = PSD_error_model, coords = I.drop('N_per_stancil').coords, name='gFT_PSD_model_error' ).expand_dims('beam')
    G_error_data[bb] =  xr.DataArray(data = PSD_error_data, coords = I.drop('N_per_stancil').coords, name='gFT_PSD_data_error' ).expand_dims('beam')


Gk['gFT_PSD_model_err'] = xr.concat(G_error_model.values(), dim='beam')
Gk['gFT_PSD_data_err']  = xr.concat(G_error_data.values(), dim='beam')



# %%
fltostr = MT.float_to_str
numtostr = MT.num_to_str

font_for_print()


#for i in x_pos_sel[::2]:
#i =x_pos_sel[20]
MT.mkdirs_r(plot_path+'B03_spectra/')

x_pos_sel =  np.arange(Gk.x.size)[~np.isnan(Gk.mean('beam').mean('k').gFT_PSD_data.data)]
x_pos_max = Gk.mean('beam').mean('k').gFT_PSD_data[~np.isnan(Gk.mean('beam').mean('k').gFT_PSD_data)].argmax().data
xpp = x_pos_sel[ [int(i) for i in np.round(np.linspace(0, x_pos_sel.size-1, 10))]]
xpp = np.insert(xpp, 0, x_pos_max)

key         = 'gt2r'
#lead_color= col.rels[key]
lead_color= col.rels['group2']

x_pos =2
i =xpp[x_pos]
#for i in xpp:

#i = xpp[0]
font_for_print()
xscale=1e3
F = M.figure_axis_xy(fig_sizes['two_column_square'][0], fig_sizes['two_column_square'][1], view_scale=0.8, container =True)

#plt.suptitle('gFT Model and Spectrograms | x='+str(Gk.x[i].data)+' \n' + track_name, y = 0.95)
gs = GridSpec(14,6,  wspace=0.1,  hspace=5)#figure=fig,


col_d = col.__dict__['rels']
beam_group = ['gt2l', 'gt2r']
for k, gss in zip(beam_group, [gs[0:3, :], gs[2:5, :]] ):


    ax0 = F.fig.add_subplot(gss)

    Gx_1 = Gx.isel(x= i).sel(beam = k)
    Gk_1 = Gk.isel(x= i).sel(beam = k)
    x_sel= Gx_1.x

    plot_model_eta(Gx_1, ax0, offset= 0,  linestyle='-', color=col_d[k], linewidth=0.8, alpha=1, zorder=12 )
    ylims= -np.nanstd(Gx_1.y_data)*3, np.nanstd(Gx_1.y_data)*3
    #add_info(Gx_1, Gk_1 , ylims )

    # oringial data
    eta_1= plot_data_eta(Gx_1 , offset= 0.05 , linestyle= '-', c='k',linewidth=1,  alpha =0.5, zorder=11)

    # reconstruct in  gaps
    FT = gFT.generalized_Fourier(Gx_1.eta + Gx_1.x, None,Gk_1.k )
    _ = FT.get_H()
    FT.b_hat=np.concatenate([ Gk_1.gFT_cos_coeff, Gk_1.gFT_sin_coeff ])
    plt.plot(Gx_1.eta, FT.model() ,'-', c='orange', linewidth=0.3, alpha=1,zorder= 2)

    if 'l' in k:
        ax0.spines['left'].set_visible(True)
        ax0.spines['right'].set_visible(False)
        ax0.spines['top'].set_linewidth(0.2)
        ax0.spines['left'].set_color(col.gray)

        ax0.spines['bottom'].set_visible(False)
        ax0.tick_params(labelbottom=False, bottom=False)

    elif 'r' in k:
        ax0.spines['left'].set_visible(False)
        ax0.spines['right'].set_visible(True)
        ax0.yaxis.set_ticks_position('right')
        ax0.yaxis.set_label_position('right')
        ax0.spines['right'].set_color(col.gray)

    ax0.tick_params(axis='both', colors=col.gray)

    ax0.set_facecolor((1.0, 1, 1, 0))

    ax0.axhline(0.05, linewidth=0.5,  color=col.black, alpha=0.5)
    ax0.axhline(0, linewidth=0.5,  color=col_d[k], alpha=1)

    ax0.spines['bottom'].set_linewidth(0.2)
    ax0.spines['bottom'].set_visible(False)
    #ax0.spines['bottom'].set_position(('data', 0))
    #plt.grid()
    dx = eta_1.diff('eta').mean().data
    plt.xlim( eta_1[0].data+0 * dx, eta_1[-1].data+  0 * dx )
    plt.ylabel('relative slope (m/m)')
    plt.ylim(-0.12, 0.12)

#eta_ticks =  np.linspace(Gx_1.eta.data[0], Gx_1.eta.data[-1], 11)
eta_ticks_labels, eta_ticks = MT.tick_formatter(np.arange(-12000, 12000+1500, 4*1500)/1e3, interval= 1, rounder=0, expt_flag= False, shift=0)

ax0.set_xticks(eta_ticks*1e3)
ax0.set_xticklabels(eta_ticks_labels)

plt.xlim( eta_1[0].data - 40 * dx, eta_1[-1].data+  40 * dx )
plt.xlabel('segment distance $\eta$ (km) @ X='+ numtostr(Gx_1.x.data/1e3)+' km')


# make spectral error
# Lpoints=  Gk.Lpoints.sel(beam = beam_group).mean('beam').data
# N_per_stancil = Gk.N_per_stancil.sel(beam = beam_group).mean('beam').data[0:-2]
# b_bat_error =  np.concatenate([Gplot.model_error_k_cos.data, Gplot.model_error_k_sin.data ])
# Z_error = gFT.complex_represenation(b_bat_error, Gplot.k.size, Lpoints)
# PSD_error_data, PSD_error_model = gFT.Z_to_power_gFT(Z_error,np.diff(Gplot.k)[0],N_per_stancil  , Lpoints )
#PSD_error_data.shape
#Gk['PSD_error_data'] = ( ('x', 'k'), PSD_error_data)


# spectra
# define threshold
k_thresh = 0.085
ax1_list = list()
dd_max=list()

for pos, k, lflag in zip([ gs[6:9, 0:3],  gs[9:12, 0:3] ],  beam_group, [True, False] ):

    ax11 = F.fig.add_subplot(pos)
    ax11.tick_params(labelleft=True)
    ax1_list.append(ax11)

    Gx_1 = Gx.isel(x= i).sel(beam = k)
    Gk_1 = Gk.isel(x= i).sel(beam = k)
    Gfft_1 = Gfft.isel(x= i).sel(beam = k)

    klim= Gk_1.k[0], Gk_1.k[-1]

    dd = Gk_1.gFT_PSD_data#.rolling(k=10, min_periods= 1, center=True).mean()
    plt.plot(Gk_1.k,  dd, color=col_d[k], linewidth=.5 ,alpha= 0.5, zorder=5 )

    dd = Gk_1.gFT_PSD_data.rolling(k=10, min_periods= 1, center=True).mean()
    plt.plot(Gk_1.k,  dd, color=col_d[k], linewidth=1, zorder=8 , label='GFT Spec')
    #plt.plot(Gk_1.k,  dd, color=col.gridcolor, linewidth=2.4, zorder=6 )

    # dd_err = Gk_1.gFT_PSD_model_err.rolling(k=10, min_periods= 1, center=True).mean()
    # plt.fill_between(Gk_1.k,  dd- 20*dd_err,  dd+ 20*dd_err, color=col_d[k], linewidth=1, zorder=8 , label='GFT Spec')

    dd_fft = Gfft_1.power_spec.rolling(k=10, min_periods= 1, center=True).mean()
    plt.plot(Gfft_1.k,  dd_fft, color=col.gray, linewidth=0.5, zorder=5, label='FFT Spec' )

    dd_max.append(np.nanmax(dd.data))
    plt.xlim(klim)
    plt.title('Beam ' + k  + ' Single Estimate', loc='left')

    if lflag:
        #plt.title('Energy Spectra', loc ='left')
        ax11.tick_params(labelbottom=False, bottom=True)
    else:
        plt.xlabel('wavenumber k (2$\pi$ m$^{-1}$)')

    plt.ylabel('10$^{-2}$ $(m/m)^2/k$')

    #plt.ylim(dd.min(), max(dd_max) * 1.1)
    # ax11.axvline(k_thresh, linewidth=1,  color='gray', alpha=1)
    # ax11.axvspan(k_thresh , klim[-1],   color='gray', alpha=0.5, zorder=12)

    ax11.spines['left'].set_color(col.gray)
    ax11.spines['bottom'].set_color(col.gray)
    ax11.tick_params(axis='both', colors=col.gray)


    x_ticks_labels, x_ticks = MT.tick_formatter(np.arange(0.02, 0.12, 0.02), interval= 2, rounder=0, expt_flag= False, shift=0)
    ax11.set_xticks(x_ticks)
    ax11.set_xticklabels(x_ticks_labels)

plt.legend()

if ~np.isnan(np.nanmax(dd_max)):
    for ax in ax1_list:
        ax.set_ylim(0, np.nanmax(dd_max) * 1.1)

        y_ticks_labels, y_ticks = MT.tick_formatter(np.arange(0, np.nanmax(dd_max)*1e2, 0.5), interval= 2, rounder=0, expt_flag= False, shift=0)
        ax.set_yticks(y_ticks/1e2)
        ax.set_yticklabels(y_ticks_labels)

# plot k-x data

Gplot= Gk.sel(beam = beam_group, x=Gk.x[0:-2]).mean('beam').rolling(k=10, x=2, min_periods= 1, center=True).median()

# define 2nd part of the axis
ax1 = F.fig.add_subplot(gs[6:9, 3:])
ax2 = F.fig.add_subplot(gs[9:12, 3:])
cbaxes = F.fig.add_subplot(gs[-2:, 3:])


F.fig.add_subplot(ax1)
dd = 10 * np.log10(Gplot.gFT_PSD_data)
dd = dd.where(~np.isinf(dd), np.nan )

clev_log = M.clevels( [dd.quantile(0.01).data * 0.5, dd.quantile(0.98).data * 2.5], 31)* 1

#plot_wavenumber_spectrogram(ax1, dd,  clev_log, title =k + ' unsmoothed',  plot_photon_density=False)


col.colormaps2(31, gamma=1)
#col.plot()
col.white_base_blgror
x_lambda= 2 * np.pi/dd.k
css = plt.pcolormesh(dd.x/1e3, x_lambda , dd, cmap=col.white_base_blgror , vmin = clev_log[0], vmax = clev_log[-1])

#plt.xlabel('Distance from the Ice Edge (km)')
plt.ylim(x_lambda[-1], x_lambda[0])


xlims= Gplot.x[0]/1e3, Gplot.x[-1]/1e3
plt.xlim(xlims)
plt.ylabel('Wave length\n(meters)')
plt.title('Mean Spectrogram', loc='left')



F.fig.add_subplot(ax2)

dd = 10 * np.log10(Gplot.gFT_PSD_data_err)
dd= np.where(~np.isinf(dd),  dd, np.nan )

#clev = M.clevels( [np.percentile(dd, 0.01)* 0.75,np.percentile(dd, 0.98)  * 1], 31)* 1

x_lambda= 2 * np.pi/Gplot.k
plt.pcolormesh(Gplot.x/1e3, x_lambda , dd, cmap=col.white_base_blgror , vmin = clev_log[0], vmax = clev_log[-1])

plt.ylim(x_lambda[-1], x_lambda[0])

xlims= Gplot.x[0]/1e3, Gplot.x[-1]/1e3
plt.xlim(xlims)
plt.ylabel('Wave length\n(meters)')
plt.xlabel('Distance from the Ice Edge (km)')
plt.title('Mean Error', loc='left')

for axx in [ax1, ax2]:
    axx.set_yscale('log')
    axx.spines['left'].set_visible(False)
    axx.spines['right'].set_visible(True)
    axx.yaxis.set_ticks_position('right')
    axx.yaxis.set_label_position('right')
    axx.spines['right'].set_color(col.gray)

    axx.tick_params(axis='both', colors=col.gray)
    axx.set_facecolor((1.0, 1, 1, 0))
    axx.axhline(0, linewidth=0.1,  color=col.gray, alpha=0.8)
    axx.spines['bottom'].set_linewidth(0.2)
    axx.axvline(x_sel/1e3, linewidth= 0.8, color= col_d[k], zorder=12)
    axx.axvline(x_sel/1e3, linewidth= 2, color= col.black, zorder=10)

    x_ticks_labels, x_ticks = MT.tick_formatter(np.arange( (Gplot.x[0]/1e4).round(0)*10 ,(Gplot.x[-1]/1e4).round(0)*10, 20), interval= 2, rounder=0, expt_flag= False, shift=1)
    axx.set_xticks(x_ticks)
    axx.set_xticklabels(x_ticks_labels)




ax1.tick_params(labelbottom=False, bottom=True)

# cbaxes.spines['left'].set_visible(False)

cbaxes.axis('off')
cbpos  = cbaxes.get_position()
cbaxes2 = F.fig.add_axes([cbpos.x0,cbpos.y0+ 1.5*cbpos.height/6,cbpos.width,cbpos.height/6])
cb = plt.colorbar(css , cax =cbaxes2, orientation= 'horizontal')

cb.set_label('Power($(m/m)^2/k$)')
cb.outline.set_visible(False)
cbaxes2.tick_params(axis='both', colors=col.gray)

#plt.gca().spines['top'].set_visible(False)
# cbaxes2.spines['bottom'].set_visible(False)
# cbaxes2.spines['top'].set_visible(False)

# F.save_light(path= plot_path, name='B03_gFT_exmpl_x'+str(i)+'_'+track_name)
# F.save_pup(path= plot_path, name='B03_gFT_exmpl_x'+str(i)+'_'+track_name)
