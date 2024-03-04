
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

load_path   = mconfig['paths']['work'] +'/B02_spectra_'+hemis+'/'
Gd      = xr.load_dataset(load_path+ '/B02_'+track_name + '_gFT_k.nc' )  #
Gdi = Gd.sel(beam = 'gt1r').isel(x = 1).sel(k =slice(0, 0.06))
Gdi.gFT_PSD_data.plot()

# %%

# def wavemodel(XX, YY, ks, ls, amps, group_phase = 0, amp_height= 1):
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
Gd.keys()
for b in Gd.beam:
    B = Gd.sel(beam= b)
    plt.plot(B['lon'], B['lat'])


B1 = Gd.sel(beam =  'gt2r')
B2 = Gd.sel(beam =  'gt1r')

dist_y_mean = np.sqrt( (B1['x_coord']-B2['x_coord'])**2 + (B1['y_coord']-B2['y_coord'])**2 ).mean()
dist_y_mean

B1.coords['dist_y'] = 0
B2.coords['dist_y'] = dist_y_mean

GG0 = xr.concat( [B1.expand_dims('dist_y'), B2.expand_dims('dist_y')] , dim= 'dist_y')


#plt.plot( GG0['x']/1e3 , GG0['y']/1e3)
plt.plot( GG0['x_coord']/1e3 , GG0['y_coord']/1e3, '^')

# %%
GG0.gFT_PSD_model.isel(dist_y= 1).rolling(x=2, k =30).mean().plot()
GG0.Z_hat_imag.isel(dist_y= 1).rolling(x=2, k =30).mean().plot()
GG0.Z_hat_real.isel(dist_y= 1).rolling(x=2, k =30).mean().plot()




k_smooth = 5
some_angle = np.arctan2( GG0.Z_hat_imag.isel(dist_y= 1).rolling(k =k_smooth).mean() , GG0.Z_hat_real.isel(dist_y= 1 ).rolling(k =k_smooth).mean() )
some_angle.plot(cmap =col.circle_big)

plt.plot(some_angle.k, some_angle.sel(x = slice(2.025e6, 2.075e6) ), c='gray', alpha = 0.3)




# %%
#GG0.sel(x = 2e6, method= 'nearest').isel( dist_y= 0, k = slice(100, 300 )).Z_hat_imag.plot()
GG0.sel(x = 2e6 , method= 'nearest').sel(k = slice(0.02, 0.05 )).isel( dist_y= 0).Z_hat_imag.plot()
GG0.sel(x = 2e6,  method= 'nearest').sel(k = slice(0.02, 0.05 )).isel( dist_y= 1).Z_hat_imag.plot()

# %%
k_smooth= 5
GG0.sel(x = 2.05e6 , method= 'nearest').rolling(k =k_smooth).mean().sel(k = slice(0.02, 0.06 )).isel( dist_y= 0).gFT_PSD_model.plot()
GG0.sel(x = 2.05e6,  method= 'nearest').rolling(k =k_smooth).mean().sel(k = slice(0.02, 0.06 )).isel( dist_y= 1).gFT_PSD_model.plot()

# %%

klim= 0, 0.06
Gi = GG0.sel(x = 2.05e6 , method= 'nearest').isel( dist_y= 0).sel(k = slice(klim[0] , klim[1]))#.rolling(k=10 ).mean()
Gi2 = GG0.sel(x = 2.05e6 , method= 'nearest').isel( dist_y= 0).sel(k = slice(klim[0] , klim[1]))#.rolling(k=5 ).mean()

#Gi.gFT_PSD_model.plot()

def plot_Z(X, Y, cmap= plt.cm.jet, line_c= 'k', k_wave=None,  shift=0,  head = 'o', head_size=30, alpha = 1 , **kargs):

    all_cols = np.array(plt.cm.jet(np.arange(0, Y.size )))

    def plot_q(xi, yi, c, kk = None, **kargs):
        #print(xi, yi)
        #xi, yi =np.sign(xi) * np.log(abs(xi)), np.sign(yi) *np.log(abs(yi))
        xi, yi =np.sign(xi) * np.sqrt(abs(xi)), np.sign(yi) *np.sqrt(abs(yi))

        #plt.plot([shift, xi+shift], [0, yi], '-', color = c, alpha = 0.5* alpha, **kargs)
        plt.plot([shift, xi+shift], [0, yi], '-', color =line_c, linewidth =1.5, alpha = 0.5* alpha, **kargs, zorder= 0)
        plt.scatter([xi+shift], [yi], head_size, marker= head  ,color = c, alpha = alpha, **kargs, edgecolors= 'black')
        if kk is not None:
            plt.text(xi+shift, yi,  '  '+str(int(kk)) )


    if k_wave is None:
        for xi,yi,cc in zip(X, Y, all_cols):
            if ~np.isnan(xi):
                plot_q(xi, yi, cc[0:3], **kargs)

    else:
        for xi,yi,cc,kk in zip(X, Y, all_cols, k_wave):
            if ~np.isnan(xi):
                plot_q(xi, yi, cc[0:3],kk= kk, **kargs)

weight = np.copy(Gi.gFT_PSD_model )
#weight = np.copy(Gi.gFT_PSD_model.sel(k = slice(klim[0] , klim[1])) )
weight[weight < 0.015] =np.nan
weight =1e3* weight/np.nanmean(weight)

# col.colormaps2( Gi.k.size )
#all_cols = np.array(col.circle_medium_triple(np.arange(0, Gi.k.size  )))

plot_Z(Gi.gFT_cos_coeff.data * weight,Gi.gFT_sin_coeff.data *weight, k_wave = np.arange(Gi.k.size), line_c  =  'k', head='o', head_size=60, shift= 0)

#Gi2.gFT_PSD_model.rolling(k=3 ).mean().plot()
# Gi2.gFT_PSD_model.plot()


weight = np.copy(Gi2.gFT_PSD_model )
weight[weight < 0.015] =np.nan
weight = 1e3 *weight/np.nanmean(weight)

plot_Z(Gi2.gFT_cos_coeff.data * weight,Gi2.gFT_sin_coeff.data *weight , k_wave = np.arange(Gi2.k.size) , shift= 0, line_c  =  'r', head='+', head_size=120 , alpha = 0.99)


plt.grid()
plt.axis('equal')

#llim= 8
#plt.xlim(- llim, llim)
#plt.ylim(- llim, llim)

#GG0.sel(x = 2.05e6 , method= 'nearest').gFT_cos_coeff
