
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


# %%
l1 = 100
k1= 2* np.pi /l1

l2 = 65
k2= 2* np.pi /l2

x=np.arange(0, 1000, 0.5)
y=np.arange(0, 1000, 0.5)

z = np.sin(k1) + np.sin(k2* x)


# %%

font_for_pres()
plt.semilogx(k, abs(Z) )
ax =plt.gca()
ax.axvline(k1)
ax.axvline(k2)

# %%

F = M.figure_axis_xy(4,4)
#kk = np.array([k1, k2])
kk = np.array([k1, k2])

kk_pos=  np.array([abs(k - ki).argmin() for ki in kk])

Zi = Z[kk_pos]/abs(Z[kk_pos])
plt.plot(Zi.real , Zi.imag, '.k' )

b =  np.arange(-3*l1, l1*6, 2)

p = np.exp( np.outer(b, kk) *1j  )
#p = np.exp( b.*kk *1j  )

Z2= p * Zi
plt.plot( Z2.real ,Z2.imag, '.r' )

plt.axis('equal')
plt.grid()
F.ax.axhline(0, color='black')
F.ax.axvline(0, color='black')

# %%
# np.angle(Zi[0]-Zi[1], deg=True)
# np.angle(Zi[1]-Zi[0], deg=True)
#
# np.angle(Zi[0]-Zi[1], deg=True) - np.angle(Zi[1]-Zi[0], deg=True)

plt.plot( np.angle(Zi) - np.angle(Z2) )
plt.plot( np.angle(Z2) - np.angle(Zi) )
np.angle(Zi[1]) - np.angle(Zi[0])

Zi[0].dot

np.angle(Zi[0], deg=True)
np.angle(Zi[1], deg=True)

#@vectorize
def cdot(a, b):
    return (a.real*b.real + a.imag*b.imag)

plt.plot(np.arccos(cdot(Zi, Z2))**2)

plt.plot( b, (np.arccos(cdot(Zi, Z2))**2).sum(1) )
plt.grid()

# np.arccos( (Zi[0]* Zi[1]).real + (Zi[0]* Zi[1]).imag)
# np.arccos( np.dot(Zi[0], Zi[1]) )
