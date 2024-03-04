import os, sys
#execfile(os.environ['PYTHONSTARTUP'])

"""
This file download all L3 data from
https://resources.marine.copernicus.eu/product-detail/WAVE_GLO_WAV_L3_SPC_NRT_OBSERVATIONS_014_002/INFORMATION
"""

exec(open(os.environ['PYTHONSTARTUP']).read())
exec(open(STARTUP_2021_IceSAT2).read())

#%matplotlib inline

import imp
import subprocess

save_path   = mconfig['paths']['work'] + '/CMEMS_WAVE_GLO_L3/'

# print(time_range)
# # create timestamp according to fiels on ftp server:
# time_stamps_ftp = np.arange(time_range[0].astype('datetime64[3h]'), time_range[1].astype('datetime64[3h]') +  np.timedelta64(3, 'h'), np.timedelta64(3, 'h'))
# time_stamps_ftp_str = [str(t).replace('-', '') for t in time_stamps_ftp]
# #plt.plot(G1['lons'], G1['lats'], '.' )


username='mhell'
pw='BkUexT#72'

# NRT product
# paths look like this:
#ftp://nrt.cmems-du.eu/Core/WAVE_GLO_WAV_L3_SPC_NRT_OBSERVATIONS_014_002/dataset-wav-sar-l3-spc-nrt-global-s1a/2020/03/dataset-wav-sar-l3-spc-nrt-global-s1a_20200302T000000Z_20200302T030000Z_P20200323T0646Z-3H-rep.nc

subpath = 'WAVE_GLO_WAV_L3_SPC_NRT_OBSERVATIONS_014_002'
path='ftp://nrt.cmems-du.eu/Core/'+subpath+ '/' #dataset-wav-sar-l3-spc-rep-global-'+sat+'/'+y+'/'+m+'/
file_card = 'dataset-wav-sar-l3-spc-nrt-global-*-3H-rep.nc'

# REP product
# paths look like this:
#ftp://my.cmems-du.eu/Core/WAVE_GLO_PHY_SPC_L3_MY_014_006/dataset-wav-sar-l3-spc-rep-global-s1a/2019/07/dataset-wav-sar-l3-spc-rep-global-s1a_20190702T120000Z_20190702T150000Z_P20210619T1207Z-3H-rep.nc

# subpath = 'WAVE_GLO_WAV_L3_SPC_REP_OBSERVATIONS_014_002'
# path='ftp://my.cmems-du.eu/Core/'+subpath+ '/'
# file_card = 'dataset-wav-sar-l3-spc-rep-global-*-3H-rep.nc'

wget_str= ['wget',  '-r', path ,'--ftp-user='+username ,'--ftp-password='+pw ,'--no-parent', '-A' , file_card ,'-nd', '-c']
wget_str.append('-P')
wget_str.append(save_path +'/'+ subpath)


print(wget_str)
print('save to ' + save_path +'/'+ subpath)

list_files = subprocess.run(' '.join(wget_str), shell=True,stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

#list_files.stdout
#print(list_files.stderr)
flist_parameters= list()
#list_files.check_returncode()
print('download sugcess:', list_files.returncode == 0)
