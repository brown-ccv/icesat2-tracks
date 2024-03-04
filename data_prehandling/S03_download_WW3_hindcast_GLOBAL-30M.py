import os, sys
#execfile(os.environ['PYTHONSTARTUP'])

"""
This file download all L3 data from
#ftp://ftp.ifremer.fr/ifremer/ww3/HINDCAST/GLOBMULTI_ERA5_GLOBCUR_01/GLOB-30M/2019/FIELD_NC/
"""

exec(open(os.environ['PYTHONSTARTUP']).read())
exec(open(STARTUP_2021_IceSAT2).read())

#%matplotlib inline

import imp
import subprocess
import glob
import time

save_path   = mconfig['paths']['scratch']# + 'GLOBMULTI_ERA5_GLOBCUR_01/'
save_path2   = mconfig['paths']['work']
#ftp://ftp.ifremer.fr/ifremer/ww3/HINDCAST/GLOBMULTI_ERA5_GLOBCUR_01/GLOB-30M/2019/FIELD_NC/LOPS_WW3-GLOB-30M_201905.nc

lat_lim= 50 # deg north or South
var_list = [ 'dir', 'dp','fp', 'hs', 'ice', 'lm', 'spr', 't01', 't02', 't0m1', 'tws',
'pdp0',  'pdp1',  'pdp2',  'pdp3',  'pdp4',  'pdp5',
'pspr0',  'pspr1',  'pspr2',  'pspr3',  'pspr4',  'pspr5',
'ptp0',  'ptp1',  'ptp2',  'ptp3',  'ptp4',  'ptp5',
'phs0',  'phs1',  'phs2',  'phs3',  'phs4',  'phs5']


flist_parameters= list()
subpath = 'GLOBMULTI_ERA5_GLOBCUR_01/'
try:
    os.mkdir(save_path2 + '/'+ subpath)
except:
    pass

#year = 2018
for year in np.arange(2018, 2022):

    year_str= str(year)
    print('-----' + year_str)
    path='ftp://ftp.ifremer.fr/ifremer/ww3/HINDCAST/'+subpath+'/GLOB-30M/'+ year_str +'/FIELD_NC/' #dataset-wav-sar-l3-spc-rep-global-'+sat+'/'+y+'/'+m+'/
    file_card = 'LOPS_WW3-GLOB-30M_'+year_str+'*.nc'

    wget_str= ['wget',  '-r', path ,'--no-parent', '-A' , file_card ,'-nd', '-c']
    wget_str.append('-P')
    wget_str.append(save_path +'/'+ subpath)

    print(' '.join(wget_str))
    print('save to ' + save_path +'/'+ subpath)

    # list_files = subprocess.run(' '.join(wget_str), shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    #
    # print('download sugcess:', list_files.returncode == 0)
    # if list_files.returncode == 0:
    #    flist_parameters.append(list_files)

    year_file_list  =  glob.glob(save_path + subpath+'*' + file_card)
    year_file_list2 = list()
    for f in year_file_list:
        if '_p2l.nc' in f:
            #os.remove(f)
            print(f)
        else:
            year_file_list2.append(f)
    print('open:')
    year_file_list2.sort()
    print(year_file_list2)
    G_all   = xr.open_mfdataset(year_file_list2)

    # NH
    G2              = G_all[var_list].isel(latitude =G_all.latitude > lat_lim )
    mm, datasets    = zip(*G2.groupby("time.month"))
    paths           = [save_path2 + '/'+ subpath +'/LOPS_WW3-GLOB-30M_'+year_str+'_'+str(m).zfill(2)+'_NH_select.nc' for m in mm]
    xr.save_mfdataset(datasets, paths)
    #G2.to_netcdf(save_path2 + '/'+ subpath +'/' + save_name )

    # SH
    G2              = G_all[var_list].isel(latitude =G_all.latitude < -lat_lim )
    mm, datasets    = zip(*G2.groupby("time.month"))
    paths           = [save_path2 + '/'+ subpath +'/LOPS_WW3-GLOB-30M_'+year_str+'_'+str(m).zfill(2)+'_SH_select.nc' for m in mm]
    xr.save_mfdataset(datasets, paths)
    #G2.to_netcdf(save_path2 + '/'+ subpath +'/' + save_name )

    print('merged and save needed variables in work directory')
    #time.sleep(5)
    #os.remove(year_file_list2)


print( flist_parameters )
