import os
import pathlib
###
#   THIS FILE IS A LOCAL FILE
#   it is not maintained via git, it contains configs specific to the machine
###

#os.environ["DISPLAY"] = "localhost:10.0"
# 14, 16,  work
#standart libraries:
import numpy as np
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.colors as colors
import pandas as pd
from icesat2_tracks.local_modules import m_colormanager_ph3 as M_color
from icesat2_tracks.local_modules import m_tools_ph3 as MT
from icesat2_tracks.local_modules import m_general_ph3 as M

import string

import xarray as xr

## Read folders and configuration paths
config_dir_path = os.path.dirname(__file__)
mconfig=MT.json_load('config',config_dir_path)

## check folders exist. Create if dont.
for folder_name, folder_path in mconfig["paths"].items():
    full_path = os.path.abspath(folder_path)
    pathlib.Path(full_path).mkdir(exist_ok=True)

# add config path
mconfig["paths"].update({"config": config_dir_path})

#load colorscheme
color_schemes=M_color.color(path=mconfig['paths']['config'], name='color_def')


lstrings =iter([i+') ' for i in list(string.ascii_lowercase)])
# define journal fig sizes
fig_sizes = mconfig['fig_sizes']['AMS']


SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 12
#csfont = {'fontname':'Comic Sans MS'}
legend_properties = {'weight':'bold'}
#font.family: sans-serif
#font.sans-serif: Helvetica Neue

#import matplotlib.font_manager as font_manager
#font_dirs = ['/home/mhell/HelveticaNeue/', ]
#font_files = font_manager.findSystemFonts(fontpaths=font_dirs)
#font_list = font_manager.createFontList(font_files)
#font_manager.fontManager.ttflist.extend(font_list)

plt.rc('font', size=SMALL_SIZE, serif='Helvetica Neue', weight='normal')          # controls default text sizes
#plt.rc('font', size=SMALL_SIZE, serif='DejaVu Sans', weight='light')
plt.rc('text', usetex='false')
plt.rc('axes', titlesize=MEDIUM_SIZE, labelweight='normal')     # fontsize of the axes title
plt.rc('axes', labelsize=SMALL_SIZE, labelweight='normal') #, family='bold')    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE, frameon=False)    # legend fontsize
plt.rc('figure', titlesize=MEDIUM_SIZE, titleweight='bold', autolayout=True) #, family='bold')  # fontsize of the figure title

#figure.autolayout : False
#matplotlib.rcParams['pdf.fonttype'] = 42
#matplotlib.rcParams['ps.fonttype'] = 42


plt.rc('path', simplify=True)

plt.rcParams['figure.figsize'] = (10, 8)#(20.0, 10.0) #inline
plt.rcParams['pcolor.shading'] = 'auto'
#rcParams['pcolor.shading'] = 'auto'
plt.rc('pcolor', shading = 'auto')

### TICKS
# see http://matplotlib.org/api/axis_api.html#matplotlib.axis.Tick
#xtick.top            : False   # draw ticks on the top side
#xtick.bottom         : True   # draw ticks on the bottom side
#xtick.major.size     : 3.5      # major tick size in points
#xtick.minor.size     : 2      # minor tick size in points
#xtick.major.width    : .8    # major tick width in points
#xtick.minor.width    : 0.6    # minor tick width in points
#xtick.major.pad      : 3.5      # distance to major tick label in points
#xtick.minor.pad      : 3.4      # distance to the minor tick label in points
#xtick.color          : k      # color of the tick labels
#xtick.labelsize      : medium # fontsize of the tick labels
#xtick.direction      : out    # direction: in, out, or inout
#xtick.minor.visible  : False  # visibility of minor ticks on x-axis
#xtick.major.top      : True   # draw x axis top major ticks
#xtick.major.bottom   : True   # draw x axis bottom major ticks
#xtick.minor.top      : True   # draw x axis top minor ticks
#xtick.minor.bottom   : True   # draw x axis bottom minor ticks

#ytick.left           : True   # draw ticks on the left side
#ytick.right          : False  # draw ticks on the right side
#ytick.major.size     : 3.5      # major tick size in points
#ytick.minor.size     : 2      # minor tick size in points
#ytick.major.width    : 0.8    # major tick width in points
#ytick.minor.width    : 0.6    # minor tick width in points
#ytick.major.pad      : 3.5      # distance to major tick label in points
#ytick.minor.pad      : 3.4      # distance to the minor tick label in points
#ytick.color          : k      # color of the tick labels
#ytick.labelsize      : medium # fontsize of the tick labels
#ytick.direction      : out    # direction: in, out, or inout
#ytick.minor.visible  : False  # visibility of minor ticks on y-axis
#ytick.major.left     : True   # draw y axis left major ticks
#ytick.major.right    : True   # draw y axis right major ticks
#ytick.minor.left     : True   # draw y axis left minor ticks
#ytick.minor.right    : True   # draw y axis right minor ticks


plt.rc('xtick.major', size= 4, width=1 )
plt.rc('ytick.major', size= 3.8, width=1 )

#axes.facecolor      : white   # axes background color
#axes.edgecolor      : black   # axes edge color
#axes.linewidth      : 0.8     # edge linewidth
#axes.grid           : False   # display grid or not
#axes.titlesize      : large   # fontsize of the axes title
#axes.titlepad       : 6.0     # pad between axes and title in points
#axes.labelsize      : medium  # fontsize of the x any y labels
#axes.labelpad       : 4.0     # space between label and axis
#axes.labelweight    : normal  # weight of the x and y labels
#axes.labelcolor     : black

plt.rc('axes', labelsize= MEDIUM_SIZE, labelweight='normal')




# axes.spines.left   : True   # display axis spines
# axes.spines.bottom : True
# axes.spines.top    : True
# axes.spines.right  : True
plt.rc('axes.spines', top= False, right=False )



def font_for_print():

    SMALL_SIZE = 6
    MEDIUM_SIZE = 8
    BIGGER_SIZE = 10
    #csfont = {'fontname':'Comic Sans MS'}
    legend_properties = {'weight':'bold'}
    #font.family: sans-serif
    #font.sans-serif: Helvetica Neue

    #import matplotlib.font_manager as font_manager
    #font_dirs = ['/home/mhell/HelveticaNeue/', ]
    #font_files = font_manager.findSystemFonts(fontpaths=font_dirs)
    #font_list = font_manager.createFontList(font_files)
    #font_manager.fontManager.ttflist.extend(font_list)

    plt.rc('font', size=SMALL_SIZE, serif='Helvetica Neue', weight='normal')          # controls default text sizes
    #plt.rc('font', size=SMALL_SIZE, serif='DejaVu Sans', weight='light')
    plt.rc('text', usetex='false')
    plt.rc('axes', titlesize=MEDIUM_SIZE, labelweight='normal')     # fontsize of the axes title
    plt.rc('axes', labelsize=SMALL_SIZE, labelweight='normal') #, family='bold')    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE, frameon=False)    # legend fontsize
    plt.rc('figure', titlesize=MEDIUM_SIZE, titleweight='bold', autolayout=True) #, family='bold')  # fontsize of the figure title

    #figure.autolayout : False
    #matplotlib.rcParams['pdf.fonttype'] = 42
    #matplotlib.rcParams['ps.fonttype'] = 42


    #plt.rc('xtick.major', size= 4, width=1 )
    #plt.rc('ytick.major', size= 3.8, width=1 )


    plt.rc('axes', labelsize= SMALL_SIZE, labelweight='normal')

def font_for_pres():

    SMALL_SIZE = 10
    MEDIUM_SIZE = 12
    BIGGER_SIZE = 14
    #csfont = {'fontname':'Comic Sans MS'}
    legend_properties = {'weight':'bold'}
    #font.family: sans-serif
    #font.sans-serif: Helvetica Neue

    #import matplotlib.font_manager as font_manager
    #font_dirs = ['/home/mhell/HelveticaNeue/', ]
    #font_files = font_manager.findSystemFonts(fontpaths=font_dirs)
    #font_list = font_manager.createFontList(font_files)
    #font_manager.fontManager.ttflist.extend(font_list)

    plt.rc('font', size=SMALL_SIZE, serif='Helvetica Neue', weight='normal')          # controls default text sizes
    #plt.rc('font', size=SMALL_SIZE, serif='DejaVu Sans', weight='light')
    plt.rc('text', usetex='false')
    plt.rc('axes', titlesize=MEDIUM_SIZE, labelweight='normal')     # fontsize of the axes title
    plt.rc('axes', labelsize=SMALL_SIZE, labelweight='normal') #, family='bold')    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE, frameon=False)    # legend fontsize
    plt.rc('figure', titlesize=MEDIUM_SIZE, titleweight='bold', autolayout=True) #, family='bold')  # fontsize of the figure title

    #figure.autolayout : False
    #matplotlib.rcParams['pdf.fonttype'] = 42
    #matplotlib.rcParams['ps.fonttype'] = 42


    #plt.rc('xtick.major', size= 4, width=1 )
    #plt.rc('ytick.major', size= 3.8, width=1 )


    plt.rc('axes', labelsize= SMALL_SIZE, labelweight='normal')



# add project depenent libraries
#sys.path.append(config['paths']['local_script'])


