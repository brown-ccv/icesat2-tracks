import logging
import os
import pathlib
import string

import matplotlib.pyplot as plt

from icesat2waves.local_modules import m_colormanager_ph3 as M_color
from icesat2waves.local_modules import m_tools_ph3 as MT

_logger = logging.getLogger(__name__)

## Read folders and configuration paths
config_dir_path = os.path.dirname(__file__)
mconfig = MT.json_load("config", config_dir_path)

## check folders exist. Create if dont.
for folder_name, folder_path in mconfig["paths"].items():
    full_path = os.path.abspath(folder_path)
    pathlib.Path(full_path).mkdir(exist_ok=True)

# add config path
mconfig["paths"].update({"config": config_dir_path})

# load colorscheme
color_schemes = M_color.color(path=mconfig["paths"]["config"], name="color_def")

lstrings = iter([i + ") " for i in list(string.ascii_lowercase)])
# define journal fig sizes
fig_sizes = mconfig["fig_sizes"]["AMS"]

SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 12

legend_properties = {"weight": "bold"}

plt.rc(
    "font", size=SMALL_SIZE, serif="Helvetica Neue", weight="normal"
)  # controls default text sizes
plt.rc("text", usetex="false")
plt.rc(
    "axes", titlesize=MEDIUM_SIZE, labelweight="normal"
)  # fontsize of the axes title
plt.rc(
    "axes", labelsize=SMALL_SIZE, labelweight="normal"
)  # fontsize of the x and y labels
plt.rc("xtick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("ytick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("legend", fontsize=SMALL_SIZE, frameon=False)  # legend fontsize
plt.rc(
    "figure", titlesize=MEDIUM_SIZE, titleweight="bold", autolayout=True
)  # fontsize of the figure title
plt.rc("path", simplify=True)
plt.rcParams["figure.figsize"] = (10, 8)
plt.rcParams["pcolor.shading"] = "auto"
plt.rc("pcolor", shading="auto")
plt.rc("xtick.major", size=4, width=1)
plt.rc("ytick.major", size=3.8, width=1)
plt.rc("axes", labelsize=MEDIUM_SIZE, labelweight="normal")
plt.rc("axes.spines", top=False, right=False)


def font_for_print(SMALL_SIZE=6, MEDIUM_SIZE=8):
    plt.rc(
        "font", size=SMALL_SIZE, serif="Helvetica Neue", weight="normal"
    )  # controls default text sizes
    plt.rc("text", usetex="false")
    plt.rc(
        "axes", titlesize=MEDIUM_SIZE, labelweight="normal"
    )  # fontsize of the axes title
    plt.rc(
        "axes", labelsize=SMALL_SIZE, labelweight="normal"
    )  # , family='bold')    # fontsize of the x and y labels
    plt.rc("xtick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc("ytick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc("legend", fontsize=SMALL_SIZE, frameon=False)  # legend fontsize
    plt.rc(
        "figure", titlesize=MEDIUM_SIZE, titleweight="bold", autolayout=True
    )  # , family='bold')  # fontsize of the figure title
    plt.rc("axes", labelsize=SMALL_SIZE, labelweight="normal")


def font_for_pres(SMALL_SIZE=10, MEDIUM_SIZE=12):
    plt.rc(
        "font", size=SMALL_SIZE, serif="Helvetica Neue", weight="normal"
    )  # controls default text sizes
    plt.rc("text", usetex="false")
    plt.rc(
        "axes", titlesize=MEDIUM_SIZE, labelweight="normal"
    )  # fontsize of the axes title
    plt.rc(
        "axes", labelsize=SMALL_SIZE, labelweight="normal"
    )  # , family='bold')    # fontsize of the x and y labels
    plt.rc("xtick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc("ytick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc("legend", fontsize=SMALL_SIZE, frameon=False)  # legend fontsize
    plt.rc(
        "figure", titlesize=MEDIUM_SIZE, titleweight="bold", autolayout=True
    )  # fontsize of the figure title
    plt.rc("axes", labelsize=SMALL_SIZE, labelweight="normal")
