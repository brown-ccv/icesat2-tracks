from __future__ import unicode_literals
from __future__ import division

from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatch
import os
import logging

_logger = logging.getLogger(__name__)


def ase_to_json(path):
    os.getcwd()
    import swatch

    A = swatch.parse(path)

    for i in A[0]["swatches"]:
        _logger.debug(
            "swatch name: %s, swatch values: %s", i["name"], i["data"]["values"]
        )

    return A


def json_save(name, path, data, return_name=False):
    import json

    if not os.path.exists(path):
        os.makedirs(path)
    full_name_root = os.path.join(path, name)
    full_name = os.path.join(full_name_root + ".json")
    with open(full_name, "w") as outfile:
        json.dump(data, outfile)
    _logger.debug("save at %s:", full_name)
    if return_name:
        return full_name_root
    else:
        return


def json_load(name, path):
    import json

    full_name = os.path.join(path, name + ".json")

    with open(full_name, "r") as ifile:
        data = json.load(ifile)
    _logger.debug("loaded from: %s", full_name)
    return data


class color:
    def __init__(self, path=None, name=None):
        self.white = (1, 1, 1)
        if (path is not None) & (name is not None):
            _logger.debug("color theme: %s", name)
            try:
                theme = json_load(name, path)
                for k, v in theme.items():
                    setattr(self, k, v)
            except:
                _logger.debug("fail load theme, fall back to default theme")
                _logger.debug("path: %s%s", path, "mhell_colortheme17")
                self.default_colors()

        else:
            self.default_colors()

    def alpha(self, color, a):
        return (color[0], color[1], color[2], a)

    def colormaps2(self, n, gamma=None):
        gamma = 1 if gamma is None else gamma

        rels = self.rels
        self.divergingwhite = LinearSegmentedColormap.from_list(
            "my_colormap",
            (rels["plus"], rels["white"], rels["minus"]),
            N=n,
            gamma=gamma,
        )
        self.divergingwhitelarge = LinearSegmentedColormap.from_list(
            "my_colormap",
            (
                rels["darkblue"],
                rels["blue"],
                rels["white"],
                rels["orange"],
                rels["red"],
            ),
            N=n,
            gamma=gamma,
        )

        self.whiteblue = LinearSegmentedColormap.from_list(
            "my_colormap",
            (rels["white"], rels["lightblue"], rels["blue"], rels["darkblue"]),
            N=n,
            gamma=gamma,
        )
        self.greyblue = LinearSegmentedColormap.from_list(
            "my_colormap",
            (rels["gridcolor"], rels["lightblue"], rels["blue"], rels["darkblue"]),
            N=n,
            gamma=gamma,
        )
        self.greyblue_r = LinearSegmentedColormap.from_list(
            "my_colormap",
            (rels["darkblue"], rels["blue"], rels["lightblue"], rels["gridcolor"]),
            N=n,
            gamma=gamma,
        )

        self.greyred = LinearSegmentedColormap.from_list(
            "my_colormap",
            (rels["gridcolor"], rels["rascade1"], rels["rascade2"]),
            N=n,
            gamma=gamma,
        )
        self.greyred_r = LinearSegmentedColormap.from_list(
            "my_colormap",
            (rels["rascade3"], rels["rascade2"], rels["rascade1"], rels["gridcolor"]),
            N=n,
            gamma=gamma,
        )

        self.greyredorange = LinearSegmentedColormap.from_list(
            "my_colormap",
            (rels["gridcolor"], rels["cascade3"], rels["red"], rels["orange"]),
            N=n,
            gamma=gamma,
        )

        self.white_base_blue = LinearSegmentedColormap.from_list(
            "my_colormap",
            (rels["white"], rels["darkblue"], rels["blue"], rels["lightblue"]),
            N=n,
            gamma=gamma,
        )
        self.white_base_blue_r = LinearSegmentedColormap.from_list(
            "my_colormap",
            (rels["white"], rels["lightblue"], rels["blue"], rels["darkblue"]),
            N=n,
            gamma=gamma,
        )

        self.white_base_bluegreen = LinearSegmentedColormap.from_list(
            "my_colormap",
            (rels["white"], rels["darkblue"], rels["blue"], rels["green"]),
            N=n,
            gamma=gamma,
        )
        self.white_base_bluegreen_r = LinearSegmentedColormap.from_list(
            "my_colormap",
            (
                rels["white"],
                rels["lightblue"],
                rels["blue"],
                rels["darkblue"],
                rels["group2"],
            ),
            N=n,
            gamma=gamma,
        )
        self.white_base_blgror = LinearSegmentedColormap.from_list(
            "my_colormap",  # rels['blue'],
            (
                rels["white"],
                rels["gridcolor"],
                rels["cascade2"],
                rels["cascade1"],
                rels["group1"],
            ),
            N=n,
            gamma=gamma,
        )

        self.rainbow = LinearSegmentedColormap.from_list(
            "my_colormap",
            (
                rels["darkblue"],
                rels["blue"],
                rels["lightblue"],
                rels["green"],
                rels["orange"],
                rels["red"],
            ),
            N=n,
            gamma=gamma,
        )
        self.cascade = LinearSegmentedColormap.from_list(
            "my_colormap",
            (rels["cascade1"], rels["cascade2"], rels["cascade3"], rels["cascade4"]),
            N=n,
            gamma=gamma,
        )
        self.cascade_r = LinearSegmentedColormap.from_list(
            "my_colormap",
            (
                rels["white"],
                rels["cascade4"],
                rels["cascade3"],
                rels["cascade2"],
                rels["cascade1"],
                rels["green"],
            ),
            N=n,
            gamma=gamma,
        )
        self.cascade_r_nowhite = LinearSegmentedColormap.from_list(
            "my_colormap",
            (
                rels["cascade4"],
                rels["cascade3"],
                rels["cascade2"],
                rels["cascade1"],
                rels["green"],
            ),
            N=n,
            gamma=gamma,
        )

        self.cascade_highcontrast = LinearSegmentedColormap.from_list(
            "my_colormap",
            (
                rels["cascade1"],
                rels["cascade3"],
                rels["aug1"],
                rels["grey"],
                rels["aug2"],
            ),
            N=n,
            gamma=gamma,
        )

        self.circle_small = LinearSegmentedColormap.from_list(
            "my_colormap",
            (rels["minus"], rels["plus"], rels["plus"], rels["minus"]),
            N=n,
            gamma=gamma,
        )
        self.circle_small_r = LinearSegmentedColormap.from_list(
            "my_colormap",
            (rels["plus"], rels["minus"], rels["minus"], rels["plus"]),
            N=n,
            gamma=gamma,
        )

        self.circle_medium = LinearSegmentedColormap.from_list(
            "my_colormap",
            (rels["red"], rels["green"], rels["darkblue"], rels["green"], rels["red"]),
            N=n,
            gamma=gamma,
        )
        self.circle_medium_r = LinearSegmentedColormap.from_list(
            "my_colormap",
            (
                rels["darkblue"],
                rels["green"],
                rels["red"],
                rels["green"],
                rels["darkblue"],
            ),
            N=n,
            gamma=gamma,
        )

        self.circle_big = LinearSegmentedColormap.from_list(
            "my_colormap",
            (
                rels["darkblue"],
                rels["blue"],
                rels["orange"],
                rels["red"],
                rels["orange"],
                rels["blue"],
            ),
            N=n,
            gamma=gamma,
        )
        self.circle_big_r = LinearSegmentedColormap.from_list(
            "my_colormap",
            (
                rels["red"],
                rels["orange"],
                rels["blue"],
                rels["darkblue"],
                rels["blue"],
                rels["orange"],
            ),
            N=n,
            gamma=gamma,
        )

        self.circle_medium_triple = LinearSegmentedColormap.from_list(
            "my_colormap",
            (
                rels["darkblue"],
                rels["orange"],
                rels["orange"],
                rels["red"],
                rels["red"],
                rels["darkblue"],
            ),
            N=n,
            gamma=gamma,
        )
        self.circle_medium_triple_r = LinearSegmentedColormap.from_list(
            "my_colormap",
            (
                rels["red"],
                rels["darkblue"],
                rels["darkblue"],
                rels["orange"],
                rels["orange"],
                rels["red"],
            ),
            N=n,
            gamma=gamma,
        )

    def colormaps(self, n, gamma=None):
        gamma = 1 if gamma is None else gamma
        self.divergingwhite = LinearSegmentedColormap.from_list(
            "my_colormap", (self.plus, self.white, self.minus), N=n, gamma=gamma
        )
        self.divergingwhitelarge = LinearSegmentedColormap.from_list(
            "my_colormap",
            (
                self.colors["darkblue"],
                self.colors["blue"],
                self.white,
                self.colors["orange"],
                self.colors["red"],
            ),
            N=n,
            gamma=gamma,
        )

        self.whiteblue = LinearSegmentedColormap.from_list(
            "my_colormap",
            (
                self.white,
                self.colors["lightblue"],
                self.colors["blue"],
                self.colors["darkblue"],
            ),
            N=n,
            gamma=gamma,
        )
        self.greyblue = LinearSegmentedColormap.from_list(
            "my_colormap",
            (
                self.gridcolor,
                self.colors["lightblue"],
                self.colors["blue"],
                self.colors["darkblue"],
            ),
            N=n,
            gamma=gamma,
        )
        self.greyblue_r = LinearSegmentedColormap.from_list(
            "my_colormap",
            (
                self.colors["darkblue"],
                self.colors["blue"],
                self.colors["lightblue"],
                self.gridcolor,
            ),
            N=n,
            gamma=gamma,
        )
        self.greyredorange = LinearSegmentedColormap.from_list(
            "my_colormap",
            (self.gridcolor, self.cascade3, self.colors["red"], self.colors["orange"]),
            N=n,
            gamma=gamma,
        )

        self.white_base_blue = LinearSegmentedColormap.from_list(
            "my_colormap",
            (
                self.white,
                self.colors["darkblue"],
                self.colors["blue"],
                self.colors["lightblue"],
            ),
            N=n,
            gamma=gamma,
        )
        self.white_base_bluegreen = LinearSegmentedColormap.from_list(
            "my_colormap",
            (
                self.white,
                self.colors["darkblue"],
                self.colors["blue"],
                self.colors["green"],
            ),
            N=n,
            gamma=gamma,
        )
        self.rainbow = LinearSegmentedColormap.from_list(
            "my_colormap",
            (
                self.colors["darkblue"],
                self.colors["blue"],
                self.colors["lightblue"],
                self.colors["green"],
                self.colors["orange"],
                self.colors["red"],
            ),
            N=n,
            gamma=gamma,
        )
        self.cascade = LinearSegmentedColormap.from_list(
            "my_colormap",
            (self.cascade1, self.cascade2, self.cascade3, self.cascade4),
            N=n,
            gamma=gamma,
        )
        self.cascade_r = LinearSegmentedColormap.from_list(
            "my_colormap",
            (
                self.white,
                self.cascade4,
                self.cascade3,
                self.cascade2,
                self.cascade1,
                self.colors["green"],
            ),
            N=n,
            gamma=gamma,
        )
        self.cascade_r_nowhite = LinearSegmentedColormap.from_list(
            "my_colormap",
            (
                self.cascade4,
                self.cascade3,
                self.cascade2,
                self.cascade1,
                self.colors["green"],
            ),
            N=n,
            gamma=gamma,
        )

        self.cascade_highcontrast = LinearSegmentedColormap.from_list(
            "my_colormap",
            (self.cascade1, self.cascade3, self.aug1, self.grey, self.aug2),
            N=n,
            gamma=gamma,
        )

    def show(self):
        for key in self.__dict__.keys():
            _logger.debug("key: %s", key)

        _logger.debug("  rels dict keys:")
        for key in self.rels.keys():
            _logger.debug("   %s", key)

    def plot(self):
        dd = self.__dict__.copy()
        dd_colors = dd["rels"]
        del dd["rels"]

        Y = 2.0
        coldd = dd
        ncolor = len(coldd)
        fig = plt.figure(figsize=[2, Y])
        ax = fig.add_axes([0, 0, 0.5, Y])
        dy = 1 / ncolor
        y = np.arange(0, 1, dy)

        j = 0
        for k, I in coldd.items():
            try:
                r1 = mpatch.Rectangle((0, y[j]), 1, dy, color=I)
                txt = ax.text(
                    1.1,
                    y[j] + dy * 0.5,
                    " " + k,
                    va="center",
                    fontsize=10,
                    weight="regular",
                )

                ax.add_patch(r1)
                j += 1
            except:
                pass

        coldd = dd_colors
        ncolor = len(coldd)
        ax1 = fig.add_axes([1.5, 0, 0.5, Y])
        dy = 1 / ncolor
        y = np.arange(0, 1, dy)

        j = 0
        for k, I in coldd.items():
            r1 = mpatch.Rectangle((0, y[j]), 1, dy, color=I)
            txt = ax1.text(
                1.1,
                y[j] + dy * 0.5,
                " " + k,
                va="center",
                fontsize=10,
                weight="regular",
            )

            ax1.add_patch(r1)
            j += 1
        plt.title("rels")

    def add_standard_colors(self):
        self.colors = dict()
        self.gridcolor = (0, 0, 0)

        self.colors["red"] = (203 / 255, 32 / 255, 39 / 255)
        self.colors["green"] = (15 / 255, 150 / 255, 72 / 255)
        self.colors["orange"] = (247 / 255, 191 / 255, 88 / 255)

        self.colors["grey1"] = (167 / 255, 180 / 255, 183 / 255)
        self.colors["grey"] = self.colors["grey1"]
        self.colors["lightgrey"] = self.colors["grey1"]
        self.colors["grey2"] = (123 / 255, 121 / 255, 125 / 255)
        self.colors["grey3"] = (72 / 255, 70 / 255, 77 / 255)

        self.colors["darkblue"] = (18 / 255, 78 / 255, 153 / 255)
        self.colors["blue"] = (85 / 255, 133 / 255, 196 / 255)
        self.colors["lightblue"] = (129 / 255, 140 / 255, 192 / 255)

        self.colors["blue4"] = (7 / 255, 137 / 255, 198 / 255)
        self.colors["black"] = (0, 0, 0)
        self.colors["white"] = (1, 1, 1)

    def default_colors(self):
        self.colors = dict()
        self.gridcolor = (0, 0, 0)

        self.colors["red"] = (203 / 255, 32 / 255, 39 / 255)
        self.colors["green"] = (15 / 255, 150 / 255, 72 / 255)
        self.colors["orange"] = (247 / 255, 191 / 255, 88 / 255)

        self.colors["grey1"] = (167 / 255, 180 / 255, 183 / 255)
        self.colors["grey"] = self.colors["grey1"]
        self.colors["lightgrey"] = self.colors["grey1"]
        self.colors["grey2"] = (123 / 255, 121 / 255, 125 / 255)
        self.colors["grey3"] = (72 / 255, 70 / 255, 77 / 255)

        self.colors["darkblue"] = (18 / 255, 78 / 255, 153 / 255)
        self.colors["blue"] = (85 / 255, 133 / 255, 196 / 255)
        self.colors["lightblue"] = (129 / 255, 140 / 255, 192 / 255)
        self.colors["blue4"] = (7 / 255, 137 / 255, 198 / 255)
        self.colors["black"] = (0, 0, 0)
        self.colors["white"] = (1, 1, 1)

        converter = {
            "lead1": "blue",
            "lead2": "red",
            "lead3": "green",
            "plus": "red",
            "minus": "blue",
            "pair2_a": "darkblue",
            "pair2_b": "green",
            "aug1": "green",
            "aug2": "orange",
            "axiscolor": "black",
            "gridcolor": "grey2",
            "cascade1": "darkblue",
            "cascade2": "blue",
            "cascade3": "blue4",
            "cascade4": "lightblue",
        }

        for k, I in converter.items():
            setattr(self, k, self.colors[I])
