import logging

import os
import matplotlib.pyplot as plt
import numpy as np
import pickle
from matplotlib import dates
from datetime import datetime
import json
import pandas as pd
import h5py
import warnings
from pandas import HDFStore
from pandas.io.pytables import PerformanceWarning
import glob

_logger = logging.getLogger(__name__)


def dt_form_timestamp(timestamp, unit="h"):
    return (timestamp[1] - timestamp[0]).astype(f"m8[{unit}]")


def tick_formatter(a, interval=2, rounder=2, expt_flag=True, shift=0):

    fact = 10 ** (int(np.log10(a.max())) - 1)
    b = np.round(a / fact, rounder + 1) * fact
    ticklabels = [" "] * len(b)

    tt = np.arange(shift, len(b), interval)

    for t in tt:
        if expt_flag:
            ticklabels[int(t)] = f"{b[t]:.2e}"
        else:

            ticklabels[int(t)] = f"{b[t]:.2f}".rstrip("0").rstrip(".")

    return ticklabels, b


def freq_lim_string(low, high):
    a = f"{low:.1e}"
    b = f"{high:.1e}"
    return f"{a[0:3]}-{b} Hz"


def float_to_str(flt, r=1):
    return str(np.round(flt, r))


def num_to_str(flt):
    return str(int(np.round(flt, 0)))


def mkdirs_r(path):

    if not os.path.exists(path):
        os.makedirs(path)


def check_year(inputstr, yearstring):
    a, ref = [np.datetime64(t).astype(object).year for t in (inputstr, yearstring)]
    return a == ref


def datetime64_to_sec(d):
    return d.astype("M8[s]").astype("float")


def datetime64_to_day(d):
    return d.astype("M8[D]").astype("float")


def float_plot_time_to_sec(pp):
    return np.datetime64(dates.num2date(pp)).astype("M8[s]").astype("float")


def float_plot_time_to_dt64(pp):
    return np.datetime64(dates.num2date(pp)).astype("M8[s]")


def sec_to_dt64(pp):
    return pp.astype("M8[s]")


def sec_to_float_plot(pp):
    return dates.date2num(pp.astype("M8[s]").astype(datetime))


def sec_to_float_plot_single(pp):
    return dates.date2num(np.datetime64(int(pp), "s").astype("M8[s]").astype(datetime))


def fake_2d_data(verbose=True, timeaxis=False):
    x = np.arange(0, 100, 1)
    y = np.arange(0, 40, 1)
    XX, YY = np.meshgrid(x, y)

    mu = x.size / 2
    sigma = x.size / 5
    z2 = 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-((XX - mu) ** 2) / (2 * sigma**2))
    z2 = z2 / z2.max()

    mu = y.size / 2
    sigma = y.size / 5
    z3 = 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-((YY - mu) ** 2) / (2 * sigma**2))
    z3 = z3 / z3.max()
    if verbose:
        _logger.debug("x %s", x.shape)
        _logger.debug("y %s", y.shape)
        _logger.debug("z %s", z3.shape)

        plt.contourf(x, y, z2 / 2 + z3 / 2)
        plt.colorbar()
        plt.axis("scaled")
        plt.show()

    return x, y, z3


def pickle_save(name, path, data, verbose=True):
    if not os.path.exists(path):
        os.makedirs(path)
    full_name = os.path.join(path, name + ".npy")

    with open(full_name, "wb") as f2:
        pickle.dump(data, f2)
    if verbose:
        _logger.debug("save at: %s", full_name)


def pickle_load(name, path, verbose=True):
    full_name = os.path.join(path, name + ".npy")

    with open(full_name, "r") as f:
        data = pickle.load(f)

    if verbose:
        _logger.debug("load from: %s", full_name)
    return data


def json_save(name, path, data, verbose=False, return_name=False):

    if not os.path.exists(path):
        os.makedirs(path)
    full_name_root = os.path.join(path, name)
    full_name = os.path.join(full_name_root + ".json")
    with open(full_name, "w") as outfile:
        json.dump(data, outfile, indent=2)
    if verbose:
        _logger.debug("save at: %s", full_name)
    if return_name:
        return full_name_root


def json_save2(name, path, data, verbose=False, return_name=False):

    class CustomJSONizer(json.JSONEncoder):
        def default(self, obj):
            return bool(obj) if isinstance(obj, np.bool_) else super().default(obj)

    if not os.path.exists(path):
        os.makedirs(path)
    full_name_root = os.path.join(path, name)
    full_name = os.path.join(full_name_root + ".json")
    with open(full_name, "w") as outfile:
        json.dump(data, outfile, cls=CustomJSONizer, indent=2)
    if verbose:
        _logger.debug("save at: %s", full_name)
    if return_name:
        return full_name_root


def json_load(name, path, verbose=False):
    full_name = os.path.join(path, name + ".json")

    with open(full_name, "r") as ifile:
        data = json.load(ifile)
    if verbose:
        _logger.debug("loaded from: %s", full_name)
    return data


def h5_load(name, path, verbose=False):

    full_name = os.path.join(path, name + ".h5")
    data = pd.read_hdf(full_name)
    return data


def h5_load_v2(name, path, verbose=False):

    with h5py.File(path + name + ".h5", "r") as h5f:
        if verbose:
            _logger.debug("%s h5f keys: %s", name, h5f.keys())

        data_dict = {k: v[:] for k, v in h5f.items()}

    return data_dict


def h5_save(name, path, data_dict, verbose=False, mode="w"):
    if not os.path.exists(path):
        os.makedirs(path)

    full_name = os.path.join(path, name + ".h5")
    with h5py.File(full_name, mode) as store:
        for k, I in data_dict.items():
            store[k] = I

    if verbose:
        _logger.debug("saved at: %s", full_name)


def load_pandas_table_dict(name, save_path):

    warnings.filterwarnings("ignore", category=PerformanceWarning)

    return_dict = dict()
    with HDFStore(save_path + "/" + name + ".h5") as store:
        for k in store.keys():
            return_dict[k[1:]] = store.get(k)

    return return_dict


def save_pandas_table(table_dict, ID, save_path):

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    warnings.filterwarnings("ignore", category=PerformanceWarning)

    with HDFStore(save_path + "/" + ID + ".h5") as store:
        for name, table in table_dict.items():
            store[name] = table


def write_log(hist, string, verbose=False, short=True, date=True):

    _now = datetime.datetime.now()
    _short = "%Y%m%d"
    _long = "%Y-%m-%d %H:%M"
    now = _now.strftime(_short) if short else _now.strftime(_long)
    message = f"\n{now} {string}" if date else f"\n  {string}"

    if verbose in [True, "all"]:
        _logger.debug(
            "hist message: %s", hist + message if verbose == "all" else message
        )

    return hist + message


def add_line_var(ss, name, var):
    return f"{ss}\n  {name.ljust(5)} {var}"


def write_variables_log(hist, var_list, locals, verbose=False, date=False):

    now = datetime.now().strftime("%Y%m%d")

    var_dict = {name: locals[name] for name in var_list}
    stringg = "\n".join([f"{name.ljust(5)}{I}" for name, I in var_dict.items()])

    message = f"\n{now} {stringg}" if date else f"\n{' '.ljust(5)} {stringg}"

    if verbose in [True, "all"]:
        _logger.debug(
            "write_variables_log: %s", hist + message if verbose == "all" else message
        )


def save_log_txt(name, path, hist, verbose=False):
    if not os.path.exists(path):
        os.makedirs(path)
    full_name = os.path.join(path, name + ".hist.txt")
    with open(full_name, "w") as ifile:
        ifile.write(str(hist))
    if verbose:
        _logger.debug("saved at: %s", full_name)


def load_log_txt(hist_file, path):
    f = []
    for h in glob.glob(os.path.join(path, hist_file)):
        with open(h, "r") as file:
            f.append(file.read())
    return "\n".join(f)


def shape(a):
    for i in a:
        _logger.debug("shape of i=%s: %s", i, i.shape)


def find_O(a, case="round"):
    if case == "round":
        for k in np.logspace(0, 24, 25):
            if np.ceil(a / k) == 1:
                return k
                break
    elif case == "floor":
        for k in np.logspace(0, 24, 25):
            if np.ceil(a / k) == 1:
                return k
                break

    elif case == "ceil":
        for k in np.logspace(0, 24, 25):
            if np.ceil(a / k) == 1:
                return k
                break
    else:
        raise Warning("no propper case")


def stats(a):
    _logger.debug(
        "shape: %s\nNans: %s\nmax: %s\nmin: %s\nmean: %s",
        a.shape,
        np.sum(np.isnan(a)),
        np.nanmax(a),
        np.nanmin(a),
        np.nanmean(a),
    )


def stats_format(a, name=None):
    _logger.debug(
        "Name: %s\n"
        "   Shape: %s\n"
        "   NaNs: %s\n"
        "   max: %s\n"
        "   min: %s\n"
        "   mean: %s",
        name,
        a.shape,
        np.sum(np.isnan(a)),
        np.nanmax(a),
        np.nanmin(a),
        np.nanmean(a),
    )
