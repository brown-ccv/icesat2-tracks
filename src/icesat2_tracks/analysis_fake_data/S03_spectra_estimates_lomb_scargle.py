import os, sys

# execfile(os.environ['PYTHONSTARTUP'])

"""
This file open a ICEsat2 track applied filters and corections and returns smoothed photon heights on a regular grid in an .nc file.
This is python 3
"""
# exec(open(os.environ['PYTHONSTARTUP']).read())
# exec(open(STARTUP_2019_DP).read())

base_path = "/Users/Shared/Projects/2021_IceSAT2_tracks/"
sys.path.append(base_path + "modules/")
sys.path.append(base_path + "modules/ICEsat2_SI_tools/")

import matplotlib.pyplot as plt

# %matplotlib inline

# import m_general as M
# import m_tools as MT
import numpy as np
import m_general_ph3 as M

import netCDF4
import datetime
import os
import xarray as xr

import ICEsat2_SI_tools.convert_GPS_time as cGPS
import ICEsat2_SI_tools.io as io
import ICEsat2_SI_tools.spectral_estimates as spec

import imp

# import s3fs
# %%
Lx = 10000
xx = np.arange(0, Lx, 0.1)

lam = 10
k = 1 / lam
print(k)
k2 = 1 / 0.21
print(k2)
# yy = np.sin( k * xx  * 2 * np.pi ) + np.sin(  k2 * xx  * 2 * np.pi ) + 0.1 *np.random.randn(len(xx))

decay = np.exp(-0.2 * (np.arange(len(xx)) / Lx))
plt.plot(decay)
yy = 2.3 * np.sin(k * xx * 2 * np.pi) + 4 * np.sin(
    10 * k * xx * 2 * np.pi
)  #  +   0.05 *np.random.randn(len(xx))
plt.plot(yy)


# %% test simple pwelch wavenumber spectrum

L = 1000
imp.reload(spec)
S = spec.wavenumber_pwelch(
    yy, xx, L, ov=None, window=np.ones(L), save_chunks=False, plot_chunks=False
)


S.specs.shape
S.spec_est.shape
S.k[1]
S.k[S.spec_est.argmax()]

plt.plot(S.k, S.spec_est)

# np.sqrt(2 *S.spec_est *S.df).max()
plt.plot(S.k, np.sqrt(2 * S.spec_est * S.df))
plt.ylabel("wave height(m)")

# %% test class
# imp.reload(specs)
# S = spec.wavenumber_spetrogram(xx, yy, L)
# G = S.cal_spectrogram()
# S.mean_spectral_error() # add x-mean spectal error estimate to xarray
#
# S.parceval(add_attrs= True)
# test of frequency is recovered
# G.k[G.mean('x').argmax().data].data

# %% lomb-scargle
from astropy.timeseries import LombScargle

k, dk = spec.calc_freq_fft(xx, yy.size)
spec_fft = spec.calc_spectrum_fft(yy, dk, yy.size)

ls = LombScargle(xx, yy)
k_ls, power_ls = ls.autopower(minimum_frequency=k[1], maximum_frequency=k[-1])
k_ls2 = k[1:][::3]
power2 = ls.power(k_ls2)


def ls_power_to_fft_power(power2, N):
    """
    convert ls output to spectral density amplitudes |unit^2|/dk as used for standard fft
    """
    return N * power2


def ls_power_to_wave_height(power2, N, dk):
    """
    convert ls output to spectral density amplitudes of [units]
    """
    return np.sqrt(2 * N * power2 * dk)


plt.plot(k, spec_fft)
plt.plot(k_ls, power_ls * 2 / np.gradient(k_ls)[0])
plt.plot(k_ls, ls_power_to_fft_power(power_ls, yy.size))

plt.plot(k_ls2, ls_power_to_wave_height(power2, yy.size, dk), ".")
plt.plot(k, np.sqrt(2 * spec_fft * dk))

# %% test wave spectra
# make fake data
import JONSWAP_gamma

dff = 0.001  # 0.0005#0.001
f = np.arange(0.01, 0.2, dff)
dff / f[0]
spec_power = JONSWAP_gamma.JONSWAP_default(f, 1e6, 10)
# plt.plot(f, spec_power)
# variance
print("JONSWAP variance", (spec_power * dff).sum())

omega = 2 * np.pi * f

dt = 0.05  # 0.01
t = np.arange(0, 10000, dt)
t.size / 40


amps = (2 * spec_power * dff) ** 0.5

ww, tt = np.meshgrid(2 * np.pi * f, t)
phi = np.random.random(len(amps)) * 2 * np.pi

instance = amps * np.sin(ww * tt + phi)
y = instance.sum(1)  # + 0.05* np.random.random(len(t))

print("data variance", y.var())  #

#

M.figure_axis_xy(6, 4, view_scale=0.5)
plt.subplot(2, 1, 1)
plt.plot(f, spec_power)
plt.ylabel("spectral power m^2/Hz")

plt.subplot(2, 1, 2)
plt.plot(t, y)
plt.xlim(0, 500)

# %% manual FFT test
#
# f_phi, df = spec.calc_freq_fft(t, t.size)
# phi = np.fft.rfft(y)
# phi_p=   2 * abs(phi)**2 * dt / df / t.size**2
# (phi_p * df).sum()
#
# plt.plot(f_phi, phi_p )
#
# plt.xlim(0, 0.2)

# %%
min_datapoint = 1 / f[0] / dt

# % standard FFT
print(t.size)

print(min_datapoint)
L = int(min_datapoint * 10)
print(L)

S = spec.wavenumber_pwelch(y, t, L)


f_fft2, df2 = spec.calc_freq_fft(t, t.size)
spec_fft2 = spec.calc_spectrum_fft(y, df2, y.size)


ls = LombScargle(t, y)
# #k_ls2 = k[1:][::3]
f_fft = S.k
f_fft = f_fft[(f_fft > 0.05) & (f_fft < 0.5)]
ls_power = ls.power(
    f_fft, normalization="psd", assume_regular_frequency="True", method="auto"
)
len(f_fft)

ls_auto_f, ls_auto_power = ls.autopower(
    minimum_frequency=0.05,
    maximum_frequency=0.5,
    normalization="psd",
    samples_per_peak=0.1,
)
len(ls_auto_f)
M.figure_axis_xy(5, 4, view_scale=0.8)

# orginal
plt.plot(f, spec_power, "k", linewidth=6, alpha=0.3, label="original")
#
plt.plot(S.k, S.spec_est, label="Pwelch FFT")
print("pwelch FFT variance", (S.spec_est * S.df).sum())

# plt.plot(f_fft2[1:], spec_fft2[1:], '-',alpha = 0.4,  label = 'FFT')
# print('pure FFT', (spec_fft2  * np.gradient(f_fft2)[0] ).sum() )
df = np.gradient(S.k)[0]

plt.plot(f_fft, 2 * ls_power / t.size / dff, "-+", label="LS")
plt.plot(ls_auto_f, 2 * ls_auto_power / t.size / dff, "-+", label="LS auto")
plt.legend()
plt.ylim(0, 21)
plt.xlim(0, 0.2)


# %%

t_noise = t + 0.05 * np.random.randn(len(t))
y_noise_instance = 1 * np.random.randn(len(t))
y_noise = y + y_noise_instance

# #k_ls2 = k[1:][::3]
# ls_f_noise= S.k
# ls_f_noise = ls_f_noise[ (ls_f_noise> 0.05) & (ls_f_noise < 0.5) ]
ls_f_noise = np.arange(S.k[1], S.k[-1], np.diff(S.k)[0] / 2)  #
# ls_f_noise = np.arange(0.05, 0.5,np.diff(S.k)[0]/2 ) # increase the spectral resolution to double of the fft one
ls_noise = LombScargle(t_noise, y_noise, fit_mean=True)
ls_noise.offset()
# %time ls_power_noise = ls_noise.power(ls_f_noise, normalization='psd')
len(ls_f_noise)
np.diff(ls_f_noise)[0]


ls_noise_auto = LombScargle(t_noise, y_noise, fit_mean=True)

# %time ls_auto_f_noise, ls_auto_power_noise = ls_noise_auto.autopower(minimum_frequency=0.05, maximum_frequency=0.25, normalization='psd', samples_per_peak=0.1)
len(ls_auto_f_noise)
# np.diff(ls_auto_f_noise)[0]

plt.plot(f, spec_power, "k", linewidth=2, label="orginal")
print("JONSWAP variance", (spec_power * np.diff(f).mean()).sum())


plt.plot(ls_f_noise, 2 * ls_power_noise / t.size / dff, "-+", label="LS noise")
ls_power_noise_P = 2 * ls_power_noise / t.size / dff * np.diff(ls_f_noise).mean()
print("LS noise variance", ls_power_noise_P.sum())


plt.plot(
    ls_auto_f_noise, 2 * ls_auto_power_noise / t.size / dff, "-+", label="LS auto noise"
)


print("data variance", y_noise.var(), "noise variance ", y_noise_instance.var())
print("diff variance", y_noise.var() - y_noise_instance.var())

plt.legend()
plt.ylim(0, 21)
plt.xlim(0, 0.2)

# %%
M.figure_axis_xy(6, 4)
plt.plot(ls_f_noise, ls_power_noise_P.cumsum(), label="LS noise")
plt.plot(
    ls_f_noise,
    ls_f_noise * 0 + (spec_power * np.diff(f).mean()).sum(),
    "k",
    label="signal variance",
)
plt.plot(ls_f_noise, ls_f_noise * 0 + y_noise.var(), "k--", label="total variance")

plt.legend()
plt.xlabel("wavenumber")
plt.ylabel("cumulative variance")

# %% test phase
f_sel = ls_f_noise  # [ (ls_f_noise> 0.05) & (ls_f_noise < 0.1) ]
m1 = ls_noise.offset() * np.ones(len(t_fit))
for fi in f_sel:
    m1 += ls_noise.model(t_fit, fi)

f_sel = ls_auto_f_noise  # [ (f_fft> 0) & (f_fft < 0.8) ]
m2 = ls_noise_auto.offset() * np.ones(len(t_fit))
for fi in f_sel:
    m2 += ls_noise_auto.model(t_fit, fi)

# alternative way
# design_matrix = ls_noise.design_matrix(best_frequency, t_fit)

# %%
plt.plot(t_fit, m1 - m1.mean(), zorder=12, label="m1 | LS noise")
plt.plot(t_fit, m2 - m2.mean(), zorder=12, label="m2 | LS auto noise")
plt.plot(t_noise[0:2000], y_noise[0:2000], ".")

plt.legend()

# %%
plt.plot(t_fit, y_noise[0:2000] - m2, ".")

# %% define functions:
# def calc_freq_LS
ls_f_noise = np.arange(S.k[1], S.k[-1], np.diff(S.k)[0] / 2)  #
# ls_f_noise = np.arange(0.05, 0.5,np.diff(S.k)[0]/2 ) # increase the spectral resolution to double of the fft one
ls_noise = LombScargle(t_noise, y_noise, fit_mean=True)

ls_noise.offset()
ls_power_noise = ls_noise.power(ls_f_noise, normalization="psd")
len(ls_f_noise)
np.diff(ls_f_noise)[0]


ls_noise_auto = LombScargle(t_noise, y_noise, fit_mean=True)

ls_auto_f_noise, ls_auto_power_noise = ls_noise_auto.autopower(
    minimum_frequency=0.05,
    maximum_frequency=0.25,
    normalization="psd",
    samples_per_peak=0.1,
)


ls_auto_f_noise - ls_auto_f_noise2

ls_noise = LombScargle(t_noise, y_noise, fit_mean=True)
ls_auto_f_noise2 = ls_noise.autofrequency(
    minimum_frequency=0.05, maximum_frequency=0.25, samples_per_peak=0.1
)
len(ls_auto_f_noise2)


# %%
# ls_noise = LombScargle(t_noise[0:10000], y_noise[0:10000], fit_mean=True)
# ls_auto_f_noise3 =  ls_noise.autofrequency(minimum_frequency=0.05, maximum_frequency=0.25, samples_per_peak=0.1)


min_datapoint = 1 / f[0] / dt

L = 1000
f3, df3 = spec.calc_freq_fft(t_noise, L)
f3.size

# def calc_freq_LS(x, N, method='fftX2', minimum_frequency=None, maximum_frequency=None, samples_per_peak=0.01):
#     """
#     calculate array of spectral variable (frequency or
#     wavenumber) in cycles per unit of N (window length)
#     x can be unevenly spaced
#     method:
#         "fftX2"     defined the frequencyt grid as for FFT, but with double its resolution
#         "LS_auto"   using LS algorithm with samples_per_peak=0.1
#
#         minimum_frequency, maximum_frequency only used for LS_auto
#     """
#
#     if method is 'fftX2':
#         neven = True if (N%2) else False
#         dx=np.diff(x).mean()
#         df = 1./((N-1)*dx) /2
#         np.round(df, 5)
#         if neven:
#             f = df*np.arange(df, N+1)
#         else:
#             f = df* np.arange(df, (N-1)  + 1 )
#
#     elif method is 'LS_auto':
#         from astropy.timeseries import LombScargle
#         f = LombScargle(x , np.random.randn(len(x)), fit_mean=True).autofrequency(minimum_frequency=minimum_frequency, maximum_frequency=maximum_frequency, samples_per_peak=samples_per_peak)##0.1)
#
#         df = np.diff(f).mean()
#         df = np.round(df, 5)
#
#     elif method is 'fixed_ratio':
#
#         neven = True if (N%2) else False
#         dx=np.diff(x).mean()
#         df = dx / 50
#         if neven:
#             f = df*np.arange(df, N +1)
#         else:
#             f = df* np.arange(df, N )
#
#     return f ,df
#
# f3, df3 = calc_freq_LS(t_noise, L, method = 'fftX2')
# f4, df4 = calc_freq_LS(t_noise, L, method = 'LS_auto', minimum_frequency=0, maximum_frequency=5)

# for tesing:
f3, df3 = spec.calc_freq_LS(t_noise, L, method="fftX2")
f4, df4 = spec.calc_freq_LS(
    t_noise, L, method="LS_auto", minimum_frequency=0, maximum_frequency=5
)


# %%
imp.reload(spec)


# dt is the sampling frequency
# f[0] is tje lowerest resolved frequuency whee there is expected to be power
min_datapoint = 1 / f[0] / dt
#'LS_auto','fixed_ratio'
# L = 10000
L = int(min_datapoint * 10)
# L = int(y_noise.size/2)
S2 = spec.wavenumber_spectrogram_LS(
    t_noise,
    y_noise,
    L,
    waven_method="fixed_ratio",
    dy=None,
    ov=None,
    window=None,
    kjumps=1,
)

dx = np.diff(t_noise).mean()

# %time G = S2.cal_spectrogram()

print(" L / k legnth  ", L / len(S2.k))
print("dk  / dx ", S2.dk / dx)
#
# S2.dk
# dx
# df =
1.0 / ((L - 1) * dx)
#
# 0.02

# %
plt.plot(f, spec_power, "k", linewidth=2, label="orginal", zorder=12)

plt.plot(G.k, G, "-", label="LS noise")
# plt.plot(ls_auto_f_noise,  2 * ls_auto_power_noise / t.size /dff,  '-+' , label= 'LS auto noise')

# plt.legend()
# plt.ylim(0, 21)
plt.xlim(0, 0.2)

S2.dk * S2.G.mean("x").sel(k=slice(0, 0.5)).sum().data
# %%
# wavenumber_spectrogram_LS.calc_var
S2.calc_var()
S2.parceval()


S2.mean_spectral_error()


plt.plot(S2.G.k, S2.G.mean("x"), "-", label="LS noise")
plt.plot(S2.G.k, S2.G["mean_El"], "-", label="LS noise")
plt.plot(S2.G.k, S2.G["mean_Eu"], "-", label="LS noise")
plt.xlim(0, 0.20)


# print( 'JONSWAP variance' ,(spec_power * dff).sum() )
# %%
np.log(G).sel(k=slice(0, 0.5)).plot()
