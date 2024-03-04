import numpy as np 

def amplitudes_from_spectrum(spec, freq):
    """
    returns amplitude (m) per frequency
    input:
        spec: spectrum power (m^2/Hz)
        freq: frequency (Hz)
    returns:
        amps: amplitude (m) per frequency
    """
    return ( 2 * spec * np.gradient(freq)) **0.5

def amplitude_from_wavenumber_spectrum(spec, wavenumber):
    """
    returns amplitude (m) per wavenumber
    input:
        spec: spectrum power (m^2/Hz)
        wavenumber: wavenumber (2 pi /m)
    """
    return ( 2* spec * np.gradient(wavenumber)) **0.5

def time_realization_from_spectrum( T, f, spec_power):
    """
    returns a realization of a random process with a given spectrum
    input:
        T: time vector (s)
        f: frequency vector (Hz)
        spec_power: spectrum power (m^2/Hz)
        phi: phase (rad)
    returns:
        instance: realization of a random process with a given spectrum
    """
    tt, ww = np.meshgrid(2* np.pi * f, T)
    phi = np.random.random(len(spec_power))*2*np.pi
    return (amplitudes_from_spectrum(spec_power, f)* np.cos( ww * tt + phi )).sum(1)

def space_realization_from_spectrum( x, k, spec_power):
    """
    returns a realization of a random process with a given spectrum
    input:
        x: space vector (m)
        k: wavenumber vector (2 pi /m)
        spec_power: spectrum power (m^2/Hz)
    returns:
        instance: realization of a random process with a given spectrum    
    """
    kk, xx = np.meshgrid(k, x)
    phi = np.random.random(len(spec_power))*2*np.pi
    return (amplitude_from_wavenumber_spectrum(spec_power, k)* np.cos( kk * xx + phi )).sum(1)


def test_variance_conservations(f, spec, realization, wave_number=False):
    """
    test if the variance is conserved
    input:
        f: frequency vector (Hz)
        spec: spectrum power (m^2/Hz)
        realization: realization of a random process with a given spectrum
    Prints the Energy of the realization and of the spectrum
    """
    if ~wave_number:
        spectral_energy = np.trapz(spec, f)
    else:
        spectral_energy = np.trapz(spec, f) / (2 * np.pi)
    return print("Energy of realization",  np.var(realization), "\nEnergy of spectrum", spectral_energy, "\nDifference ", np.round(np.var(realization) - spectral_energy , 5 ))
