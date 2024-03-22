"""
This library contains method, and classes used to search for the best angle given x,y data using single frequecy fits.
"""

from numba import jit
import numpy as np
import lmfit as LM
import matplotlib.pyplot as plt

numba_parallel = False


def get_wavenumbers_polar(amp, angle_rad):
    """
    inputs:

    amp     length of peak k vector in radial coordinates
    angle_rad   angle of peak k vector in radians between - pi/2 to  + pi/2

    returns:
    wavenumber k,l
    """

    k0 = amp * np.cos(angle_rad)
    l0 = amp * np.sin(angle_rad)

    return k0, l0


def wavemodel(XX, YY, ks, ls, amps, group_phase=0):

    G = np.vstack(
        [
            np.cos(np.outer(XX, ks) + np.outer(YY, ls)).T,
            np.sin(np.outer(XX, ks) + np.outer(YY, ls)).T,
        ]
    ).T

    b = np.hstack([np.cos(group_phase) * amps, np.sin(group_phase) * amps]).squeeze()
    z_model = G @ b

    return z_model


@jit(nopython=True, parallel=numba_parallel)
def wavemodel_single_wave(XX, YY, ks, ls, amps, group_phase=0):
    z_model = amps * np.cos(XX * ks + YY * ls + group_phase)
    return z_model


def get_z_model(x_positions, y_position, K_prime, K_amp, alpha_rad, group_phase):

    K_abs = K_prime / np.cos(alpha_rad)

    k = K_abs * np.cos(alpha_rad)
    l = K_abs * np.sin(alpha_rad)

    _wavemodel = wavemodel(
        x_positions, y_position, k, l, np.array(K_amp), group_phase=group_phase
    )
    return _wavemodel


@jit(nopython=True, parallel=False)
def get_z_model_single_wave(
    x_positions, y_position, K_prime, K_amp, alpha_rad, group_phase
):

    K_abs = K_prime / np.cos(alpha_rad)

    k = K_abs * np.cos(alpha_rad)
    l = K_abs * np.sin(alpha_rad)

    _wavemodel_single_wave = wavemodel_single_wave(
        x_positions, y_position, k, l, K_amp, group_phase=group_phase
    )

    return _wavemodel_single_wave


def objective_func(pars, x, y, z, test_flag=False, prior=None, prior_weight=2):

    "objective function that returns the residual array"
    z_model = get_z_model_single_wave(
        x,
        y,
        pars["K_prime"].value,
        pars["K_amp"].value,
        pars["alpha"].value,
        pars["phase"].value,
    )
    if prior is not None:
        a_0, a_std = prior["alpha"]
        penalties = np.array([((a_0 - pars["alpha"]) / a_std) ** 2])
    else:
        penalties = np.array([0])

    cost = ((z - z_model) / z.std()) ** 2
    if test_flag:
        return z_model
    else:
        residual = np.concatenate([cost, prior_weight * penalties])
        return residual


def likelyhood_func(
    pars, x, y, z, z_error=None, test_flag=False, prior=None, prior_weight=2
):
    """
    ---not well tested ---
    likelyhood function using log(p)

    """
    # get model
    z_model = get_z_model(
        x, y, pars["K_prime"], pars["K_amp"], pars["alpha"], pars["phase"]
    )
    # define cost
    cost_sqrt = (z - z_model) ** 2

    # estimate total variance
    if z_error is None:
        tot_var = z.std() ** 2 * 0.01 + z_model**2
    else:
        tot_var = z_error**2 + z_model**2

    def simple_log_panelty(x, x0, sigma):
        return -(((x - x0) / sigma) ** 2.0) / 2.0

    # try ot get prior
    if prior is not None:
        a_0, a_std = prior["alpha"]
        # this corresponds to the the panelty log( p(alpha) )
        penalties = simple_log_panelty(pars["alpha"], a_0, a_std)
    else:
        penalties = np.array([0])

    if test_flag:
        return z_model
    else:
        return (
            -0.5 * (cost_sqrt / tot_var + np.log(tot_var)).sum()
            + prior_weight * penalties
        )


class SampleWithMcmc:
    """
    sample a 2nd surface using mcmc and other methods. its made for getting a quick estimate!

    Example:
    SM = sample_with_mcmc(params_dict)
    k_prime_max= 0.01947787
    SM.params.add('K_prime', k_prime_max ,  vary=False  , min=k_prime_max*0.5, max=k_prime_max*1.5)
    amp_Z= 1
    SM.params.add('K_amp', amp_Z         ,  vary=False  , min=amp_Z*.0       , max=amp_Z*5)

    SM.set_objective_func(objective_func)


    SM.fitting_args = fitting_args = (x_concat, y_concat, z_concat)
    SM.fitting_kargs = fitting_kargs = {'prior': None , 'prior_weight' : 10 }

    try:
        SM.test_objective_func()
    except:
        raise ValueError('Objective function test fails')


    SM.brute(verbose= True)
    SM.plot_brute()
    SM.sample(verbose= True, steps=200,)
    SM.optimize(verbose= True)


    """

    def __init__(self, params):

        self.LM = LM

        self.set_parameters(params)
        self.prior = None
        self.prior_weight = 2

    def set_objective_func(self, ofunc):
        self.objective_func = ofunc

    def set_parameters(self, par_dict, verbose=False):
        """
        defines params object at inital seed for mcmc
        par_dict should contain: var_name : [min, max, nseed]
        """

        params = self.LM.Parameters()

        for k, I in par_dict.items():
            params.add(k, (I[0] + I[1]) / 2, vary=True, min=I[0], max=I[1])

        var_seeds = [np.linspace(I[0], I[1], I[2]) for _, I in par_dict.items()]

        if len(var_seeds) > 2:
            raise ValueError("Dimensions larger than 2 not supported")

        self.nwalkers = var_seeds[0].size * var_seeds[1].size

        pxx, pyy = np.meshgrid(var_seeds[0], var_seeds[1])
        self.seeds = np.vstack([pxx.flatten(), pyy.flatten()]).T
        self.params = params
        if verbose:
            print("Nwalker: ", self.nwalkers)
            print("Seeds: ", self.seeds.shape)
            print(self.params)

    def test_objective_func(self):
        obj_func = self.objective_func(
            self.params, *self.fitting_args, **self.fitting_kargs
        )
        return obj_func

    def sample(
        self, fitting_args=None, method="emcee", steps=100, verbose=True, **kwargs
    ):

        fitting_args, fitting_kargs = self.fitting_args, self.fitting_kargs
        # TODO: this function  throws an error in CI. The nan_policy='omit' policy  was added to avoid this issue
        # according to the guidelines in  https://lmfit.github.io/lmfit-py/faq.html#i-get-errors-from-nan-in-my-fit-what-can-i-do
        self.fitter = self.LM.minimize(
            self.objective_func,
            self.params,
            method=method,
            args=fitting_args,
            kws=fitting_kargs,
            nwalkers=self.nwalkers,
            steps=steps,
            pos=self.seeds,
            nan_policy="omit",
            **kwargs,
        )
        if verbose:
            print(self.LM.report_fit(self.fitter))
            print("results at self.fitter")

    def plot_sample(self, **kwargs):

        chain = self.chain()
        nwalkers = self.nwalkers
        for n in np.arange(nwalkers):
            plt.plot(chain[:, n, 1], chain[:, n, 0], "-", **kwargs)
            plt.plot(chain[:, n, 1], chain[:, n, 0], ".", **kwargs)

    def optimize(self, fitting_args=None, method="dual_annealing", verbose=True):

        fitting_args = self.fitting_args
        fitting_kargs = self.fitting_kargs

        self.fitter_optimize = self.LM.minimize(
            self.objective_func,
            self.params,
            method=method,
            args=fitting_args,
            kws=fitting_kargs,
        )
        if verbose:
            print(self.LM.report_fit(self.fitter_optimize))
            print("results at self.fitter_optimize")

    def plot_optimze(self, **kargs):

        plt.plot(
            self.fitter_optimize.params["phase"].value,
            self.fitter_optimize.params["alpha"].value,
            ".",
            **kargs,
        )

    def brute(self, fitting_args=None, method="brute", verbose=True, N_grid=30):

        fitting_args = self.fitting_args
        fitting_kargs = self.fitting_kargs

        self.fitter_brute = self.LM.minimize(
            self.objective_func,
            self.params,
            method=method,
            args=fitting_args,
            kws=fitting_kargs,
            Ns=N_grid,
        )

        if verbose:
            print(self.LM.report_fit(self.fitter_brute))
            print("results at self.fitter_brute")

    def plot_brute(self, clevel=np.linspace(-3.2, 3.2, 30), **kargs):

        fitter_brute = self.fitter_brute

        dd = (
            fitter_brute.brute_Jout - fitter_brute.brute_Jout.mean()
        ) / fitter_brute.brute_Jout.std()
        plt.contourf(
            fitter_brute.brute_grid[1, :, :],
            fitter_brute.brute_grid[0, :, :],
            dd,
            clevel,
            cmap=plt.cm.YlGnBu_r,
        )

        x_name, y_name = (
            list(fitter_brute.params.keys())[1],
            list(fitter_brute.params.keys())[0],
        )
        plt.xlabel(x_name)
        plt.ylabel(y_name)

    def chain(self, burn=None):
        "return results as nparray contains walk of each walker"
        if burn is not None:
            return self.fitter.chain[burn:, :, :]
        else:
            return self.fitter.chain

    def flatchain(self, burn=None):
        "returns results as pandas table"
        if burn is not None:
            return self.fitter.flatchain.loc[burn:]
        else:
            return self.fitter.flatchain

    def get_marginal_dist(
        self, var, var_dx, burn=None, plot_flag=False, normalize=True
    ):
        """
        retrurn the marginal distribution from self.params object

        inputs:
        var         variable name in self.params
        var_dx      bin distance fr histogram
        burn        (None) amount of samples burned at the beginning of the MCMC
        plot_flag   (False) If True its plots a stairs plot of the resulting histogram
        normalize   (True) if True the histogram is normalized by the the amount of data points,
                    otherwise it returns the rar histogram
        """

        bins = np.arange(self.params[var].min, self.params[var].max + var_dx, var_dx)

        y_hist, _ = np.histogram(self.fitter.flatchain.loc[burn:][var], bins)
        bins_pos = bins[0:-1] + np.diff(bins) / 2

        if normalize:
            y_hist = y_hist / var_dx / y_hist.sum()

        if plot_flag:
            plt.stairs(y_hist, bins)

        return y_hist, bins, bins_pos
