
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
import concurrent.futures as futures

col.colormaps2(21)
# %%



Lx, Ly = 20, 10
x = np.linspace(0,Lx,200)
y = np.linspace(0, Ly,100)

xx,yy = np.meshgrid(x, y)

def objective_func(params):
    sn2  = 0.1**2
    return - cost(params['x'], params['y'])  + np.log(sn2)

def cost(x, y):
    z = 4+ np.sin(4*  2 * np.pi *x/Lx) +  np.sin( 3 * np.pi *x/Lx - np.pi/5) + np.cos(1*  2 * np.pi *y/Ly) +  np.sin( 3 * np.pi *y/Ly - np.pi/3)
    return z**2

class sample_with_mcmc:
    """
    sample a 2nd surface using mcmc. its make for getting a quick estimate!
    """

    def __init__(self):

        import lmfit as LM
        self.LM = LM

    def objective_func(self, params):
        sn2  = 0.1**2
        return - cost(params['x'], params['y'])  + np.log(sn2)

    # def test_ojective_func(self, model_func):
    #     return self.objective_func(self.params, self.data, model_func, self.freq)

    def set_parameters(self, par_dict, verbose= False):
        """
        defines params object at inital seed for mcmc
        par_dict should contain: var_name : [min, max, nseed]
        """

        params = self.LM.Parameters()

        var_seeds = list()
        for k,I in par_dict.items():
            params.add(k, (I[0]+ I[1])/2,  vary=True  , min=I[0], max=I[1])

            var_seeds.append( np.linspace(I[0],I[1], I[2]))


        if len(var_seeds) > 2:
            raise ValueError('nor proframmed for 3d')

        self.nwalkers= int(var_seeds[0].size * var_seeds[1].size)

        pxx, pyy = np.meshgrid(var_seeds[0], var_seeds[1])
        self.seeds = np.vstack([pxx.flatten(), pyy.flatten() ]).T
        self.params = params
        if verbose:
            print('Nwalker: ', self.nwalkers)
            print('Seeds: ', self.seeds.shape)
            print(self.params)

    def sample(self, fitting_args= None , method='emcee', steps=100, verbose= True):

        self.fitter = self.LM.minimize(self.objective_func, self.params,  method=method,
                        nwalkers=self.nwalkers, steps=steps, pos= self.seeds)
        if verbose:
            print(self.LM.report_fit(self.fitter))

    def optimize(self, fitting_args= None , method='dual_annealing', verbose= True):

        self.fitter_optimize = self.LM.minimize(self.objective_func, self.params,  method=method,
                        nwalkers=self.nwalkers, steps=steps, pos= self.seeds)
        if verbose:
            print(self.LM.report_fit(self.fitter))

    def chain(self, burn=None):
        "return results as nparray contains walk of each walker"
        if burn is not None:
            return self.fitter.chain[burn:, :, :]
        else:
            return self.fitter.chain

    def flatchain(self, burn=None):
        "returns results as pandas table"
        if burn is not None:
            return fitter.flatchain.loc[burn:]
        else:
            return fitter.flatchain

    def get_marginal_dist(self, var, var_dx, burn = None, plot_flag= False, normalize = True):

        data = self.flatchain(burn)
        #fitter.flatchain.loc[100:][var]
        bins = np.arange(self.params[var].min,self.params[var].max+ var_dx,var_dx)

        y_hist, _ = np.histogram(fitter.flatchain.loc[100:][var], bins)
        bins_pos = (b[0:-1] + np.diff(b)/2).shape

        if normalize:
            y_hist = y_hist/var_dx/y_hist.sum()

        if plot_flag:
            import matplotlib.pyplot as plt
            plt.stairs(y_hist, bins)

        return y_hist, bins, bins_pos

par_dict = {'x': [0,Lx, 10], 'y':[0,Ly, 10] }

SM = sample_with_mcmc()
SM.set_parameters(par_dict, verbose= True)
SM.params
SM.seeds
SM.nwalkers

SM.sample()
SM.chain().shape
SM.flatchain(burn = 15000)
SM.get_marginal_dist('x', 0.1, plot_flag= True, normalize = True)

np.log(1/5.5)


def gaussian(x,x0,sigma):
  return np.exp(-np.power((x - x0)/sigma, 2.)/2.)

def simple_log_panelty(x, x0, sigma):
    return -np.power((x - x0)/sigma, 2.)/2.

# %%

#fitter.flatchain.loc[100:]

import lmfit as LM
params = LM.Parameters()

#p0= (Lx/2  *np.random.rand(1), Ly/2  *np.random.rand(1))
p0= [Lx/20  , Ly/2 ]

params.add('x', p0[0],  vary=True  , min=0, max=Lx)
params.add('y', p0[1],  vary=True  , min=0, max=Ly)

#fitting_args = (x_concat, y_concat, z_concat)



p0x = np.linspace(0,Lx, 10)
p0y = np.linspace(0, Ly,15)
nwalkers= p0x.size * p0y.size
pxx, pyy = np.meshgrid(p0x, p0y)

#pos0 = np.repeat(p0, nwalkers).reshape(len(p0), nwalkers,  )

pos0= np.vstack([pxx.flatten(), pyy.flatten() ])
#%timeit fitter = LM.minimize(objective_func, params, method='dual_annealing',max_nfev=None)
#%timeit fitter = LM.minimize(objective_func, params, method='emcee', nwalkers=nwalkers, steps=200, pos= pos0.T, workers= 3)

%timeit fitter = LM.minimize(objective_func, params, method='emcee', nwalkers=nwalkers, steps=100, pos= pos0.T)
fitter = LM.minimize(objective_func, params, method='emcee', nwalkers=nwalkers, steps=100, pos= pos0.T)

print(LM.report_fit(fitter))
#print(fitter.pretty_print())
#%timeit
#fitter = LM.minimize(objective_func, params, method='brute', workers=1 , Ns=120)

cost_2d= cost(xx, yy)

print( 'print final diff:' ,  cost_2d.min()- cost(fitter.params['x'].value, fitter.params['y'].value) )

# %%
plt.contourf(x, y, cost(xx, yy) )
plt.axis('equal')

for n in np.arange(nwalkers):
    plt.plot(fitter.chain[:,n,0], fitter.chain[:,n,1] , '-', markersize= 2, linewidth= 0.8, alpha= 0.2, color= 'white')
    plt.plot(fitter.chain[:,n,0], fitter.chain[:,n,1] , '.', markersize= 2, linewidth= 0.8, alpha= 0.2, color= 'white')
#
#
# plt.plot(fitter.params['x'].value, fitter.params['y'].value, '.r', markersize=20)
# plt.plot(p0[0], p0[1], '.b', markersize=20)
# plt.plot(pos0[0,:], pos0[1,:], '*', color='orange')

# print(LM.report_fit(fitter))
# plt.colorbar()


# %%
fitter.flatchain.shape
fitter.chain.shape
burn= 10
chain_burned= fitter.chain[burn:, :, :]
cs = chain_burned.shape
chain_flat = chain_burned.reshape(cs[0] * cs[1], 2)

chain_flat.shape
plt.hist(chain_flat[:, 1], bins= 60)
 # %%
var = 'y'
var_dx = 0.2

y_hist, _ = np.histogram(fitter.flatchain.loc[100:][var], bins)
bins_pos = (b[0:-1] + np.diff(b)/2).shape



y_hist_normed.sum() * var_dx
