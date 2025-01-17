
import numpy as np
from numpy import array as npa
from numpy import imag, sqrt, real
import matplotlib.pyplot as pl 
from matplotlib import colors 
from scipy.fft import fftshift, fftfreq, fft2, fft
from scipy.constants import pi
from itertools import product

from load import * 

class SpecData(SimData):
    def __init__(self, filename, R=np.eye(3), eta=0, bounds=None, legacy=False, groupname="spectroscopy"):
        super().__init__(filename)
        self.load_group(groupname)
        # load raw data and rotate it 
        tau = self.data["taus"]
        if legacy:
            t = self.data["ts"]
            self.data["dt"] = t[1] - t[0]
        else: 
            t = tau.copy()
            self.data["ts"] = t
        keys = ["MA", "MB", "MAB"]
        data_rotated = [ np.einsum("ijk,kl", npa(self.data[key]), R) for key in keys ] # rotate data to proper frame 

        if bounds != None:
            t_, tau_ = bounds 
            tbound = (t > t_[0]) & (t < t_[1])
            taubound = (tau > tau_[0]) & (tau < tau_[1])
            self.data["ts"] = t[tbound]
            self.data["taus"] = tau[taubound]
            data_rotated = [dat[taubound,:,:][:,tbound,:] for dat in data_rotated]

        filter = np.exp(-eta * np.outer(self.data["ts"], self.data["taus"]))
        for i, value in enumerate(data_rotated):
            self.data[keys[i]] = (value.T * filter ).T

        self.data["MNL"] = self.data["MAB"] -  self.data["MA"] - self.data["MB"] 
        self.keys = ("MA", "MB", "MAB", "MNL")

    def get_FT_data(self):
        dt = self.data["dt"]
        Nf = self.data["ts"].size
        Nnu = self.data["taus"].size
        f = fftshift(fftfreq(Nf, d=dt))
        nu = fftshift(fftfreq(Nnu, d=dt))
        
        self.data["f_t"] = f
        self.data["f_tau"] = nu 

        for key,comp in product(self.keys, range(3)):
            self.data["{}_FT_{}".format(key, comp)] = fftshift(fft2(self.data[key][:,:,comp]))

        # for comp in range(3):
        #     self.data["MNL_FT_{}".format(comp)] = self.data["MAB_FT_{}".format(comp)] - self.data["MB_FT_{}".format(comp)] - self.data["MA_FT_{}".format(comp)]  

def plot_linear_response(simdata, maskorigin=True, label="xyz", **kwargs):
    # ts = simdata.data["ts"]
    f_t = simdata.data["f_t"]
    fig, axs = pl.subplots(1, 3, figsize=(12, 4))

    for i, comp in enumerate(range(3)):
        MB = simdata.data["MB"][0, :, comp]
        MBw = fftshift(fft(MB))
        intensity = sqrt(real(MBw)**2 + imag(MBw)**2)
        if maskorigin:
            intensity[f_t == 0.0] = np.NaN
        axs[i].plot(f_t * 2 * pi, np.log(intensity), color="black", **kwargs)
        axs[i].set_ylabel(r"log$|M(\omega)|$ (arb.)")
        axs[i].set_xlabel(r"$\omega/|K|$")
        axs[i].set_title(r"$M^{}$".format(label[comp]))

    return fig, axs 


def plot_2d_spec_time(simdata, comp, **kwargs):
    t = simdata.data["ts"]
    tau = simdata.data["taus"]
    label = ["x", "y", "z"]
    data = [ simdata.data[key] for key in simdata.keys]

    ts, taus = np.meshgrid(t, tau)
    images = []
    titles = [r"$M_A^{}$".format(label[comp]), r"$M_B^{}$".format(label[comp]), 
              r"$M_{{AB}}^{}$".format(label[comp]), r"$M_{{NL}}^{}$".format(label[comp])]
    fig, axs = pl.subplots(2,2, sharex=True, sharey=True, figsize=(8, 8))
    for ind, ax in enumerate(axs.flatten()):
        im = ax.pcolormesh(ts, taus, data[ind][:,:,comp], **kwargs) #/data[ind][:,:,comp].max()
        images.append(im)
        ax.set_title(titles[ind])
        if ind in (0, 2):
            ax.set_ylabel(r"$\tau$ (1/|J|)")
        if ind in (2, 3):
            ax.set_xlabel("$t$ (1/|J|)")
        fig.colorbar(im, ax=ax, orientation='vertical', fraction=0.05)

    fig.tight_layout()
    
    return fig, axs 


def plot_2d_spec_freq(simdata, comp,  plotreal=True, maskorigin=False, **kwargs):
    keys = []
    for i, key in enumerate(simdata.keys):
        newkey = key + "_FT_{}".format(comp)
        keys.append(newkey)

    data = [ simdata.data[key] for key in keys]

    label=["x", "y","z"]
    titles = [r"$M_A^{}$".format(label[comp]), r"$M_B^{}$".format(label[comp]), 
              r"$M_{{AB}}^{}$".format(label[comp]), r"$M_{{NL}}^{}$".format(label[comp])]
    
    fig, axs = pl.subplots(2,2, sharex=True, sharey=True, figsize=(8, 8))
    xf = 2*pi*simdata.data["f_t"]
    yf = 2*pi*simdata.data["f_tau"]
    xlabel = r"$\omega_t/|K|$"
    ylabel = r"$\omega_\tau/|K|$"
    
    for ind, ax in enumerate(axs.flatten()):
        intensity = real(data[ind]) if plotreal else imag(data[ind])
        if maskorigin:
            intensity[xf == 0, yf == 0] = 0.0

        im = ax.pcolormesh(xf, yf, intensity, **kwargs)
        ax.set_title(titles[ind])
        if ind in (0, 2):
            ax.set_ylabel(ylabel)
        if ind in (2, 3):
            ax.set_xlabel(xlabel)

        fig.colorbar(im, ax=ax, orientation='vertical', fraction=0.05)
        
    fig.tight_layout()
    
    return fig, axs 

def plot_spec_NL(simdata,plotreal=True, maskorigin=False,label="xyz", comp="all", **kwargs):
    # hbar = 6.58211957e-13
    data = [simdata.data["MNL_FT_{}".format(comp)] for comp in range(3)]
    f_t = simdata.data["f_t"]
    f_tau = simdata.data["f_tau"]

    extent=[2*pi*f_t.min(), 2*pi*f_t.max(), 2*pi*f_tau.min(), 2*pi*f_tau.max()]
    

    if comp == "all":
        fig, axs = pl.subplots(1, 3, figsize=(12, 6), sharey=True)
        for comp, ax in enumerate(axs):
            spec = data[comp]
            # ax.set_title(r"$M^{}_{{NL}}$".format(label[comp]))

            # intensity = real(spec) if plotreal else imag(spec)
            intensity = sqrt(real(spec)**2 + imag(spec)**2)

            if maskorigin:
                intensity[f_tau== 0, f_t == 0] = 0.0
            
            im = ax.imshow(intensity/intensity.max(), extent=extent,**kwargs)

            ax.set_aspect("equal")
            cbar = fig.colorbar(im, ax=ax, orientation='horizontal', fraction=0.025)
            cbar.ax.set_xlabel(r"$|M_{{NL}}^{}(\omega_t,\omega_{{\tau}})|$ (arb.)".format(label[comp]))
            # fig.colorbar(im, ax=ax, orientation='vertical', fraction=0.05)
        fig.tight_layout(pad=2.0)
        
        return fig, axs
    elif comp in range(3):
        fig, ax = pl.subplots()
        spec = data[comp]
        # ax.set_title(r"$M^{}_{{NL}}$".format(label[comp]))

        # intensity = real(spec) if plotreal else imag(spec)
        intensity = sqrt(real(spec)**2 + imag(spec)**2)
        if maskorigin:
            intensity[f_t == 0, f_tau== 0] = 0.0
        
        im = ax.imshow(intensity/intensity.max(), extent=extent,**kwargs)

        ax.set_aspect("equal")
        cbar = fig.colorbar(im, ax=ax, orientation='horizontal', fraction=0.025)
        cbar.ax.set_xlabel(r"$|M_{{NL}}^{}(\omega_t,\omega_{{\tau}})|$ (arb.)".format(label[comp]))
        fig.tight_layout()
        return fig, ax 
    else:
        raise ValueError("comp must be in [0,1,2] or 'all'")