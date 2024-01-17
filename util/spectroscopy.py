
import numpy as np
from numpy import array as npa
from numpy import imag, sqrt, real
import matplotlib.pyplot as pl 
from matplotlib import colors 
from scipy.fft import fftshift, fftfreq, fft2
from scipy.constants import pi

from load import * 

def get_2D_spectroscopy_data(filename, R=np.eye(3)):
    data = SimData(filename)
    data.load_group("spectroscopy")
    keys = ["MA", "MB", "MAB", "MNL"]
    data_rotated = [ np.einsum("ijk,kl", npa(data.data[key]), R) for key in keys[:-1] ] # rotate data to proper frame 
    for i, value in enumerate(data_rotated):
        data.data[keys[i]] = value 

    data.data["MNL"] = data.data["MAB"] -  data.data["MA"] - data.data["MB"] 
    return data 

def plot_2d_spec_time(simdata, comp, bounds=None, **kwargs):
    t = simdata.data["ts"]
    tau = simdata.data["taus"]
    keys = ["MA", "MB", "MAB", "MNL"]
    data = [ simdata.data[key] for key in keys]
    
    if bounds != None:
        t_, tau_ = bounds 
        tbound = (t > t_[0]) & (t < t_[1])
        taubound = (tau > tau_[0]) & (tau < tau_[1])
        t = t[tbound]
        tau = tau[taubound]
        data = [dat[taubound,:,:][:,tbound,:] for dat in data]

    ts, taus = np.meshgrid(t, tau)
    images = []
    titles = [r"$M_A$", r"$M_B$", r"$M_{AB}$", r"$M_{NL}$"]
    fig, axs = pl.subplots(2,2, sharex=True, sharey=True, figsize=(8, 8))
    for ind, ax in enumerate(axs.flatten()):
        im = ax.pcolormesh(ts, taus, data[ind][:,:,comp]/data[ind][:,:,comp].max(), **kwargs)
        images.append(im)
        ax.set_title(titles[ind])
        if ind in (0, 2):
            ax.set_ylabel(r"$\tau$ (1/|J|)")
        if ind in (2, 3):
            ax.set_xlabel("$t$ (1/|J|)")

    norm = colors.Normalize(vmin=-1, vmax=1)
    for im in images:
        im.set_norm(norm)

    fig.colorbar(images[0], ax=axs, orientation='vertical', fraction=0.05)
    
    return fig, axs 


def plot_2d_spec_freq(simdata, comp, bounds = None, **kwargs):
    t = simdata.data["ts"]
    tau = simdata.data["taus"]
    keys = ["MA", "MB", "MAB", "MNL"]
    data = [ simdata.data[key] for key in keys]
    
    if bounds != None:
        t_, tau_ = bounds 
        tbound = (t > t_[0]) & (t < t_[1])
        taubound = (tau > tau_[0]) & (tau < tau_[1])
        t = t[tbound]
        tau = tau[taubound]
        data = [dat[taubound,:,:][:,tbound,:] for dat in data]
        
    Nf = t.size
    Nnu = tau.size
    f = fftshift(fftfreq(Nf, d=(t[1]-t[0])))
    nu = fftshift(fftfreq(Nnu, d=(tau[1]-tau[0])))
    titles = [r"$M_A$", r"$M_B$", r"$M_{AB}$", r"$M_{NL}$"]
    fs, nus = np.meshgrid(f, nu)
    fig, axs = pl.subplots(2,2, sharex=True, sharey=True, figsize=(8, 8))
    images = []
    ints = []

    xf = 2*pi*fs
    yf = 2*pi*nus
    xlabel = r"$\omega_f/|K|$"
    ylabel = r"$\omega_\nu/|K|$"
    
    for ind, ax in enumerate(axs.flatten()):
        spec = fftshift(fft2(data[ind][:,:,comp]))
        intensity = sqrt(real(spec)**2 + imag(spec)**2)
        ints.append(intensity)
        
        im = ax.pcolormesh(xf, yf, intensity/intensity.max(), **kwargs)
        images.append(im)
        ax.set_title(titles[ind])
        if ind in (0, 2):
            ax.set_ylabel(ylabel)
        if ind in (2, 3):
            ax.set_xlabel(xlabel)
    fig.colorbar(images[-1], ax=axs, orientation='vertical', fraction=0.05)
    
    return fig, axs 

def plot_spec_NL(simdata, comp, bounds=None, maskorigin=False,**kwargs):
    t = simdata.data["ts"]
    tau = simdata.data["taus"]
    keys = ["MA", "MB", "MAB", "MNL"]
    data = [ simdata.data[key] for key in keys]
    label = ["x", "y", "z"]
    
    if bounds != None:
        t_, tau_ = bounds 
        tbound = (t > t_[0]) & (t < t_[1])
        taubound = (tau > tau_[0]) & (tau < tau_[1])
        t = t[tbound]
        tau = tau[taubound]
        data = [dat[taubound,:,:][:,tbound,:] for dat in data]
        
    Nf = t.size
    Nnu = tau.size
    f = fftshift(fftfreq(Nf, d=(t[1]-t[0])))
    nu = fftshift(fftfreq(Nnu, d=(tau[1]-tau[0])))
    fs, nus = np.meshgrid(f, nu)
    fig, ax = pl.subplots(figsize=(5, 5))
    if comp == "norm":
        spec = fftshift(fft2(data[3].sum(axis=-1)))
        ax.set_title(r"$|M_{{NL}}|$")
    else:
        spec = fftshift(fft2(data[3][:,:,comp]))   
        ax.set_title(r"$M^{}_{{NL}}$".format(label[comp]))
        
    intensity = sqrt(real(spec)**2 + imag(spec)**2)
    if maskorigin:
        intensity[nu == 0, f == 0] = 0.0
    
    im = ax.imshow(intensity/intensity.max(), extent=[2*pi*fs.min(), 2*pi*fs.max(), 2*pi*nus.min(), 2*pi*nus.max()],**kwargs)
    ax.set_ylabel(r"$\omega_\nu/|K|$")
    ax.set_xlabel(r"$\omega_f/|K|$")
    ax.set_aspect("equal")
    fig.colorbar(im, ax=ax, orientation='vertical', fraction=0.05)
    return fig, ax