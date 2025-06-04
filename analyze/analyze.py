import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from pathlib import Path
from scipy import interpolate, optimize


def plot_Iq_prec(folder, params, nskip=0):
    plt.figure(figsize=(8, 6))
    for i, param in enumerate(params):
        pdType, N, sigma = param
        finfo = f"pdType_{pdType}_N_{N:.2f}_sigma_{sigma:.2f}"
        data = np.genfromtxt(f"{folder}/{finfo}_Iq.csv", delimiter=",")
        q, Iq = data[0, 1:], data[1, 1:]
        plt.plot(q[nskip:], Iq[nskip:], label=f"{finfo}")
    plt.yscale("log")
    plt.xlabel("QD")
    plt.ylabel("I(QD)")
    plt.legend()
    plt.grid()
    plt.savefig(f"{folder}/Iq_plot.png")
    plt.show()
    plt.close()


def calc_box_size(N):
    nHS = 16 * 16 * 16 * 4
    box_size = (nHS / N) ** (1 / 3)
    return box_size / 2


def calc_sphere_Pq(R, Q):
    qR = Q * R
    FQ = 3 * (np.sin(qR) - qR * np.cos(qR)) / (qR) ** 3
    PQ = FQ**2
    return PQ


def plot_Iq_prec_by_param(folder, params, nskip=0, spine=False):
    # fig = plt.figure(figsize=(8, 6))
    fig, axs = plt.subplots(len(params[0]), 1, figsize=(8, 4 * len(params[0])), sharex=True)
    params = np.asarray(params)  # shape (n, 3)
    colors = plt.cm.rainbow(np.linspace(0, 1, len(params)))
    norms = [Normalize(vmin=params[:, j].min(), vmax=params[:, j].max()) for j in range(len(params[0]))]
    params_tex = ["L", "pdType", "N", "sigma"]
    for i, param in enumerate(params):
        L, pdType, N, sigma = param
        # if N > 0.4:
        # continue
        finfo = f"L_{L:.0f}_pdType_{pdType:.0f}_N_{N:.2f}_sigma_{sigma:.2f}"
        data = np.genfromtxt(f"{folder}/{finfo}_Iq.csv", delimiter=",")
        q, Iq = data[0, 1:], data[1, 1:]
        print(finfo, q, Iq)
        for j in range(len(param)):
            if spine:
                axs[j].plot(q[nskip::5], Iq[nskip::5], "o", mfc="None", label=f"{finfo}", color=plt.cm.rainbow(norms[j](param[j])), lw=0.5, alpha=0.7)
                f_interp = interpolate.interp1d(q[nskip:], Iq[nskip:], kind="cubic", fill_value="extrapolate")
                Iq_interp = f_interp(q)
                axs[j].plot(q, Iq_interp, color=plt.cm.rainbow(norms[j](param[j])), lw=1, ls="-")
            else:
                axs[j].plot(q[nskip:], Iq[nskip:], "-", label=f"{finfo}", color=plt.cm.rainbow(norms[j](param[j])), lw=0.5, alpha=0.7)

    for j in range(len(param)):
        axs[j].set_yscale("log")
        axs[j].set_ylabel("I(QD)")
        axs[j].set_xlabel("QD")
        # axs[j].legend()
        axs[j].grid()
        sm = plt.cm.ScalarMappable(cmap="rainbow", norm=norms[j])
        cbar = fig.colorbar(sm, ax=axs[j])
        cbar.set_label(params_tex[j])

    unique_N = np.unique(params[:, 1])
    for i in range(len(unique_N)):
        box_size = calc_box_size(unique_N[i])
        PQ = calc_sphere_Pq(box_size * 0.5, q)
        axs[1].plot(q, PQ, label=f"box_size={box_size:.2f}", color=plt.cm.rainbow(norms[1](unique_N[i])), lw=1, ls="--")

        # PQ = calc_sphere_Pq(box_size*0.5*1.25, q)
        # axs[1].plot(q, PQ, label=f"box_size={box_size:.2f}", color=plt.cm.rainbow(norms[1](unique_N[i])), lw=1, ls="-")

    plt.tight_layout()
    plt.savefig(f"{folder}/Iq_plot_by_param.png")
    plt.show()
    plt.close()
