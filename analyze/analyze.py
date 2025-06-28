import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from pathlib import Path
from scipy import interpolate, optimize
from analyze_PY import *
from scipy.signal import savgol_filter
from scipy.interpolate import make_interp_spline, make_smoothing_spline



def get_smoothed_Iq_per_file(folder, finfo, nskip=0):
    data = np.genfromtxt(f"{folder}/{finfo}_Iq.csv", delimiter=",")
    L = data[0, 1]
    pdType = data[1, 1]
    eta = data[2, 1]
    sigma = data[3, 1]
    # Extract q and I(q) data (starting from row 5, skipping header)
    q, Iq = data[5:, 0], data[5:, 1]

    # Smooth the data using scipy's savgol_filter
    Iq_smooth = Iq.copy()
    q_smooth1 = 3.1 + 5 * eta

    nsmooth1 = np.argmin(np.abs(q - q_smooth1))
    if nsmooth1 > 5:
        spl = make_smoothing_spline(q[:nsmooth1], Iq[:nsmooth1])  # cubic spline
        q_smooth = np.linspace(q[0], q[nsmooth1-1], nsmooth1)
        Iq_smooth[:nsmooth1] = spl(q_smooth)

    return q, Iq, Iq_smooth, L, pdType, eta, sigma

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


def find_smoothing_q(eta):
    return (3, 3.1 + 5 * eta)  # this is emperical definition, we remove a effect of simulation box form factor


def plot_Iq_prec_by_param(folder, params):
    # fig = plt.figure(figsize=(8, 6))
    fig, axs = plt.subplots(2, 2, figsize=(15, 10), sharex=True)
    axs = axs.flatten()
    params = np.asarray(params)  # shape (n, 3)
    colors = plt.cm.rainbow(np.linspace(0, 1, len(params)))
    norms = [Normalize(vmin=params[:, j].min(), vmax=params[:, j].max()) for j in range(len(params[0]))]
    params_tex = ["L", "pdType", "eta", "sigma"]
    for i, param in enumerate(params):
        L, pdType, eta, sigma = param
        finfo = f"L_{L:.0f}_pdType_{pdType:.0f}_eta_{eta:.2f}_sigma_{sigma:.2f}"
        q, Iq, Iq_smooth, L, pdType, eta, sigma = get_smoothed_Iq_per_file(folder, finfo)

        # Fit with cubic polynomial (degree=3) for extralopation
        # really not necessary turn out
        '''
        coefs = np.polyfit(q[nsmooth0:nsmooth1], np.log10(Iq[nsmooth0:nsmooth1]), deg=3)
        print(f"Polynomial coefficients for {finfo}: {coefs}")
        poly_func = np.poly1d(coefs)
        Iq_smooth[:nsmooth1] = np.power(10,poly_func(q[:nsmooth1]))
        # Extend the q range to include lower values starting from 1
        q_spacing = q[1] - q[0]  # Get the spacing between q points
        n_extra_points = int((q[0] - 1) / q_spacing)  # Calculate how many points needed to reach q=1
        q_low = np.linspace(1, q[0] - q_spacing, n_extra_points)  # Create array from 1 to just before q[0]
        # Create extended q array and corresponding Iq_smooth values
        q_extended = np.concatenate((q_low, q))
        Iq_extended = np.concatenate((10**poly_func(q_low), Iq_smooth))
        '''

        for j in range(len(param)):
            #axs[j].plot(q, Iq, "o", ms=2, mfc="None", label=f"{finfo}", color=plt.cm.rainbow(norms[j](param[j])), lw=0.5, alpha=0.7)
            axs[j].plot(q, Iq, "o", ms=2, mfc="None", label=f"{finfo}", color="gray", lw=0.5, alpha=0.7)
            axs[j].plot(q, Iq_smooth, "-", mfc="None", label=f"{finfo}", color=plt.cm.rainbow(norms[j](param[j])), lw=0.5, alpha=0.7)
            #axs[j].plot(q_extended, Iq_extended, "-", mfc="None", label=f"{finfo}", color=plt.cm.rainbow(norms[j](param[j])), lw=0.5, alpha=0.7)

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


def calc_RMSE(q, Iq, Iq_fit):
    return np.sqrt(np.mean((Iq - Iq_fit) ** 2))


def compare_Iq_by_param(folder, params, nskip=0, label=None):
    RMSEs = []

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(22, 6))

    # Store eta, sigma, and RMSE values for the scatter plot
    eta_values = []
    sigma_values = []
    RMSEs_beta = []
    RMSEs = []

    # First subplot: q vs Iq
    for i, param in enumerate(params):
        L, pdType, eta, sigma = param
        finfo = f"L_{L:.0f}_pdType_{pdType}_eta_{eta:.2f}_sigma_{sigma:.2f}"
        q, Iq, Iq_smooth, L, pdType, eta, sigma = get_smoothed_Iq_per_file(folder, finfo, nskip)
        if Iq[0] == 0:
            continue
        Iq_PY = calc_HS_PY_IQ(q, eta, sigma, pdType, beta_correction=False)
        Iq_PY_beta = calc_HS_PY_IQ(q, eta, sigma, pdType, beta_correction=True)
        ax1.plot(q, Iq_smooth, "o", mfc="None", label=f"{finfo}")
        color = ax1.lines[-1].get_color()
        RMSE = np.sqrt(np.mean((np.log10(Iq_smooth/Iq_PY)) ** 2))
        RMSE_beta = np.sqrt(np.mean((np.log10(Iq_smooth/Iq_PY_beta)) ** 2))

        # Store values for scatter plot
        eta_values.append(eta)
        sigma_values.append(sigma)
        RMSEs.append(RMSE)
        RMSEs_beta.append(RMSE_beta)

        ax1.plot(q, Iq_PY, ls="-", color=color)

    ax1.set_yscale("log")
    ax1.set_xlabel("QD")
    ax1.set_ylabel("I(QD)")
    ax1.legend()
    ax1.grid()
    ax1.set_title("Structure Factor Comparison")

    # Calculate the range for the colorbar to use for both plots
    vmin = min(min(RMSEs), min(RMSEs_beta))
    vmax = max(max(RMSEs), max(RMSEs_beta))

    # Second subplot: RMSE as scatter in eta-sigma plane (without beta correction)
    scatter1 = ax2.scatter(eta_values, sigma_values, c=RMSEs, s=[r*1000 for r in RMSEs], cmap='rainbow', alpha=0.7, edgecolors='black', vmin=vmin, vmax=vmax)
    ax2.set_xlabel("Volume Fraction (η)")
    ax2.set_ylabel("Polydispersity (σ)")
    ax2.set_title("RMSE without Beta Correction")
    ax2.grid(True)

    # Third subplot: RMSE as scatter in eta-sigma plane (with beta correction)
    scatter2 = ax3.scatter(eta_values, sigma_values, c=RMSEs_beta, s=[r*1000 for r in RMSEs_beta], cmap='rainbow', alpha=0.7, edgecolors='black', vmin=vmin, vmax=vmax)
    ax3.set_xlabel("Volume Fraction (η)")
    ax3.set_ylabel("Polydispersity (σ)")
    ax3.set_title("RMSE with Beta Correction")
    ax3.grid(True)

    # Add shared colorbar
    cbar = fig.colorbar(scatter2, ax=[ax2, ax3])
    cbar.set_label('RMSE')

    plt.tight_layout()
    plt.savefig(f"{folder}/Iq_plot_check_{label}.png")
    plt.show()
    plt.close()

    return RMSEs




def fit_Iq_by_param(folder, params, nskip=0, label=None):
    for i, param in enumerate(params):
        L, pdType, N, sigma = param
        finfo = f"L_{L:.0f}_pdType_{pdType}_N_{N:.2f}_sigma_{sigma:.2f}"
        data = np.genfromtxt(f"{folder}/{finfo}_Iq.csv", delimiter=",")
        q, Iq = data[0, 1:], data[1, 1:]
        if Iq[0] == 0:
            continue
        eta = calc_volume_fraction(0.5, N, sigma, pdType)
        # print(finfo, q, Iq)
        plt.plot(q[nskip:], Iq[nskip:], "o", mfc="None", label=f"N={N},eta={eta}, sigma={sigma:.2f}")

        fit_PY = fit_eta_sigma(q[nskip:], Iq[nskip:], pdType=pdType)
        # Create a denser q grid for PY calculation
        q_dense = np.linspace(q[nskip:].min(), q[nskip:].max(), len(q[nskip:]) * 5)

        Iq_PY_dense = calc_HS_PY_IQ(q_dense, fit_PY["eta"], fit_PY["sigma"], pdType)

        color = plt.gca().lines[-1].get_color()
        plt.plot(q_dense, Iq_PY_dense, ls="-", color=color, label=f"fit: eta={fit_PY['eta']:.2f}, sigma={fit_PY['sigma']:.2f}")
    plt.yscale("log")
    plt.xlabel("QD")
    plt.ylabel("I(QD)")
    plt.legend()
    plt.grid()
    plt.savefig(f"{folder}/Iq_plot_fit_{label}.png")
    plt.show()
    plt.close()
