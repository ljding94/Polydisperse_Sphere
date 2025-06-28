import numpy as np
import sys
import os

import matplotlib.pyplot as plt

# Add parent directory to path so we can import from there
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "analyze"))
from analyze import *
from VAE_model import *


def plot_gen_vs_PY_RMSE(tex_lw=240.71031, ppi=72):
    print("plotting generative model vs PY model RMSE comparison")

    data_folder = "../data/20250613"
    #model_path = "../data/20250615/L_18_pdType_1_gen_state_dict.pt"
    model_path = "../data/data_pack/L_18_pdType_1_gen_state_dict.pt"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gen_model = Generator(input_dim=2, latent_dim=3).to(device)
    gen_model.load_state_dict(torch.load(model_path, map_location=device))
    gen_model.eval()

    etas = np.arange(0.05, 0.451, 0.05)
    sigmas = np.arange(0.01, 0.111, 0.01)

    fig = plt.figure(figsize=(tex_lw / ppi * 1.0, tex_lw / ppi * 0.4))
    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132, sharey=ax1, sharex=ax1)
    ax3 = fig.add_subplot(133, sharey=ax1, sharex=ax1)

    RMSEs_gen = []
    RMSEs_PY = []
    RMSEs_PYbeta = []

    L, pdType = 18, 1
    gen_folder = "../data/20250615"
    gen_label = f"L_{L:.0f}_pdType_{pdType:.0f}"
    eta_values = []
    sigma_values = []
    for eta in etas:
        for sigma in sigmas:
            finfo = f"L_{L:.0f}_pdType_{pdType:.0f}_eta_{eta:.2f}_sigma_{sigma:.2f}"
            q, Iq, Iq_smooth, L, pdType, eta, sigma = get_smoothed_Iq_per_file(data_folder, finfo, nskip=0)
            if Iq[0] == 0:
                continue

            log10Iq_gen, mu, logvar = gen_model(torch.tensor([eta, sigma], dtype=torch.float32).unsqueeze(0).to(device))
            log10Iq_gen = denormalize_generated_Iq(gen_folder, gen_label, log10Iq_gen.detach())
            log10Iq_gen = log10Iq_gen.cpu().numpy().flatten()
            Iq_PY = calc_HS_PY_IQ(q, eta, sigma, pdType, beta_correction=False)
            Iq_PY_beta = calc_HS_PY_IQ(q, eta, sigma, pdType, beta_correction=True)

            RMSE_gen = np.sqrt(np.mean(((np.log10(Iq_smooth) - log10Iq_gen) ** 2)))
            RMSE_PY = np.sqrt(np.mean((np.log10(Iq_smooth / Iq_PY)) ** 2))
            RMSE_PYbeta = np.sqrt(np.mean((np.log10(Iq_smooth / Iq_PY_beta)) ** 2))

            eta_values.append(eta)
            sigma_values.append(sigma)
            RMSEs_gen.append(RMSE_gen)
            RMSEs_PY.append(RMSE_PY)
            RMSEs_PYbeta.append(RMSE_PYbeta)

    # Calculate the range for the colorbar to use for both plots
    vmin = min(min(RMSEs_gen), min(RMSEs_PY), min(RMSEs_PYbeta))
    vmax = max(max(RMSEs_gen), max(RMSEs_PY), max(RMSEs_PYbeta))

    # First subplot: RMSE as scatter in eta-sigma plane (generative model)
    sc = 200
    scatter1 = ax1.scatter(eta_values, sigma_values, c=RMSEs_gen, s=[r * sc for r in RMSEs_gen], cmap="rainbow", alpha=0.7, edgecolors="black", vmin=vmin, vmax=vmax)
    ax1.set_xlabel(r"$\eta$", fontsize=9)
    ax1.set_ylabel(r"$\sigma$", fontsize=9)
    ax1.grid(True)
    ax1.tick_params(axis='both', direction='in', which='major', labelsize=7)

    # Second subplot: RMSE as scatter in eta-sigma plane (without beta correction)
    scatter2 = ax2.scatter(eta_values, sigma_values, c=RMSEs_PY, s=[r * sc for r in RMSEs_PY], cmap="rainbow", alpha=0.7, edgecolors="black", vmin=vmin, vmax=vmax)
    ax2.set_xlabel(r"$\eta$", fontsize=9)
    ax2.grid(True)
    ax2.tick_params(axis='both', direction='in', which='major', labelsize=7, labelleft=False)

    # Third subplot: RMSE as scatter in eta-sigma plane (with beta correction)
    scatter3 = ax3.scatter(eta_values, sigma_values, c=RMSEs_PYbeta, s=[r * sc for r in RMSEs_PYbeta], cmap="rainbow", alpha=0.7, edgecolors="black", vmin=vmin, vmax=vmax)
    ax3.set_xlabel(r"$\eta$", fontsize=9)
    ax3.grid(True)
    ax3.tick_params(axis='both', direction='in', which='major', labelsize=7, labelleft=False)

    # Add colorbar on the right side of ax3
    cbar = fig.colorbar(scatter3, ax=ax3, shrink=0.6)
    cbar.set_label("RMSE", fontsize=7)
    cbar.ax.tick_params(labelsize=7, top=True, labeltop=True, bottom=False, labelbottom=False)

    plt.tight_layout(pad=0.1)
    plt.savefig("./figures/gen_vs_PY_RMSE.png", dpi=300)
    plt.savefig("./figures/gen_vs_PY_RMSE.pdf")
    plt.show()
    plt.close()
