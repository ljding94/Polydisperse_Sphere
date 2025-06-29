import numpy as np
import sys
import os
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import Divider, Size

# Add parent directory to path so we can import from there
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "analyze"))
from analyze import *
from VAE_model import *


def plot_gen_vs_PY_MSE(tex_lw=240.71031, ppi=72):
    print("plotting generative model vs PY model MSE comparison")

    data_folder = "../data/20250613"
    # model_path = "../data/20250615/L_18_pdType_1_gen_state_dict.pt"
    model_path = "../data/data_pack/L_18_pdType_1_gen_state_dict.pt"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gen_model = Generator(input_dim=2, latent_dim=3).to(device)
    gen_model.load_state_dict(torch.load(model_path, map_location=device))
    gen_model.eval()

    etas = np.arange(0.05, 0.451, 0.05)
    sigmas = np.arange(0.01, 0.111, 0.01)

    fig = plt.figure(figsize=(tex_lw / ppi * 1.0, tex_lw / ppi * 0.9))
    axs = fig.subplots(3, 3, sharex=True, sharey=True)
    ax11, ax12, ax13 = axs[0]
    ax21, ax22, ax23 = axs[1]
    ax31, ax32, ax33 = axs[2]

    MSEs_gen = []
    MSEs_PY = []
    MSEs_PYbeta = []
    eta_values = []
    sigma_values = []

    data_folder = "../data/20250613"
    gen_folder = "../data/data_pack"
    model_paths = ["../data/data_pack/L_18_pdType_1_gen_state_dict.pt", "../data/data_pack/L_18_pdType_2_gen_state_dict.pt"]
    pdTypes = [1, 2]

    for i in range(len(model_paths)):
        pdType = pdTypes[i]
        MSEs_gen.append([])
        MSEs_PY.append([])
        MSEs_PYbeta.append([])
        eta_values.append([])
        sigma_values.append([])
        gen_label = f"L_18_pdType_{pdTypes[i]:.0f}"
        for eta in etas:
            for sigma in sigmas:
                finfo = f"L_18_pdType_{pdType:.0f}_eta_{eta:.2f}_sigma_{sigma:.2f}"
                q, Iq, Iq_smooth, L, pdType, eta, sigma = get_smoothed_Iq_per_file(data_folder, finfo, nskip=0)
                if Iq[0] == 0:
                    continue

                # normalize params
                _, params_norm = normalize_Iq(gen_folder, gen_label, np.log10(Iq), [eta, sigma])
                log10Iq_gen, mu, logvar = gen_model(torch.tensor(params_norm, dtype=torch.float32).unsqueeze(0).to(device))
                log10Iq_gen, _ = denormalize_generated_Iq(gen_folder, gen_label, log10Iq_gen.detach(), [eta, sigma])
                log10Iq_gen = log10Iq_gen.cpu().numpy().flatten()
                Iq_PY = calc_HS_PY_IQ(q, eta, sigma, pdType, beta_correction=False)
                Iq_PY_beta = calc_HS_PY_IQ(q, eta, sigma, pdType, beta_correction=True)

                MSE_gen = np.mean(((np.log10(Iq_smooth) - log10Iq_gen) ** 2))
                MSE_PY = np.mean((np.log10(Iq_smooth / Iq_PY)) ** 2)
                MSE_PYbeta = np.mean((np.log10(Iq_smooth / Iq_PY_beta)) ** 2)

                eta_values[-1].append(eta)
                sigma_values[-1].append(sigma)
                MSEs_gen[-1].append(MSE_gen)
                MSEs_PY[-1].append(MSE_PY)
                MSEs_PYbeta[-1].append(MSE_PYbeta)

    # Calculate the range for the colorbar to use for both plots
    all_mses_gen = [mse for sublist in MSEs_gen for mse in sublist]
    all_mses_py = [mse for sublist in MSEs_PY for mse in sublist]
    all_mses_pybeta = [mse for sublist in MSEs_PYbeta for mse in sublist]
    vmin = min(min(all_mses_gen), min(all_mses_py), min(all_mses_pybeta))
    vmax = max(max(all_mses_gen), max(all_mses_py), max(all_mses_pybeta))

    for i in range(len(model_paths)):
        # First subplot: MSE as scatter in eta-sigma plane (generative model)
        sc = 10000
        scatter1 = axs[i, 0].scatter(eta_values[i], sigma_values[i], c=MSEs_gen[i], s=[r * sc for r in MSEs_gen[i]], cmap="rainbow", alpha=0.7, edgecolors="none", vmin=vmin, vmax=vmax)
        axs[i, 0].set_xlabel(r"$\eta$", fontsize=9, labelpad=0)
        axs[i, 0].set_ylabel(r"$\sigma$", fontsize=9, labelpad=0)
        axs[i, 0].tick_params(axis="both", direction="in", which="major", labelsize=7)
        axs[i, 0].xaxis.set_major_locator(plt.MultipleLocator(0.2))
        axs[i, 0].xaxis.set_minor_locator(plt.MultipleLocator(0.1))
        axs[i, 0].yaxis.set_major_locator(plt.MultipleLocator(0.04))
        axs[i, 0].yaxis.set_minor_locator(plt.MultipleLocator(0.02))

        # Second subplot: MSE as scatter in eta-sigma plane (without beta correction)
        scatter2 = axs[i, 1].scatter(eta_values[i], sigma_values[i], c=MSEs_PY[i], s=[r * sc for r in MSEs_PY[i]], cmap="rainbow", alpha=0.7, edgecolors="none", vmin=vmin, vmax=vmax)
        axs[i, 1].set_xlabel(r"$\eta$", fontsize=9, labelpad=0)
        axs[i, 1].tick_params(axis="both", direction="in", which="major", labelsize=7, labelleft=False)

        # Third subplot: MSE as scatter in eta-sigma plane (with beta correction)
        scatter3 = axs[i, 2].scatter(eta_values[i], sigma_values[i], c=MSEs_PYbeta[i], s=[r * sc for r in MSEs_PYbeta[i]], cmap="rainbow", alpha=0.7, edgecolors="none", vmin=vmin, vmax=vmax)
        axs[i, 2].set_xlabel(r"$\eta$", fontsize=9, labelpad=0)
        axs[i, 2].tick_params(axis="both", direction="in", which="major", labelsize=7, labelleft=False)

    # Add colorbar on top spanning all axes
    # Create colorbar outside the axes
    cbar_ax = fig.add_axes([0.3, 0.85, 0.4, 0.05])  # [left, bottom, width, height]
    cbar = fig.colorbar(scatter1, cax=cbar_ax, orientation="horizontal")
    cbar.set_label("MSE", fontsize=7, labelpad=0)
    cbar.ax.xaxis.set_label_position("top")
    cbar.ax.xaxis.set_ticks_position("top")
    cbar.ax.tick_params(labelsize=7, direction="in", pad=0)
    # cbar.ax.xaxis.set_major_locator(plt.MultipleLocator(0.02))

    plt.tight_layout(pad=0.1, rect=[0, 0, 1, 0.8])
    # plt.tight_layout(pad=0.1)
    plt.savefig("./figures/gen_vs_PY_MSE.png", dpi=300)
    plt.savefig("./figures/gen_vs_PY_MSE.pdf")
    plt.show()
    plt.close()


def plot_gen_vs_PY_Iq(tex_lw=240.71031, ppi=72):
    print("plotting generative model vs PY model MSE comparison")

    data_folder = "../data/20250613"
    # model_path = "../data/20250615/L_18_pdType_1_gen_state_dict.pt"
    model_path = "../data/data_pack/L_18_pdType_1_gen_state_dict.pt"
    L, pdType = 18, 1
    gen_folder = "../data/20250615"
    gen_label = f"L_{L:.0f}_pdType_{pdType:.0f}"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    eta_sigma = [(0.1, 0.05), (0.40, 0.05), (0.40, 0.10)]
    shifts = [1e0, 1e2, 1e4]
    colors = ["blue", "red", "green"]

    gen_model = Generator(input_dim=2, latent_dim=3).to(device)
    gen_model.load_state_dict(torch.load(model_path, map_location=device))
    gen_model.eval()

    fig, axes = plt.subplots(3, 2, figsize=(tex_lw / ppi * 1, tex_lw / ppi * 1.2))

    titles = ["NN", "PY", "PY w/ β"]
    for row, (method_name, beta_correction) in enumerate([("Gen Model", None), ("PY", False), ("PY + β", True)]):
        ax_comparison = axes[row, 0]
        ax_comparison.text(0.9, 0.95, titles[row], transform=ax_comparison.transAxes, fontsize=9, va="top", ha="right")
        ax_difference = axes[row, 1]
        ax_difference.text(0.9, 0.95, titles[row], transform=ax_difference.transAxes, fontsize=9, va="top", ha="right")
        for i, (eta, sigma) in enumerate(eta_sigma):
            finfo = f"L_{L:.0f}_pdType_{pdType:.0f}_eta_{eta:.2f}_sigma_{sigma:.2f}"
            q, Iq, Iq_smooth, L, pdType, eta, sigma = get_smoothed_Iq_per_file(data_folder, finfo, nskip=0)

            if method_name == "Gen Model":
                # Generate using generative model
                _, params_norm = normalize_Iq(gen_folder, gen_label, np.log10(Iq), [eta, sigma])
                log10Iq_gen, mu, logvar = gen_model(torch.tensor(params_norm, dtype=torch.float32).unsqueeze(0).to(device))
                log10Iq_pred, _ = denormalize_generated_Iq(gen_folder, gen_label, log10Iq_gen.detach(), [eta, sigma])
                Iq_pred = 10 ** log10Iq_pred.cpu().numpy().flatten()
            else:
                # Calculate PY model
                Iq_pred = calc_HS_PY_IQ(q, eta, sigma, pdType, beta_correction=beta_correction)

            # Comparison plot
            ax_comparison.plot(q[::3], Iq_smooth[::3] * shifts[i], color=colors[i], marker="o", ms=3, mfc="None", ls="none", mew=0.5, label=f"({eta},{sigma})" if row == 0 else None)
            ax_comparison.plot(q, Iq_pred * shifts[i], color=colors[i], ls="-", linewidth=1.0)

            # Difference plot (ratio)
            diff = np.log10(Iq_smooth / Iq_pred)
            ax_difference.plot(q, diff, color=colors[i], linestyle="-", linewidth=1.0)

        # Format comparison subplot
        ax_comparison.set_yscale("log")
        ax_comparison.tick_params(axis="both", direction="in", which="major", labelsize=7, labelleft=True, labelbottom=(row == 2), pad=0.5)
        ax_comparison.set_ylabel(r"$I(Q)$", fontsize=9, labelpad=0)
        ax_comparison.xaxis.set_major_locator(plt.MultipleLocator(4))
        ax_comparison.xaxis.set_minor_locator(plt.MultipleLocator(2))

        ax_difference.axhline(y=0, color="black", linestyle="--", alpha=0.5, linewidth=0.8)
        ax_difference.tick_params(axis="both", direction="in", which="major", labelsize=7, labelleft=True, labelbottom=(row == 2), pad=0.5)
        ax_difference.xaxis.set_major_locator(plt.MultipleLocator(4))
        ax_difference.xaxis.set_minor_locator(plt.MultipleLocator(2))
        ax_difference.yaxis.set_major_locator(plt.MultipleLocator(0.1))
        ax_difference.yaxis.set_minor_locator(plt.MultipleLocator(0.05))
        ax_difference.set_ylabel(r"$\Delta\log_{10}I(Q)$", fontsize=9, labelpad=0)
        ax_difference.set_ylim(-0.15, 0.15)

        if row == 0:
            ax_difference.legend(fontsize=7, loc="upper right", ncol=1, columnspacing=0.5, handlelength=1, handletextpad=0.2, frameon=False)
        if row == 2:
            ax_comparison.set_xlabel(r"$Q$", fontsize=9, labelpad=0)
            ax_difference.set_xlabel(r"$Q$", fontsize=9, labelpad=0)

    # add annotations
    axs = axes.flatten()
    annos = [r"$(a)$", r"$(b)$", r"$(c)$", r"$(d)$", r"$(e)$", r"$(f)$"]
    for i, ax in enumerate(axs):
        ax.text(0.8, 0.075, annos[i], fontsize=9, transform=ax.transAxes)

    plt.tight_layout(pad=0.1)
    plt.savefig("./figures/gen_vs_PY_Iq_comparison.png", dpi=300)
    plt.savefig("./figures/gen_vs_PY_Iq_comparison.pdf")
    plt.show()
    plt.close()
