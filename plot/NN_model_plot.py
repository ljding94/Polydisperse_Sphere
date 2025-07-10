import numpy as np
import sys
import os
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import Divider, Size

# Add parent directory to path so we can import from there
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "analyze"))
from analyze import *
from VAE_model import *
import torch
from matplotlib.colors import LogNorm


def plot_parm_in_latent_space(tex_lw=240.71031, ppi=72):
    print("plotting parameter distribution in latent space")
    folder = "../data/data_pack"

    label = "L_18_pdType_1"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "../data/data_pack/L_18_pdType_1_inf_state_dict.pt"
    inf_model = Inferrer().to(device)
    inf_model.load_state_dict(torch.load(model_path, map_location=device))
    inf_model.eval()

    test_loader, _ = create_dataloader(folder, label, "test", batch_size=32, shuffle=False)

    # get alllatent variables using the inferrer
    all_latent_mu = []
    all_latent_logvar = []
    all_params = []
    with torch.no_grad():
        for log10Iq, _, p in test_loader:
            log10Iq, p = log10Iq.to(device), p.to(device)
            # Get predictions from inferrer
            pred_avg, mu, logvar = inf_model(log10Iq)
            _, param_denorm = denormalize_generated_Iq(folder, label, log10Iq, p)
            all_latent_mu.append(mu.cpu().numpy())
            all_latent_logvar.append(logvar.cpu().numpy())
            all_params.append(param_denorm.cpu().numpy())

    fig = plt.figure(figsize=(tex_lw / ppi * 1, tex_lw / ppi * 0.8))
    ax1 = fig.add_subplot(221, projection="3d")
    ax2 = fig.add_subplot(222, projection="3d")
    ax3 = fig.add_subplot(223, projection="3d")
    ax4 = fig.add_subplot(224, projection="3d")

    ax1.set_box_aspect([1, 1, 1.2])
    ax2.set_box_aspect([1, 1, 1.2])
    ax3.set_box_aspect([1, 1, 1.2])
    ax4.set_box_aspect([1, 1, 1.2])

    all_latent_mu = np.concatenate(all_latent_mu, axis=0)
    all_latent_logvar = np.concatenate(all_latent_logvar, axis=0)
    all_params = np.concatenate(all_params, axis=0)
    sc1 = ax1.scatter(all_latent_mu[:, 0], all_latent_mu[:, 1], all_latent_mu[:, 2], s=1, c=all_params[:, 0], cmap="rainbow", rasterized=True)
    cbar1 = plt.colorbar(sc1, ax=ax1, shrink=0.4, pad=-0.05)
    cbar1.ax.set_title(r"$\eta$", fontsize=9, pad=0)
    cbar1.ax.tick_params(labelsize=7, direction="in")

    sc2 = ax2.scatter(all_latent_mu[:, 0], all_latent_mu[:, 1], all_latent_mu[:, 2], s=1, c=all_params[:, 1], cmap="rainbow", rasterized=True)
    cbar2 = plt.colorbar(sc2, ax=ax2, shrink=0.4, pad=-0.05)
    cbar2.ax.set_title(r"$\sigma$", fontsize=9, pad=0)
    cbar2.ax.tick_params(labelsize=7, direction="in")

    sc3 = ax3.scatter(all_latent_logvar[:, 0], all_latent_logvar[:, 1], all_latent_logvar[:, 2], s=1, c=all_params[:, 0], cmap="rainbow", rasterized=True)
    cbar3 = plt.colorbar(sc3, ax=ax3, shrink=0.4,  pad=-0.05)
    cbar3.ax.set_title(r"$\eta$", fontsize=9, pad=0)
    cbar3.ax.tick_params(labelsize=7, direction="in")

    sc4 = ax4.scatter(all_latent_logvar[:, 0], all_latent_logvar[:, 1], all_latent_logvar[:, 2], s=1, c=all_params[:, 1], cmap="rainbow", rasterized=True)
    cbar4 = plt.colorbar(sc4, ax=ax4, shrink=0.4,  pad=-0.05)
    cbar4.ax.set_title(r"$\sigma$", fontsize=9, pad=0)
    cbar4.ax.tick_params(labelsize=7, direction="in")

    for ax in [ax1, ax2]:
        ax.set_xlabel(r"$\mu_0$", fontsize=9, labelpad=-10)
        ax.set_ylabel(r"$\mu_1$", fontsize=9, labelpad=-10)
        ax.set_zlabel(r"$\mu_2$", fontsize=9, labelpad=-7)
        ax.tick_params(axis="both", direction="in", labelsize=7, labelleft=True, labelbottom=True, pad=-6)
        ax.tick_params(axis="z", direction="in", labelsize=7, pad=-3)
        ax.view_init(elev=-170, azim=-40)
    for ax in [ax3, ax4]:
        ax.set_xlabel(r"$\log(s^2_0)$", fontsize=9, labelpad=-10)
        ax.set_ylabel(r"$\log(s^2_1)$", fontsize=9, labelpad=-10)
        ax.set_zlabel(r"$\log(s^2_2)$", fontsize=9, labelpad=-7)
        ax.tick_params(axis="both", direction="in", labelsize=7, labelleft=True, labelbottom=True, pad=-6)
        ax.tick_params(axis="z", direction="in", labelsize=7, pad=-3)
        ax.view_init(elev=-170, azim=-60)

    # add annotations
    annos = [r"$(a)$", r"$(b)$", r"$(c)$", r"$(d)$"]
    for ax, anno in zip([ax1, ax2, ax3, ax4], annos):
        ax.text2D(0.8, 0.8, anno, fontsize=9, transform=ax.transAxes)

    plt.tight_layout(pad=0.5)
    plt.subplots_adjust(left=0.02, bottom=0.07, right=0.98)
    plt.savefig("./figures/parm_in_latent_space.png", dpi=300)
    plt.savefig("./figures/parm_in_latent_space.pdf", dpi=500)
    plt.show()


def plot_gen_vs_PY_Iq(tex_lw=240.71031, ppi=72):
    print("plotting generative model vs PY model MSE comparison")

    # data_folder = "../data/20250613"
    data_folder = "../data/20250701"
    # model_path = "../data/20250615/L_18_pdType_1_gen_state_dict.pt"
    model_path = "../data/data_pack/L_18_pdType_1_gen_state_dict.pt"
    L, pdType = 18, 1
    # gen_folder = "../data/20250615"
    gen_folder = "../data/data_pack"
    gen_label = f"L_{L:.0f}_pdType_{pdType:.0f}"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    eta_sigma = [(0.1, 0.10), (0.50, 0.10), (0.50, 0.30)]
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
            ax_comparison.plot(q[::3], Iq_smooth[::3] * shifts[i], color=colors[i], marker="o", ms=3, mec="black", mfc="None", ls="none", mew=0.5, label=f"({eta},{sigma})" if row == 0 else None)
            ax_comparison.plot(q, Iq_pred * shifts[i], color=colors[i], ls="-", linewidth=1.0)

            # Difference plot (ratio)
            diff = np.log10(Iq_smooth / Iq_pred)
            ax_difference.plot(q, diff, color=colors[i], linestyle="-", linewidth=1.0, label=f"({eta:.1f},{sigma:.2f})")

        # Format comparison subplot
        ax_comparison.set_yscale("log")
        ax_comparison.tick_params(axis="both", direction="in", which="major", labelsize=7, labelleft=True, labelbottom=(row == 2), pad=0.5)
        ax_comparison.set_ylabel(r"$I(Q)$", fontsize=9, labelpad=0)
        ax_comparison.xaxis.set_major_locator(plt.MultipleLocator(4))
        ax_comparison.xaxis.set_minor_locator(plt.MultipleLocator(2))
        ax_comparison.set_ylim(
            1e-4,
        )

        ax_difference.axhline(y=0, color="black", linestyle="--", alpha=0.5, linewidth=0.8)
        ax_difference.tick_params(axis="both", direction="in", which="major", labelsize=7, labelleft=True, labelbottom=(row == 2), pad=0.5)
        ax_difference.xaxis.set_major_locator(plt.MultipleLocator(4))
        ax_difference.xaxis.set_minor_locator(plt.MultipleLocator(2))
        ax_difference.yaxis.set_major_locator(plt.MultipleLocator(0.1))
        ax_difference.yaxis.set_minor_locator(plt.MultipleLocator(0.05))
        ax_difference.set_ylabel(r"$\Delta\log_{10}I(Q)$", fontsize=9, labelpad=0)
        ax_difference.set_ylim(-0.2, 0.30)

        if row == 0:
            ax_difference.legend(title=r"$(\eta,\sigma)$", fontsize=7, loc="upper left", ncol=1, columnspacing=0.5, labelspacing=0.1, handlelength=0.5, handletextpad=0.2, frameon=False)
            # ax_difference.legend(fontsize=7, loc="upper right", ncol=1, columnspacing=0.5, handlelength=1, handletextpad=0.2, frameon=False)
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


def plot_gen_vs_PY_MSE_all(tex_lw=240.71031, ppi=72):
    print("plotting generative model vs PY model MSE comparison")

    data_folder = "../data/20250613"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    etas = np.arange(0.05, 0.451, 0.05)
    sigmas = np.arange(0.01, 0.101, 0.01)

    eta_ranges = [np.arange(0.05, 0.451, 0.05), np.arange(0.05, 0.451, 0.05), np.arange(0.05, 0.451, 0.05)]
    sigma_ranges = [np.arange(0.01, 0.131, 0.01), np.arange(0.01, 0.111, 0.01), np.arange(0.01, 0.111, 0.01)]

    fig = plt.figure(figsize=(tex_lw / ppi * 1.0, tex_lw / ppi * 1.2))
    axs = fig.subplots(3, 3, sharex=True)
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
    model_paths = ["../data/data_pack/L_18_pdType_1_gen_state_dict.pt", "../data/data_pack/L_18_pdType_2_gen_state_dict.pt", "../data/data_pack/L_18_pdType_2_gen_state_dict.pt"]
    # model_paths = ["../data/data_pack/L_18_pdType_2_gen_state_dict.pt"]
    pdTypes = [1, 2, 2]

    for i in range(len(model_paths)):
        gen_model = Generator(input_dim=2, latent_dim=3).to(device)
        gen_model.load_state_dict(torch.load(model_paths[i], map_location=device))
        gen_model.eval()
        pdType = pdTypes[i]
        MSEs_gen.append([])
        MSEs_PY.append([])
        MSEs_PYbeta.append([])
        eta_values.append([])
        sigma_values.append([])
        gen_label = f"L_18_pdType_{pdTypes[i]:.0f}"
        # for eta in etas:
        # for sigma in sigmas:
        for eta in eta_ranges[i]:
            for sigma in sigma_ranges[i]:
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

    smin, smax = 0.1, 100
    for i in range(len(model_paths)):
        # First subplot: MSE as scatter in eta-sigma plane (generative model)
        # sc = 5000
        scatter1 = axs[i, 0].scatter(
            eta_values[i],
            sigma_values[i],
            c=MSEs_gen[i],
            s=[(r - vmin) / (vmax - vmin) * (smax - smin) + smin for r in MSEs_gen[i]],
            cmap="rainbow",
            alpha=0.7,
            edgecolors="none",
            vmin=vmin,
            vmax=vmax,
        )
        # scatter1 = axs[i, 0].scatter(eta_values[i], sigma_values[i], c=MSEs_gen[i], s=logsizes(MSEs_gen[i], scale=sc), norm=lnorm, cmap="rainbow", alpha=0.7, edgecolors="none") # tried log scale, but even worse
        axs[i, 0].set_ylabel(r"$\sigma$", fontsize=9, labelpad=0)
        axs[i, 0].tick_params(axis="both", direction="in", which="major", labelsize=7)
        axs[i, 0].xaxis.set_major_locator(plt.MultipleLocator(0.2))
        axs[i, 0].xaxis.set_minor_locator(plt.MultipleLocator(0.1))
        axs[i, 0].yaxis.set_major_locator(plt.MultipleLocator(0.04))
        axs[i, 0].yaxis.set_minor_locator(plt.MultipleLocator(0.02))

        # Second subplot: MSE as scatter in eta-sigma plane (without beta correction)
        scatter2 = axs[i, 1].scatter(
            eta_values[i], sigma_values[i], c=MSEs_PY[i], s=[(r - vmin) / (vmax - vmin) * (smax - smin) + smin for r in MSEs_PY[i]], cmap="rainbow", alpha=0.7, edgecolors="none", vmin=vmin, vmax=vmax
        )
        # axs[i, 1].set_xlabel(r"$\eta$", fontsize=9, labelpad=0)
        axs[i, 1].tick_params(which="both", axis="both", direction="in", labelsize=7, labelleft=False)

        # Third subplot: MSE as scatter in eta-sigma plane (with beta correction)
        scatter3 = axs[i, 2].scatter(
            eta_values[i],
            sigma_values[i],
            c=MSEs_PYbeta[i],
            s=[(r - vmin) / (vmax - vmin) * (smax - smin) + smin for r in MSEs_PYbeta[i]],
            cmap="rainbow",
            alpha=0.7,
            edgecolors="none",
            vmin=vmin,
            vmax=vmax,
        )
        # axs[i, 2].set_xlabel(r"$\eta$", fontsize=9, labelpad=0)
        axs[i, 2].tick_params(which="both", axis="both", direction="in", labelsize=7, labelleft=False)

    for j in range(3):
        axs[2, j].set_xlabel(r"$\eta$", fontsize=9, labelpad=0)
        axs[2, j].tick_params(labelbottom=True)

    # Add colorbar on top spanning all axes
    # Create colorbar outside the axes
    cbar_ax = fig.add_axes([0.325, 0.925, 0.4, 0.02])  # [left, bottom, width, height]
    cbar = fig.colorbar(scatter1, cax=cbar_ax, orientation="horizontal")
    cbar.set_label("MSE", fontsize=7, labelpad=0)
    cbar.ax.xaxis.set_label_position("top")
    cbar.ax.xaxis.set_ticks_position("top")
    cbar.ax.tick_params(labelsize=7, direction="in", pad=0)
    cbar.ax.xaxis.set_major_locator(plt.MultipleLocator(0.004))

    axs[0, 0].text(-0.3, 1.05, "uniform", fontsize=9, transform=axs[0, 0].transAxes)
    axs[1, 0].text(-0.3, 1.05, "normal", fontsize=9, transform=axs[1, 0].transAxes)
    axs[2, 0].text(-0.3, 1.05, "lognormal", fontsize=9, transform=axs[2, 0].transAxes)

    axs[0, 0].text(0.95, 1.05, "NN", fontsize=9, transform=axs[0, 0].transAxes, ha="right")
    axs[0, 1].text(0.95, 1.05, "PY", fontsize=9, transform=axs[0, 1].transAxes, ha="right")
    axs[0, 2].text(0.95, 1.05, "PY w/ β", fontsize=9, transform=axs[0, 2].transAxes, ha="right")

    # add annotations
    annos = [r"$(a)$", r"$(b)$", r"$(c)$", r"$(d)$", r"$(e)$", r"$(f)$", r"$(g)$", r"$(h)$", r"$(i)$"]
    for i, ax in enumerate(axs.flatten()):
        ax.text(0.15, 0.1, annos[i], fontsize=9, transform=ax.transAxes)

    plt.tight_layout(pad=0.1, rect=[0, 0, 1, 0.925])
    # plt.tight_layout(pad=0.1)
    plt.savefig("./figures/gen_vs_PY_MSE_all.png", dpi=300)
    plt.savefig("./figures/gen_vs_PY_MSE_all.pdf")
    plt.show()
    plt.close()


def plot_inferrer_prediction_all(tex_lw=240.71031, ppi=72):
    print("plotting inferrer prediction")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    fig = plt.figure(figsize=(tex_lw / ppi * 1.0, tex_lw / ppi * 1.3))
    axs = fig.subplots(3, 2)

    inf_folder = "../data/data_pack"
    model_paths = ["../data/data_pack/L_18_pdType_1_inf_state_dict.pt", "../data/data_pack/L_18_pdType_2_inf_state_dict.pt", "../data/data_pack/L_18_pdType_3_inf_state_dict.pt"]
    # model_paths = ["../data/data_pack/L_18_pdType_2_gen_state_dict.pt"]
    pdTypes = [1, 2, 3]
    pdLabels = ["uniform", "normal", "lognormal"]

    for i in range(len(model_paths)):
        inferrer = Inferrer(input_dim=2, latent_dim=3).to(device)
        inferrer.load_state_dict(torch.load(model_paths[i], map_location=device))
        inferrer.eval()
        label = f"L_18_pdType_{pdTypes[i]:.0f}"
        # get data from npz
        test_loader, _ = create_dataloader(inf_folder, label, "test", batch_size=32, shuffle=False)

        all_true_params = []
        all_pred_params = []

        with torch.no_grad():
            for log10Iq, q, p in test_loader:
                log10Iq, p = log10Iq.to(device), p.to(device)
                # Get predictions from inferrer
                pred_avg, mu, logvar = inferrer(log10Iq)

                # Collect true and predicted parameters
                all_true_params.append(p.cpu().numpy())
                all_pred_params.append(pred_avg.cpu().numpy())

        # Concatenate all batches
        all_true_params = np.concatenate(all_true_params, axis=0)
        all_pred_params = np.concatenate(all_pred_params, axis=0)

        all_true_params_denorm = []
        all_pred_params_denorm = []
        for k in range(len(all_true_params)):
            _, true_denorm = denormalize_generated_Iq(inf_folder, label, np.zeros(100), all_true_params[k])
            _, pred_denorm = denormalize_generated_Iq(inf_folder, label, np.zeros(100), all_pred_params[k])
            all_true_params_denorm.append(true_denorm)
            all_pred_params_denorm.append(pred_denorm)

        all_true_params_denorm = np.array(all_true_params_denorm)
        all_pred_params_denorm = np.array(all_pred_params_denorm)

        for j in range(2):
            ax = axs[i, j]
            Y_pred = all_pred_params_denorm[:, j]
            Y = all_true_params_denorm[:, j]
            ax.scatter(Y, Y_pred, s=1, alpha=0.5)
            # ax.plot([0, 1], [0, 1], ls="--", color="black", lw=0.5)
            ax.tick_params(axis="both", direction="in", which="both", labelsize=7, labelbottom=True, labelleft=True, pad=1)
            ax.set_aspect("equal")

            scale = np.mean(np.abs(Y))  # or np.median(np.abs(Y)), or (Y.max()-Y.min())
            Err = 100 * np.abs(Y_pred - Y) / scale
            # Err = 100*np.abs(Y_pred - Y)/np.maximum(np.abs(Y), np.abs(Y_pred))
            Err = np.mean(Err)
            ax.text(0.4, 0.25, f"Err: {Err:.2f}%", fontsize=9, transform=ax.transAxes, ha="left")

            ax.text(0.1, 0.85, pdLabels[i], fontsize=9, transform=ax.transAxes, ha="left")

        # axs[i,0].text(0.9, 0.9, r"$\eta$", fontsize=9, transform=axs[i, 0].transAxes, ha="right")
        # axs[i,1].text(0.9, 0.9, r"$\sigma$", fontsize=9, transform=axs[i, 1].transAxes, ha="right")

        axs[i, 0].xaxis.set_major_locator(plt.MultipleLocator(0.2))
        axs[i, 0].xaxis.set_minor_locator(plt.MultipleLocator(0.1))
        axs[i, 0].yaxis.set_major_locator(plt.MultipleLocator(0.2))
        axs[i, 0].yaxis.set_minor_locator(plt.MultipleLocator(0.1))

        axs[i, 1].xaxis.set_major_locator(plt.MultipleLocator(0.04))
        axs[i, 1].xaxis.set_minor_locator(plt.MultipleLocator(0.02))
        axs[i, 1].yaxis.set_major_locator(plt.MultipleLocator(0.04))
        axs[i, 1].yaxis.set_minor_locator(plt.MultipleLocator(0.02))

        axs[0, 1].xaxis.set_major_locator(plt.MultipleLocator(0.1))
        axs[0, 1].xaxis.set_minor_locator(plt.MultipleLocator(0.05))
        axs[0, 1].yaxis.set_major_locator(plt.MultipleLocator(0.1))
        axs[0, 1].yaxis.set_minor_locator(plt.MultipleLocator(0.05))

    axs[0, 0].set_title(r"$\eta$", fontsize=9, pad=0)
    axs[0, 1].set_title(r"$\sigma$", fontsize=9, pad=0)

    # add annotations
    annos = [r"$(a)$", r"$(b)$", r"$(c)$", r"$(d)$", r"$(e)$", r"$(f)$"]
    for i, ax in enumerate(axs.flatten()):
        ax.text(0.85, 0.1, annos[i], fontsize=9, transform=ax.transAxes)

    axall = fig.add_subplot(111, frameon=False)
    axall.tick_params(labelcolor="none", which="both", top=False, bottom=False, left=False, right=False)
    axall.set_xlabel(r"Ground Truth", fontsize=9, labelpad=-8)
    axall.set_ylabel(r"NN Inferred", fontsize=9, labelpad=-14)

    plt.tight_layout(pad=0.1)
    plt.subplots_adjust(left=0.1, bottom=0.07, right=0.99)
    plt.savefig("./figures/inferrer_prediction_all.png", dpi=300)
    plt.savefig("./figures/inferrer_prediction_all.pdf")
    plt.show()
    plt.close()


def plot_gen_vs_PY_MSE_uni(tex_lw=240.71031, ppi=72):
    print("plotting generative model vs PY model MSE comparison")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    eta_ranges = [np.arange(0.05, 0.501, 0.05)]
    sigma_ranges = [np.arange(0.02, 0.281, 0.02)]

    fig = plt.figure(figsize=(tex_lw / ppi * 1.0, tex_lw / ppi * 0.5))
    # plt.rcParams['text.usetex'] = True
    axs = fig.subplots(1, 3, sharex=True)

    MSEs_gen = []
    MSEs_PY = []
    MSEs_PYbeta = []
    eta_values = []
    sigma_values = []

    data_folder = "../data/20250701"
    gen_folder = "../data/data_pack"
    model_paths = ["../data/data_pack/L_18_pdType_1_gen_state_dict.pt"]
    pdTypes = [1]

    for i in range(len(model_paths)):
        gen_model = Generator(input_dim=2, latent_dim=3).to(device)
        gen_model.load_state_dict(torch.load(model_paths[i], map_location=device))
        gen_model.eval()
        pdType = pdTypes[i]
        MSEs_gen.append([])
        MSEs_PY.append([])
        MSEs_PYbeta.append([])
        eta_values.append([])
        sigma_values.append([])
        gen_label = f"L_18_pdType_{pdTypes[i]:.0f}"
        # for eta in etas:
        # for sigma in sigmas:
        for eta in eta_ranges[i]:
            for sigma in sigma_ranges[i]:
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

    smin, smax = 0.1, 100
    for i in range(len(model_paths)):
        # First subplot: MSE as scatter in eta-sigma plane (generative model)
        # sc = 5000
        scatter1 = axs[0].scatter(
            eta_values[i],
            sigma_values[i],
            c=MSEs_gen[i],
            s=[(r - vmin) / (vmax - vmin) * (smax - smin) + smin for r in MSEs_gen[i]],
            cmap="rainbow",
            alpha=0.7,
            edgecolors="none",
            vmin=vmin,
            vmax=vmax,
        )
        # scatter1 = axs[0].scatter(eta_values[i], sigma_values[i], c=MSEs_gen[i], s=logsizes(MSEs_gen[i], scale=sc), norm=lnorm, cmap="rainbow", alpha=0.7, edgecolors="none") # tried log scale, but even worse
        axs[0].set_ylabel(r"$\sigma$", fontsize=9, labelpad=0)
        axs[0].tick_params(axis="both", direction="in", which="major", labelsize=7)
        axs[0].xaxis.set_major_locator(plt.MultipleLocator(0.2))
        axs[0].xaxis.set_minor_locator(plt.MultipleLocator(0.1))
        axs[0].yaxis.set_major_locator(plt.MultipleLocator(0.08))
        axs[0].yaxis.set_minor_locator(plt.MultipleLocator(0.04))

        # Second subplot: MSE as scatter in eta-sigma plane (without beta correction)
        scatter2 = axs[1].scatter(
            eta_values[i], sigma_values[i], c=MSEs_PY[i], s=[(r - vmin) / (vmax - vmin) * (smax - smin) + smin for r in MSEs_PY[i]], cmap="rainbow", alpha=0.7, edgecolors="none", vmin=vmin, vmax=vmax
        )
        # axs[1].set_xlabel(r"$\eta$", fontsize=9, labelpad=0)
        axs[1].tick_params(which="both", axis="both", direction="in", labelsize=7, labelleft=False)

        # Third subplot: MSE as scatter in eta-sigma plane (with beta correction)
        scatter3 = axs[2].scatter(
            eta_values[i],
            sigma_values[i],
            c=MSEs_PYbeta[i],
            s=[(r - vmin) / (vmax - vmin) * (smax - smin) + smin for r in MSEs_PYbeta[i]],
            cmap="rainbow",
            alpha=0.7,
            edgecolors="none",
            vmin=vmin,
            vmax=vmax,
        )
        # axs[2].set_xlabel(r"$\eta$", fontsize=9, labelpad=0)
        axs[2].tick_params(which="both", axis="both", direction="in", labelsize=7, labelleft=False)

    for j in range(3):
        axs[j].set_xlabel(r"$\eta$", fontsize=9, labelpad=0)
        axs[j].tick_params(labelbottom=True)

    # Add colorbar on top spanning all axes
    # Create colorbar outside the axes
    cbar_ax = fig.add_axes([0.325, 0.86, 0.4, 0.02])  # [left, bottom, width, height]
    cbar = fig.colorbar(scatter1, cax=cbar_ax, orientation="horizontal")
    cbar.set_label("MSE", fontsize=7, labelpad=0)
    cbar.ax.xaxis.set_label_position("top")
    cbar.ax.xaxis.set_ticks_position("top")
    cbar.ax.tick_params(labelsize=7, direction="in", pad=0)
    cbar.ax.xaxis.set_major_locator(plt.MultipleLocator(0.01))

    # axs[0].text(0.95, 1.05, "NN", fontsize=9, transform=axs[0].transAxes, ha="right")
    # axs[1].text(0.95, 1.05, "PY", fontsize=9, transform=axs[1].transAxes, ha="right")
    # axs[2].text(0.95, 1.05, "PY w/ β", fontsize=9, transform=axs[2].transAxes, ha="right")

    axs[0].set_title("NN", fontsize=9, pad=0)
    axs[1].set_title("PY", fontsize=9, pad=0)
    axs[2].set_title("PY w/ β", fontsize=9, pad=0)

    # add annotations
    annos = [r"$(a)$", r"$(b)$", r"$(c)$", r"$(d)$", r"$(e)$", r"$(f)$", r"$(g)$", r"$(h)$", r"$(i)$"]
    for i, ax in enumerate(axs.flatten()):
        ax.text(0.15, 0.1, annos[i], fontsize=9, transform=ax.transAxes)

    plt.tight_layout(pad=0.1, rect=[0, 0, 1, 0.85])
    # plt.tight_layout(pad=0.1)
    plt.savefig("./figures/gen_vs_PY_MSE_uni.png", dpi=300)
    plt.savefig("./figures/gen_vs_PY_MSE_uni.pdf")
    plt.show()
    plt.close()


def plot_inferrer_prediction_uni(tex_lw=240.71031, ppi=72):
    print("plotting inferrer prediction")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    fig = plt.figure(figsize=(tex_lw / ppi * 1.0, tex_lw / ppi * 0.5))
    axs = fig.subplots(1, 2)

    inf_folder = "../data/data_pack"
    model_paths = ["../data/data_pack/L_18_pdType_1_inf_state_dict.pt"]
    pdTypes = [1]

    for i in range(len(model_paths)):
        inferrer = Inferrer().to(device)
        inferrer.load_state_dict(torch.load(model_paths[i], map_location=device))
        inferrer.eval()
        label = f"L_18_pdType_{pdTypes[i]:.0f}"
        # get data from npz
        test_loader, _ = create_dataloader(inf_folder, label, "test", batch_size=32, shuffle=False)

        all_true_params = []
        all_pred_params = []

        with torch.no_grad():
            for log10Iq, q, p in test_loader:
                log10Iq, p = log10Iq.to(device), p.to(device)
                # Get predictions from inferrer
                pred_avg, mu, logvar = inferrer(log10Iq)

                # Collect true and predicted parameters
                all_true_params.append(p.cpu().numpy())
                all_pred_params.append(pred_avg.cpu().numpy())

        # Concatenate all batches
        all_true_params = np.concatenate(all_true_params, axis=0)
        all_pred_params = np.concatenate(all_pred_params, axis=0)

        all_true_params_denorm = []
        all_pred_params_denorm = []
        for k in range(len(all_true_params)):
            _, true_denorm = denormalize_generated_Iq(inf_folder, label, np.zeros(100), all_true_params[k])
            _, pred_denorm = denormalize_generated_Iq(inf_folder, label, np.zeros(100), all_pred_params[k])
            all_true_params_denorm.append(true_denorm)
            all_pred_params_denorm.append(pred_denorm)

        all_true_params_denorm = np.array(all_true_params_denorm)
        all_pred_params_denorm = np.array(all_pred_params_denorm)

        for j in range(2):
            ax = axs[j]
            Y_pred = all_pred_params_denorm[:, j]
            Y = all_true_params_denorm[:, j]
            ax.scatter(Y, Y_pred, facecolor="none", edgecolor="royalblue", s=2, alpha=0.5, linewidth=0.2)
            # ax.plot([0, 1], [0, 1], ls="--", color="black", lw=0.5)
            ax.tick_params(axis="both", direction="in", which="both", labelsize=7, labelbottom=True, labelleft=True, pad=1)
            ax.set_aspect("equal")

            scale = np.mean(np.abs(Y))  # or np.median(np.abs(Y)), or (Y.max()-Y.min())
            Err = 100 * np.abs(Y_pred - Y) / scale
            # Err = 100*np.abs(Y_pred - Y)/np.maximum(np.abs(Y), np.abs(Y_pred))
            Err = np.mean(Err)
            ax.text(0.4, 0.25, f"Err: {Err:.2f}%", fontsize=9, transform=ax.transAxes, ha="left")

        axs[0].xaxis.set_major_locator(plt.MultipleLocator(0.2))
        axs[0].xaxis.set_minor_locator(plt.MultipleLocator(0.1))
        axs[0].yaxis.set_major_locator(plt.MultipleLocator(0.2))
        axs[0].yaxis.set_minor_locator(plt.MultipleLocator(0.1))

        axs[1].xaxis.set_major_locator(plt.MultipleLocator(0.1))
        axs[1].xaxis.set_minor_locator(plt.MultipleLocator(0.05))
        axs[1].yaxis.set_major_locator(plt.MultipleLocator(0.1))
        axs[1].yaxis.set_minor_locator(plt.MultipleLocator(0.05))

    axs[0].text(0.2, 0.85, r"$\eta$", fontsize=9, transform=axs[0].transAxes, ha="left")
    axs[1].text(0.2, 0.85, r"$\sigma$", fontsize=9, transform=axs[1].transAxes, ha="left")
    axs[0].grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.5)
    axs[1].grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.5)
    """
    axs[0].set_xlabel(r"$\eta$", fontsize=9, labelpad=0)
    axs[0].set_ylabel(r"$\eta'$", fontsize=9, labelpad=0)
    axs[1].set_xlabel(r"$\sigma$", fontsize=9, labelpad=0)
    axs[1].set_ylabel(r"$\sigma'$", fontsize=9, labelpad=0)
    """

    # add annotations
    annos = [r"$(a)$", r"$(b)$", r"$(c)$", r"$(d)$", r"$(e)$", r"$(f)$"]
    for i, ax in enumerate(axs.flatten()):
        ax.text(0.85, 0.1, annos[i], fontsize=9, transform=ax.transAxes)

    axall = fig.add_subplot(111, frameon=False)
    axall.tick_params(labelcolor="none", which="both", top=False, bottom=False, left=False, right=False)
    axall.set_xlabel(r"Ground Truth", fontsize=9, labelpad=-15)
    axall.set_ylabel(r"NN Inferred", fontsize=9, labelpad=-10)

    plt.tight_layout(pad=0.1)
    plt.subplots_adjust(left=0.1, bottom=0.08, right=0.99, top=0.99)
    plt.savefig("./figures/inferrer_prediction_uni.png", dpi=300)
    plt.savefig("./figures/inferrer_prediction_uni.pdf")
    plt.show()
    plt.close()


def plot_gen_vs_PY_MSE_other(tex_lw=240.71031, ppi=72):
    print("plotting generative model vs PY model MSE comparison")

    data_folder = "../data/20250613"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    etas = np.arange(0.05, 0.451, 0.05)
    sigmas = np.arange(0.01, 0.101, 0.01)

    eta_ranges = [np.arange(0.05, 0.451, 0.05), np.arange(0.05, 0.451, 0.05)]
    sigma_ranges = [np.arange(0.01, 0.111, 0.01), np.arange(0.01, 0.121, 0.01)]

    fig = plt.figure(figsize=(tex_lw / ppi * 1.0, tex_lw / ppi * 0.9))
    axs = fig.subplots(2, 3, sharex=True)

    MSEs_gen = []
    MSEs_PY = []
    MSEs_PYbeta = []
    eta_values = []
    sigma_values = []

    data_folder = "../data/20250613"
    gen_folder = "../data/data_pack"
    model_paths = ["../data/data_pack/L_18_pdType_2_gen_state_dict.pt", "../data/data_pack/L_18_pdType_3_gen_state_dict.pt"]
    # model_paths = ["../data/data_pack/L_18_pdType_2_gen_state_dict.pt"]
    pdTypes = [2, 3]

    for i in range(len(model_paths)):
        gen_model = Generator(input_dim=2, latent_dim=3).to(device)
        gen_model.load_state_dict(torch.load(model_paths[i], map_location=device))
        gen_model.eval()
        pdType = pdTypes[i]
        MSEs_gen.append([])
        MSEs_PY.append([])
        MSEs_PYbeta.append([])
        eta_values.append([])
        sigma_values.append([])
        gen_label = f"L_18_pdType_{pdTypes[i]:.0f}"
        # for eta in etas:
        # for sigma in sigmas:
        for eta in eta_ranges[i]:
            for sigma in sigma_ranges[i]:
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

    smin, smax = 0.1, 100
    for i in range(len(model_paths)):
        # First subplot: MSE as scatter in eta-sigma plane (generative model)
        # sc = 5000
        scatter1 = axs[i, 0].scatter(
            eta_values[i],
            sigma_values[i],
            c=MSEs_gen[i],
            s=[(r - vmin) / (vmax - vmin) * (smax - smin) + smin for r in MSEs_gen[i]],
            cmap="rainbow",
            alpha=0.7,
            edgecolors="none",
            vmin=vmin,
            vmax=vmax,
        )
        # scatter1 = axs[i, 0].scatter(eta_values[i], sigma_values[i], c=MSEs_gen[i], s=logsizes(MSEs_gen[i], scale=sc), norm=lnorm, cmap="rainbow", alpha=0.7, edgecolors="none") # tried log scale, but even worse
        axs[i, 0].set_ylabel(r"$\sigma$", fontsize=9, labelpad=0)
        axs[i, 0].tick_params(axis="both", direction="in", which="major", labelsize=7)
        axs[i, 0].xaxis.set_major_locator(plt.MultipleLocator(0.2))
        axs[i, 0].xaxis.set_minor_locator(plt.MultipleLocator(0.1))
        axs[i, 0].yaxis.set_major_locator(plt.MultipleLocator(0.04))
        axs[i, 0].yaxis.set_minor_locator(plt.MultipleLocator(0.02))

        # Second subplot: MSE as scatter in eta-sigma plane (without beta correction)
        scatter2 = axs[i, 1].scatter(
            eta_values[i], sigma_values[i], c=MSEs_PY[i], s=[(r - vmin) / (vmax - vmin) * (smax - smin) + smin for r in MSEs_PY[i]], cmap="rainbow", alpha=0.7, edgecolors="none", vmin=vmin, vmax=vmax
        )
        # axs[i, 1].set_xlabel(r"$\eta$", fontsize=9, labelpad=0)
        axs[i, 1].tick_params(which="both", axis="both", direction="in", labelsize=7, labelleft=False)

        # Third subplot: MSE as scatter in eta-sigma plane (with beta correction)
        scatter3 = axs[i, 2].scatter(
            eta_values[i],
            sigma_values[i],
            c=MSEs_PYbeta[i],
            s=[(r - vmin) / (vmax - vmin) * (smax - smin) + smin for r in MSEs_PYbeta[i]],
            cmap="rainbow",
            alpha=0.7,
            edgecolors="none",
            vmin=vmin,
            vmax=vmax,
        )
        # axs[i, 2].set_xlabel(r"$\eta$", fontsize=9, labelpad=0)
        axs[i, 2].tick_params(which="both", axis="both", direction="in", labelsize=7, labelleft=False)

    for j in range(3):
        axs[1, j].set_xlabel(r"$\eta$", fontsize=9, labelpad=0)
        axs[1, j].tick_params(labelbottom=True)

    # Add colorbar on top spanning all axes
    # Create colorbar outside the axes
    cbar_ax = fig.add_axes([0.325, 0.91, 0.4, 0.02])  # [left, bottom, width, height]
    cbar = fig.colorbar(scatter1, cax=cbar_ax, orientation="horizontal")
    cbar.set_label("MSE", fontsize=7, labelpad=0)
    cbar.ax.xaxis.set_label_position("top")
    cbar.ax.xaxis.set_ticks_position("top")
    cbar.ax.tick_params(labelsize=7, direction="in", pad=0)
    cbar.ax.xaxis.set_major_locator(plt.MultipleLocator(0.004))

    # axs[0, 0].text(-0.3, 1.05, "uniform", fontsize=9, transform=axs[0, 0].transAxes)
    axs[0, 0].text(-0.3, 1.05, "normal", fontsize=9, transform=axs[0, 0].transAxes)
    axs[1, 0].text(-0.3, 1.05, "lognormal", fontsize=9, transform=axs[1, 0].transAxes)

    axs[0, 0].text(0.95, 1.05, "NN", fontsize=9, transform=axs[0, 0].transAxes, ha="right")
    axs[0, 1].text(0.95, 1.05, "PY", fontsize=9, transform=axs[0, 1].transAxes, ha="right")
    axs[0, 2].text(0.95, 1.05, "PY w/ β", fontsize=9, transform=axs[0, 2].transAxes, ha="right")

    # add annotations
    annos = [r"$(a)$", r"$(b)$", r"$(c)$", r"$(d)$", r"$(e)$", r"$(f)$", r"$(g)$", r"$(h)$", r"$(i)$"]
    for i, ax in enumerate(axs.flatten()):
        ax.text(0.15, 0.1, annos[i], fontsize=9, transform=ax.transAxes)

    plt.tight_layout(pad=0.1, rect=[0, 0, 1, 0.9])
    # plt.tight_layout(pad=0.1)
    plt.savefig("./figures/gen_vs_PY_MSE_other.png", dpi=300)
    plt.savefig("./figures/gen_vs_PY_MSE_other.pdf")
    plt.show()
    plt.close()


def plot_inferrer_prediction_other(tex_lw=240.71031, ppi=72):
    print("plotting inferrer prediction")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    fig = plt.figure(figsize=(tex_lw / ppi * 1.0, tex_lw / ppi * 1))
    axs = fig.subplots(2, 2)

    inf_folder = "../data/data_pack"
    model_paths = ["../data/data_pack/L_18_pdType_2_inf_state_dict.pt", "../data/data_pack/L_18_pdType_3_inf_state_dict.pt"]
    # model_paths = ["../data/data_pack/L_18_pdType_2_gen_state_dict.pt"]
    pdTypes = [2, 3]
    pdLabels = ["normal", "lognormal"]

    for i in range(len(model_paths)):
        # inferrer = Inferrer(input_dim=2, latent_dim=3).to(device)
        inferrer = Inferrer().to(device)
        inferrer.load_state_dict(torch.load(model_paths[i], map_location=device))
        inferrer.eval()
        label = f"L_18_pdType_{pdTypes[i]:.0f}"
        # get data from npz
        test_loader, _ = create_dataloader(inf_folder, label, "test", batch_size=32, shuffle=False)

        all_true_params = []
        all_pred_params = []

        with torch.no_grad():
            for log10Iq, q, p in test_loader:
                log10Iq, p = log10Iq.to(device), p.to(device)
                # Get predictions from inferrer
                pred_avg, mu, logvar = inferrer(log10Iq)

                # Collect true and predicted parameters
                all_true_params.append(p.cpu().numpy())
                all_pred_params.append(pred_avg.cpu().numpy())

        # Concatenate all batches
        all_true_params = np.concatenate(all_true_params, axis=0)
        all_pred_params = np.concatenate(all_pred_params, axis=0)

        all_true_params_denorm = []
        all_pred_params_denorm = []
        for k in range(len(all_true_params)):
            _, true_denorm = denormalize_generated_Iq(inf_folder, label, np.zeros(100), all_true_params[k])
            _, pred_denorm = denormalize_generated_Iq(inf_folder, label, np.zeros(100), all_pred_params[k])
            all_true_params_denorm.append(true_denorm)
            all_pred_params_denorm.append(pred_denorm)

        all_true_params_denorm = np.array(all_true_params_denorm)
        all_pred_params_denorm = np.array(all_pred_params_denorm)

        for j in range(2):
            ax = axs[i, j]
            Y_pred = all_pred_params_denorm[:, j]
            Y = all_true_params_denorm[:, j]
            # ax.scatter(Y, Y_pred, s=1, alpha=0.5)
            ax.scatter(Y, Y_pred, facecolor="none", edgecolor="royalblue", s=2, alpha=0.5, linewidth=0.2)
            # ax.plot([0, 1], [0, 1], ls="--", color="black", lw=0.5)
            ax.tick_params(axis="both", direction="in", which="both", labelsize=7, labelbottom=True, labelleft=True, pad=1)
            ax.set_aspect("equal")

            scale = np.mean(np.abs(Y))  # or np.median(np.abs(Y)), or (Y.max()-Y.min())
            Err = 100 * np.abs(Y_pred - Y) / scale
            # Err = 100*np.abs(Y_pred - Y)/np.maximum(np.abs(Y), np.abs(Y_pred))
            Err = np.mean(Err)
            ax.text(0.4, 0.25, f"Err: {Err:.2f}%", fontsize=9, transform=ax.transAxes, ha="left")

            ax.text(0.1, 0.85, pdLabels[i], fontsize=9, transform=ax.transAxes, ha="left")

        # axs[i,0].text(0.9, 0.9, r"$\eta$", fontsize=9, transform=axs[i, 0].transAxes, ha="right")
        # axs[i,1].text(0.9, 0.9, r"$\sigma$", fontsize=9, transform=axs[i, 1].transAxes, ha="right")

        axs[i, 0].xaxis.set_major_locator(plt.MultipleLocator(0.2))
        axs[i, 0].xaxis.set_minor_locator(plt.MultipleLocator(0.1))
        axs[i, 0].yaxis.set_major_locator(plt.MultipleLocator(0.2))
        axs[i, 0].yaxis.set_minor_locator(plt.MultipleLocator(0.1))

        axs[i, 1].xaxis.set_major_locator(plt.MultipleLocator(0.04))
        axs[i, 1].xaxis.set_minor_locator(plt.MultipleLocator(0.02))
        axs[i, 1].yaxis.set_major_locator(plt.MultipleLocator(0.04))
        axs[i, 1].yaxis.set_minor_locator(plt.MultipleLocator(0.02))

        # axs[0, 1].xaxis.set_major_locator(plt.MultipleLocator(0.1))
        # axs[0, 1].xaxis.set_minor_locator(plt.MultipleLocator(0.05))
        # axs[0, 1].yaxis.set_major_locator(plt.MultipleLocator(0.1))
        # axs[0, 1].yaxis.set_minor_locator(plt.MultipleLocator(0.05))

    axs[0, 0].set_title(r"$\eta$", fontsize=9, pad=0)
    axs[0, 1].set_title(r"$\sigma$", fontsize=9, pad=0)

    # add annotations
    annos = [r"$(a)$", r"$(b)$", r"$(c)$", r"$(d)$", r"$(e)$", r"$(f)$"]
    for i, ax in enumerate(axs.flatten()):
        ax.text(0.85, 0.1, annos[i], fontsize=9, transform=ax.transAxes)

    axall = fig.add_subplot(111, frameon=False)
    axall.tick_params(labelcolor="none", which="both", top=False, bottom=False, left=False, right=False)
    axall.set_xlabel(r"Ground Truth", fontsize=9, labelpad=-8)
    axall.set_ylabel(r"NN Inferred", fontsize=9, labelpad=-12)

    plt.tight_layout(pad=0.1)
    plt.subplots_adjust(left=0.1, bottom=0.07, right=0.98)
    plt.savefig("./figures/inferrer_prediction_other.png", dpi=300)
    plt.savefig("./figures/inferrer_prediction_other.pdf")
    plt.show()
    plt.close()
