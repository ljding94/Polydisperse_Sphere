import numpy as np
import sys
import os

import matplotlib.pyplot as plt

# Add parent directory to path so we can import from there
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "analyze"))
from analyze import *


def plot_three_distribution(tex_lw=240.71031, ppi=72):
    fig = plt.figure(figsize=(tex_lw / ppi * 1.0, tex_lw / ppi * 0.8))

    ax1 = fig.add_subplot(212)
    ax2 = fig.add_subplot(231)
    ax3 = fig.add_subplot(232)
    ax4 = fig.add_subplot(233)

    # plot distributions in ax1
    sigma = 0.10

    # 1. uniform distribution
    x = np.linspace(0.6, 1.4, 100)
    y_unif = np.zeros_like(x)
    mask_unif = (x >= 1 - sigma) & (x <= 1 + sigma)
    y_unif[mask_unif] = 1 / (2 * sigma)  # PDF height = 1/(b-a) where b-a is range width
    ax1.plot(x, y_unif, lw=1, label="uniform")

    # 2 Normal disttribution with mean=1 and std=sigma
    y_norm = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - 1) / sigma) ** 2)
    ax1.plot(x, y_norm, lw=1, label="normal", color="orange")
    # 3. log-normal distribution with mean=1 and std=sigma
    y_lognorm = (1 / (x * sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((np.log(x) - np.log(1)) / sigma) ** 2)
    ax1.plot(x, y_lognorm, lw=1, label="log-normal", color="green")
    ax1.set_xlabel(r"$D$", fontsize=9, labelpad=0)
    ax1.set_ylabel(r"$P(D)$", fontsize=9, labelpad=0)
    ax1.tick_params(axis="both", direction="in", labelsize=7)
    # ax1.legend(frameon=False, fontsize=9)
    ax1.legend(loc="center right", ncol=1, columnspacing=0.5, handlelength=1, handletextpad=0.2, frameon=False, fontsize=9)

    # Load and display image in ax2
    img = plt.imread("figures/L_4_pdType_1_eta_0.10_sigma_0.10.png")
    ax2.imshow(img)
    ax2.axis("off")  # Hide axis for cleaner image display

    img = plt.imread("figures/L_4_pdType_2_eta_0.10_sigma_0.10.png")
    ax3.imshow(img)
    ax3.axis("off")  # Hide axis for cleaner image display
    img = plt.imread("figures/L_4_pdType_3_eta_0.10_sigma_0.10.png")
    ax4.imshow(img)
    ax4.axis("off")  # Hide axis for cleaner image display
    # Create a colormap (blue-white-red)

    cmap = plt.cm.get_cmap("bwr")
    # Create a mock ScalarMappable for the colorbar
    norm = plt.Normalize(0.6, 1.4)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    # Add colorbar to the left of ax2
    cbar_ax = fig.add_axes([0.06, 0.65, 0.01, 0.2])  # [left, bottom, width, height]
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.ax.set_title(r"$D$", fontsize=9, pad=5)
    cbar.ax.yaxis.set_ticks_position("left")
    cbar.ax.yaxis.set_label_position("left")
    cbar.ax.tick_params(direction="in", labelsize=7)
    # Set specific ticks for the colorbar
    cbar.set_ticks([0.6, 1.0, 1.4])

    # add annotations
    ax2.text(0.9, 0.0, r"$(a)$", fontsize=9, transform=ax2.transAxes)
    ax3.text(0.9, 0.0, r"$(b)$", fontsize=9, transform=ax3.transAxes)
    ax4.text(0.9, 0.0, r"$(c)$", fontsize=9, transform=ax4.transAxes)
    ax1.text(0.9, 0.15, r"$(d)$", fontsize=9, transform=ax1.transAxes)

    plt.tight_layout(pad=0.1)
    plt.savefig("figures/three_distribution.png", dpi=300)
    plt.savefig("figures/three_distribution.pdf", dpi=300)
    plt.show()


def plot_Iq_and_config(tex_lw=240.71031, ppi=72):
    fig = plt.figure(figsize=(tex_lw / ppi * 1.0, tex_lw / ppi * 0.6))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)


    img = plt.imread("figures/L_18_pdType_1_eta_0.10_sigma_0.30.png")
    ax1.imshow(img) #, extent=[0.2, 0.8, 0.2, 0.8])
    ax1.axis("off")  # Hide axis for cleaner image display
    ax1.text(0.0, -0.1, r"$\eta=0.1,\sigma=0.3$", fontsize=9, transform=ax1.transAxes)

    # Add colorbar to the left of ax1
    cmap = plt.cm.get_cmap("bwr")
    # Create a mock ScalarMappable for the colorbar
    norm = plt.Normalize(0.7, 1.3)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar_ax = fig.add_axes([0.06, 0.4, 0.01, 0.4])  # [left, bottom, width, height]
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.ax.set_title(r"$D$", fontsize=9, pad=5)
    cbar.ax.yaxis.set_ticks_position("left")
    cbar.ax.yaxis.set_label_position("left")
    cbar.ax.tick_params(direction="in", labelsize=7)
    # Set specific ticks for the colorbar
    cbar.set_ticks([0.7, 1.0, 1.3])

    shift = 1
    eta_sigma_paris = [(0.1, 0.0), (0.10, 0.3), (0.3, 0.0), (0.30, 0.3)]
    for eta, sigma in eta_sigma_paris:
        finfo = f"L_18_pdType_1_eta_{eta:.2f}_sigma_{sigma:.2f}"
        q, Iq, Iq_smooth, L, pdType, eta, sigma = get_smoothed_Iq_per_file("../data/data_pack/prec_data/20250701", finfo, 0)
        ax2.plot(q, Iq_smooth * shift, "o-", ms=1.5, mfc="None", mew=0.5, label=f"({eta},{sigma})", lw=0.5)
        shift *= 100

    ax2.set_yscale("log")
    ax2.set_xlabel(r"$Q$", fontsize=9, labelpad=0)
    ax2.set_ylabel(r"$I(Q)$", fontsize=9, labelpad=0)
    ax2.xaxis.set_major_locator(plt.MultipleLocator(4))
    ax2.xaxis.set_minor_locator(plt.MultipleLocator(2))
    ax2.tick_params(axis="both", which="both", direction="in", labelsize=7)
    ax2.legend(title=r"$(\eta,\sigma)$", frameon=False, fontsize=7, loc="lower left", ncol=1, columnspacing=0.5, handlelength=1, handletextpad=0.2, labelspacing=0.1)
    # Set major locator for better tick spacing

    # add annotations
    ax1.text(0.9, 0.0, r"$(a)$", fontsize=9, transform=ax1.transAxes)
    ax2.text(0.8, 0.075, r"$(b)$", fontsize=9, transform=ax2.transAxes)

    plt.tight_layout(pad=0.1)
    plt.subplots_adjust(left=0.08, right=0.99, top=0.98, bottom=0.15)
    plt.savefig("figures/Iq_and_config.png", dpi=500)
    plt.savefig("figures/Iq_and_config.pdf", dpi=500)
    plt.show()
    plt.close()


def plot_Iq_per_eta_sigma(tex_lw=240.71031, ppi=72):
    fig = plt.figure(figsize=(tex_lw / ppi * 1.0, tex_lw / ppi * 0.6))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122, sharex=ax1, sharey=ax1)

    sigma = 0.05
    for i, eta in enumerate(np.arange(0.05, 0.501, 0.15)):
        finfo = f"L_18_pdType_1_eta_{eta:.2f}_sigma_{sigma:.2f}"
        q, Iq, Iq_smooth, L, pdType, eta, sigma = get_smoothed_Iq_per_file("../data/20250701", finfo, 0)
        ax1.plot(q, Iq_smooth * 100**i, "o-", ms=1.5, mfc="None", mew=0.5, label=f"({eta},{sigma})", lw=0.5)
        ax1.text(13, 100 ** (i - 1.7), r"$\eta$" + f"={eta:.2f}", fontsize=7, ha="right")
    ax1.text(4, 10**-4, r"$\sigma$="+f"{sigma:.2f}", fontsize=7, ha="left")
    ax1.set_yscale("log")
    ax1.set_xlabel(r"$Q$", fontsize=9, labelpad=0)
    ax1.set_ylabel(r"$I(Q)$", fontsize=9, labelpad=0)
    ax1.xaxis.set_major_locator(plt.MultipleLocator(4))
    ax1.xaxis.set_minor_locator(plt.MultipleLocator(2))
    ax1.tick_params(axis="both", which="both", direction="in", labelsize=7)
    # ax1.legend(title=r"$(\eta,\sigma)$",frameon=False, fontsize=7, loc="lower left", ncol=1, columnspacing=0.5, handlelength=1, handletextpad=0.2)

    eta = 0.30
    for i, sigma in enumerate(np.arange(0.0, 0.31, 0.1)):
        finfo = f"L_18_pdType_1_eta_{eta:.2f}_sigma_{sigma:.2f}"
        q, Iq, Iq_smooth, L, pdType, eta, sigma = get_smoothed_Iq_per_file("../data/20250701", finfo, 0)
        ax2.plot(q, Iq_smooth * 100**i, "o-", ms=1.5, mfc="None", mew=0.5, label=f"({eta},{sigma})", lw=0.5)
        ax2.text(13, 100 ** (i - 1.8), r"$\sigma$" + f"={sigma:.2f}", fontsize=7, ha="right")
    ax2.text(4, 10**-4, r"$\eta$="+f"{eta:.2f}", fontsize=7, ha="left")

    ax2.set_yscale("log")
    ax2.set_xlabel(r"$Q$", fontsize=9, labelpad=0)
    # ax2.set_ylabel(r"$I(Q)$", fontsize=9, labelpad=0)
    ax2.xaxis.set_major_locator(plt.MultipleLocator(4))
    ax2.xaxis.set_minor_locator(plt.MultipleLocator(2))
    ax2.tick_params(axis="both", which="both", direction="in", labelsize=7, labelleft=False)
    # ax2.legend(title=r"$(\eta,\sigma)$",frameon=False, fontsize=7, loc="lower left", ncol=1, columnspacing=0.5, handlelength=1, handletextpad=0.2)
    # Set major locator for better tick spacing

    # add annotations
    ax1.text(0.8, 0.075, r"$(a)$", fontsize=9, transform=ax1.transAxes)
    ax2.text(0.8, 0.075, r"$(b)$", fontsize=9, transform=ax2.transAxes)

    plt.tight_layout(pad=0.5)
    plt.savefig("figures/Iq_example.png", dpi=500)
    plt.savefig("figures/Iq_example.pdf")
    plt.show()
    plt.close()
