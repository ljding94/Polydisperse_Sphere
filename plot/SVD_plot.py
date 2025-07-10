import matplotlib.pyplot as plt
import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "analyze"))
from analyze import *
from VAE_model import *


def plot_svd_analysis(tex_lw=240.71031, ppi=72):
    print("Plotting SVD analysis of I(q) data...")

    folder = "../data/20250613"
    folder = "../data/data_pack"
    label = "L_18_pdType_1"
    Iq_data_train = np.load(f"{folder}/{label}_train_data.npz")
    Iq_data_test = np.load(f"{folder}/{label}_test_data.npz")
    all_Iq = np.concatenate([Iq_data_train["log10Iq"], Iq_data_test["log10Iq"]], axis=0)
    print("all_Iq, shape:", all_Iq.shape)
    q = Iq_data_train["q"]
    all_params = np.concatenate([Iq_data_train["params"], Iq_data_test["params"]], axis=0)
    params_name = Iq_data_train["params_name"]

    fig = plt.figure(figsize=(tex_lw / ppi * 1.0, tex_lw / ppi * 0.45))

    ax1 = fig.add_subplot(121)  # SVD plot
    ax2 = fig.add_subplot(122)  # V plot
    # ax3 = fig.add_subplot(223, projection="3d") # eta distribution plot
    # ax4 = fig.add_subplot(224, projection="3d") # sigma distribution plot

    F = np.array(all_Iq)  # already log10Iq
    print("samples, shape:", F.shape)
    U, S, Vh = np.linalg.svd(F, full_matrices=False)
    print("Singular values:", S)

    # Plot singular values
    ax1.plot(range(1, len(S) + 1), S, "o--", ms=5, lw=1, markerfacecolor="none")
    # ax1.set_yscale("log")
    ax1.set_xscale("log")
    ax1.set_xlabel("SVR", fontsize=9, labelpad=0)
    ax1.set_ylabel(r"$\Sigma$", fontsize=9, labelpad=0)
    ax1.tick_params(axis="both", direction="in", labelsize=7)
    ax1.yaxis.set_major_locator(plt.MultipleLocator(400))
    ax1.yaxis.set_minor_locator(plt.MultipleLocator(200))

    # Plot first three singular vectors I(q) vs q
    for i in range(3):
        ax2.plot(q, Vh[i, :], lw=1, label=f"V{i}")
    ax2.set_xlabel(r"$Q$", fontsize=9, labelpad=0)
    ax2.set_ylabel(r"$V$", fontsize=9, labelpad=0)
    ax2.tick_params(axis="both", direction="in", labelsize=7)
    ax2.legend(loc="upper right", fontsize=7, frameon=False, ncol=1, columnspacing=0.5, handlelength=1, handletextpad=0.2)
    ax2.xaxis.set_major_locator(plt.MultipleLocator(4))
    ax2.xaxis.set_minor_locator(plt.MultipleLocator(2))
    ax2.yaxis.set_major_locator(plt.MultipleLocator(0.2))
    ax2.yaxis.set_minor_locator(plt.MultipleLocator(0.1))
    ax2.set_ylim(-0.25,0.4)

    # add annotations
    ax1.text(0.8, 0.1, r"$(a)$", fontsize=9, transform=ax1.transAxes)
    ax2.text(0.8, 0.1, r"$(b)$", fontsize=9, transform=ax2.transAxes)

    plt.tight_layout(pad=0.1)
    plt.savefig("./figures/svd_analysis.png", dpi=300)
    plt.savefig("./figures/svd_analysis.pdf", format="pdf")
    plt.show()

    plt.close()
    """
    #distribution of eta and sigma

    fig = plt.figure(figsize=(tex_lw / ppi * 1.0, tex_lw / ppi * 0.5))
    ax3 = fig.add_subplot(121, projection="3d")  # eta distribution plot
    ax4 = fig.add_subplot(122, projection="3d")  # sigma distribution

    eta = all_params[:, 2]
    sigma = all_params[:, 3]

    FV = np.dot(F, np.transpose(Vh))  # Project data onto the right singular vectors
    sc3 = ax3.scatter(FV[:, 0], FV[:, 1], FV[:, 2], c=eta, cmap="rainbow", s=5)
    ax3.set_xlabel(r"$FV0$", fontsize=9, labelpad=-10)
    ax3.set_ylabel(r"$FV1$", fontsize=9, labelpad=-10)
    ax3.set_zlabel(r"$FV2$", fontsize=9, labelpad=-12)
    ax3.xaxis.set_major_locator(plt.MultipleLocator(1))
    ax3.yaxis.set_major_locator(plt.MultipleLocator(2))
    ax3.zaxis.set_major_locator(plt.MultipleLocator(2))
    ax3.view_init(elev=20.0, azim=-150)
    ax3.set_proj_type('persp', focal_length=0.5)
    ax3.tick_params(axis='both', direction="in", labelsize=7, pad=-5)
    #cbar3 = plt.colorbar(sc3, ax=ax3, shrink=0.4, orientation="horizontal", location="left")
    cbar3 = plt.colorbar(sc3, ax=ax3, shrink=0.4, orientation="vertical", location="right")
    cbar3.ax.tick_params(labelsize=7, pad=0)
    cbar3.ax.set_title(r"$\eta$", fontsize=9, pad=0)



    sc4 = ax4.scatter(FV[:, 0], FV[:, 1], FV[:, 2], c=sigma, cmap="rainbow", s=5)
    ax4.set_xlabel(r"$FV0$", fontsize=9, labelpad=-10)
    ax4.set_ylabel(r"$FV1$", fontsize=9, labelpad=-10)
    ax4.set_zlabel(r"$FV2$", fontsize=9, labelpad=-10)
    ax4.tick_params(axis='both', direction="in", labelsize=7, pad=-4)
    #ax4.set_yscale("log")
    #cbar4 = plt.colorbar(sc4, ax=ax4, shrink=0.4, orientation="horizontal", location="top")
    cbar4 = plt.colorbar(sc4, ax=ax4, shrink=0.4, orientation="vertical", location="right")
    cbar4.ax.set_title(r"$\sigma$", fontsize=9, pad=0)
    cbar4.ax.tick_params(labelsize=7, pad=0)
    ax4.view_init(elev=20.0, azim=-150)

    plt.tight_layout(pad==1.0)
    plt.savefig("./figures/svd_distribution.png", dpi=300)
    plt.savefig("./figures/svd_distribution.pdf", format='pdf')
    plt.show()

    plt.close()
    """


def plot_svd_distribution(tex_lw=240.71031, ppi=72):
    print("Plotting SVD analysis of I(q) data...")

    folder = "../data/20250613"
    folder = "../data/data_pack"
    label = "L_18_pdType_1"
    Iq_data_train = np.load(f"{folder}/{label}_train_data.npz")
    Iq_data_test = np.load(f"{folder}/{label}_test_data.npz")
    all_Iq = np.concatenate([Iq_data_train["log10Iq"], Iq_data_test["log10Iq"]], axis=0)
    print("all_Iq, shape:", all_Iq.shape)
    q = Iq_data_train["q"]
    all_params = np.concatenate([Iq_data_train["params"], Iq_data_test["params"]], axis=0)
    params_name = Iq_data_train["params_name"]

    F = np.array(all_Iq)  # already log10Iq
    print("samples, shape:", F.shape)
    U, S, Vh = np.linalg.svd(F, full_matrices=False)
    print("Singular values:", S)

    fig = plt.figure(figsize=(tex_lw / ppi * 1.0, tex_lw / ppi * 0.5))

    ax3 = fig.add_subplot(121, projection="3d")  # eta distribution plot
    ax4 = fig.add_subplot(122, projection="3d")  # sigma distribution

    eta = all_params[:, 2]
    sigma = all_params[:, 3]

    ax3.set_box_aspect([1, 1, 1.2])
    ax4.set_box_aspect([1, 1, 1.2])
    FV = np.dot(F, np.transpose(Vh))  # Project data onto the right singular vectors
    sc3 = ax3.scatter(FV[:, 0], FV[:, 1], FV[:, 2], c=eta, cmap="rainbow", s=1, rasterized=True)
    ax3.set_xlabel(r"$FV0$", fontsize=9, labelpad=-10)
    ax3.set_ylabel(r"$FV1$", fontsize=9, labelpad=-10)
    ax3.set_zlabel(r"$FV2$", fontsize=9, labelpad=-12)
    ax3.xaxis.set_major_locator(plt.MultipleLocator(1))
    ax3.yaxis.set_major_locator(plt.MultipleLocator(2))
    ax3.zaxis.set_major_locator(plt.MultipleLocator(2))
    ax3.view_init(elev=20.0, azim=-155)
    ax3.set_proj_type("persp", focal_length=0.5)
    ax3.tick_params(axis="both", direction="in", labelsize=7, pad=-4)
    # cbar3 = plt.colorbar(sc3, ax=ax3, shrink=0.4, orientation="horizontal", location="left")
    cbar3 = plt.colorbar(sc3, ax=ax3, shrink=0.4, orientation="vertical", location="right")
    cbar3.ax.tick_params(labelsize=7, pad=0)
    cbar3.ax.set_title(r"$\eta$", fontsize=9, pad=0)

    sc4 = ax4.scatter(FV[:, 0], FV[:, 1], FV[:, 2], c=sigma, cmap="rainbow", s=1, rasterized=True)
    ax4.set_xlabel(r"$FV0$", fontsize=9, labelpad=-10)
    ax4.set_ylabel(r"$FV1$", fontsize=9, labelpad=-10)
    ax4.set_zlabel(r"$FV2$", fontsize=9, labelpad=-12)
    ax4.xaxis.set_major_locator(plt.MultipleLocator(1))
    ax4.yaxis.set_major_locator(plt.MultipleLocator(2))
    ax4.zaxis.set_major_locator(plt.MultipleLocator(2))
    ax4.tick_params(axis="both", direction="in", labelsize=7, pad=-4)
    # ax4.set_yscale("log")
    # cbar4 = plt.colorbar(sc4, ax=ax4, shrink=0.4, orientation="horizontal", location="top")
    cbar4 = plt.colorbar(sc4, ax=ax4, shrink=0.4, orientation="vertical", location="right")
    cbar4.ax.set_title(r"$\sigma$", fontsize=9, pad=0)
    cbar4.ax.tick_params(labelsize=7, pad=0)
    ax4.view_init(elev=20.0, azim=-155)

    # add annotations
    ax3.text2D(1.1, 0.05, r"$(c)$", fontsize=9, transform=ax3.transAxes)
    ax4.text2D(1.1, 0.05, r"$(d)$", fontsize=9, transform=ax4.transAxes)

    plt.tight_layout(pad=0.5)
    plt.subplots_adjust(left=0.05, right=0.975, top=1.15, bottom=0.00)
    plt.savefig("./figures/svd_distribution.png", dpi=300)
    plt.savefig("./figures/svd_distribution.pdf", format="pdf", dpi=600)
    plt.show()

    plt.close()


from pypdf import PdfReader, PdfWriter, PageObject,  Transformation


def plot_combine_svd():
    top_pdf = "figures/svd_analysis.pdf"
    bot_pdf = "figures/svd_distribution.pdf"
    out_pdf = "figures/svd_combined.pdf"

    # ---- read the FIRST page from each file ----
    top_page    = PdfReader(top_pdf).pages[0]
    bottom_page = PdfReader(bot_pdf).pages[0]

    # ---- build a blank page tall enough for both ----
    width  = top_page.mediabox.width
    height = top_page.mediabox.height + bottom_page.mediabox.height
    stacked = PageObject.create_blank_page(width=width, height=height)

    # ---- paste the bottom page at (0, 0) ----
    stacked.merge_page(bottom_page)

    # ---- paste the top page, shifted upward ----
    shift = Transformation().translate(tx=0, ty=bottom_page.mediabox.height)
    # either of these two lines works – pick one
    stacked.merge_transformed_page(top_page, shift)   # 1️⃣
    # top_page.add_transformation(shift); stacked.merge_page(top_page)  # 2️⃣

    # ---- write out ----
    writer = PdfWriter()
    writer.add_page(stacked)
    with open(out_pdf, "wb") as f:
        writer.write(f)
