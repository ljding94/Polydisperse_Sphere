from illustrate_plot import *
from NN_model_plot import *
def main():
    print("plotting figures for pd HS")

    # 1 illustrate 3 distribution
    #plot_three_distribution()

    # 2, illustrative system config of HS, and some I(Q) plot

    #plot_Iq_and_config()


    # 3, SVD analysis of the I(Q) data set


    # 4, NN architecture (along with  Inkspace plot)

    # 5. distribution of latent space ? needed?

    # 6, MSE analysis of generative model vs. PY model with and without beta correction
    plot_gen_vs_PY_MSE()

    # 7. generative model vs PY model (with and without beta correction) I(Q) plot, illustrative
    #plot_gen_vs_PY_Iq()

    # 8, regressing using generative model






if __name__ == "__main__":
    main()