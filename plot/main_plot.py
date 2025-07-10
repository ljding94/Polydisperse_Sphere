from illustrate_plot import *
from NN_model_plot import *
from SVD_plot import *
def main():
    print("plotting figures for pd HS")

    # 1 illustrate 3 distribution
    #plot_three_distribution()

    # 2, illustrative system config of HS, and some I(Q) plot
    #plot_Iq_and_config()

    #plot_Iq_per_eta_sigma()

    # 3, SVD analysis of the I(Q) data set
    #plot_svd_analysis()
    #plot_svd_distribution()
    #plot_combine_svd()


    # 4, NN architecture (along with  Inkspace plot)
    # done

    # 5. distribution of latent space ? needed? (maybe not)
    plot_parm_in_latent_space()

    # 6. generative model vs PY model (with and without beta correction) I(Q) plot, illustrative
    #plot_gen_vs_PY_Iq()

    # 7, MSE analysis of generative model vs. PY model with and without beta correction
    #plot_gen_vs_PY_MSE()
    #plot_gen_vs_PY_MSE_uni()



    # 8, regressing using generative model
    #plot_inferrer_prediction()
    #plot_inferrer_prediction_uni()


    # 9. 10. other distribution
    #plot_gen_vs_PY_MSE_other()
    #plot_inferrer_prediction_other()




if __name__ == "__main__":
    main()