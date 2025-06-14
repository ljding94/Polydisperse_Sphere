import numpy as np
from analyze import *


def main():
    """
    Main function to run the analysis.
    """
    if 0:
        folder = "../data/20250602"
        params = []
        L = 18
        for pdType in [1,2,3]:
            for N in np.arange(0.05, 0.61, 0.05):
                for sigma in np.arange(0.00, 0.151, 0.01):
                    params.append((L, pdType, N, sigma))
        print(params.__len__())
        #plot_Iq_prec(folder, params, 6)
        plot_Iq_prec_by_param(folder, params, 15)
        #plot_Iq_prec_by_param_1(folder, params, 6)

    if 0:
        folder = "../data/20250601"
        params = []
        for L in [5,10,15,20]:
            for pdType in [2]:
                for N in np.arange(0.05, 0.61, 0.05):
                    #for sigma in np.arange(0.00, 0.151, 0.03):
                    for sigma in [0.0]:
                        params.append((L, pdType, N, sigma))
        print(params.__len__())
        #plot_Iq_prec(folder, params, 6)
        plot_Iq_prec_by_param(folder, params, 0)
        #plot_Iq_prec_by_param_1(folder, params, 6)


    if 1:
        folder = "../data/20250602"
        params = []
        L = 18
        for pdType in [2]:
            for N in np.arange(0.10, 0.61, 0.10):
                #for sigma in np.arange(0.00, 0.151, 0.01):
                sigma = 0.01
                params.append((L, pdType, N, sigma))
        print(params.__len__())
        #plot_Iq_prec(folder, params, 6)
        #plot_Iq_prec_by_param(folder, params, 15)
        #plot_Iq_prec_by_param_1(folder, params, 6)
        #compare_Iq_by_param(folder, params, 15, label="N")
        fit_Iq_by_param(folder, params, 15, label="N")

        params = []
        for pdType in [2]:
            N = 0.4
            for sigma in np.arange(0.00, 0.151, 0.03):
                params.append((L, pdType, N, sigma))
        print(params.__len__())
        #plot_Iq_prec(folder, params, 6)
        #plot_Iq_prec_by_param(folder, params, 15)
        #plot_Iq_prec_by_param_1(folder, params, 6)
        #compare_Iq_by_param(folder, params, 15, label="sigma")
        fit_Iq_by_param(folder, params, 15, label="sigma")

if __name__ == "__main__":
    main()
