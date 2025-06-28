import numpy as np
from analyze import *
from analyze import plot_Iq_prec_by_param, compare_Iq_by_param


def main():
    """
    Main function to run the analysis.
    """
    if 0:
        folder = "../data/20250613"
        L = 18
        params = []
        for pdType in [1]:
            for eta in np.arange(0.05, 0.51, 0.05):
                for sigma in np.arange(0.00, 0.151, 0.01):
                    params.append((L, pdType, eta, sigma))
        print(params.__len__())
        #plot_Iq_prec(folder, params, 6)
        plot_Iq_prec_by_param(folder, params)
        return 0

    if 1:
        folder = "../data/20250613"
        params = []
        L = 18
        #for pdType in [1,2,3]:
        for pdType in [3]:
            for eta in np.arange(0.05, 0.51, 0.05):
                #for sigma in [0.0, 0.10]:
                for sigma in np.arange(0.00, 0.151, 0.01):
                    params.append((L, pdType, eta, sigma))
        print(params.__len__())

        compare_Iq_by_param(folder, params, 0, label="check")


if __name__ == "__main__":
    main()
