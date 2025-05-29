import numpy as np
from analyze import *


def main():
    """
    Main function to run the analysis.
    """
    folder = "../data/20250526"
    params = []
    for pdType in [1]:
        for N in np.arange(0.05, 0.61, 0.05):
            for sigma in np.arange(0.00, 0.151, 0.01):
                params.append((pdType, N, sigma))
    print(params.__len__())
    #plot_Iq_prec(folder, params, 6)
    plot_Iq_prec_by_param(folder, params, 0)
    #plot_Iq_prec_by_param_1(folder, params, 6)


if __name__ == "__main__":
    main()
