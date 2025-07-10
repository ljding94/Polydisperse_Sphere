from ML_analyze import *


def main():
    print("analyzing data ing......")
    #folder = "../data/20250606"
    folder = "../data/20250615" # pdType 1 data
    #folder = "../data/20250623" # pdType 2 data
    #folder = "../data/20250628" # pdType 3 data
    finfos = []
    L = 18
    pdType = 1
    for run in range(6144):
        finfos.append(f"L_{L:.0f}_pdType_{pdType:.0f}_run_{run:.0f}")

    svd_analysis(folder, finfos, max_nfiles=5000)

    #wrap_Iq_data(folder, finfos, label=f"L_{L:.0f}_pdType_{pdType:.0f}", train_perc=0.8, max_nfiles=5000)
if __name__ == "__main__":
    main()
