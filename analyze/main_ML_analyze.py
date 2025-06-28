from ML_analyze import *


def main():
    print("analyzing data ing......")
    #folder = "../data/20250606"
    #folder = "../data/20250615"
    folder = "../data/20250623"
    finfos = []
    L = 18
    pdType = 2
    for run in range(6144):
        finfos.append(f"L_{L:.0f}_pdType_{pdType:.0f}_run_{run:.0f}")

    svd_analysis(folder, finfos, max_nfiles=5000)

    wrap_Iq_data(folder, finfos, label=f"L_{L:.0f}_pdType_{pdType:.0f}")

if __name__ == "__main__":
    main()
