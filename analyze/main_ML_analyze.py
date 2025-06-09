from ML_analyze import *


def main():
    print("analyzing data ing......")
    folder = "../data/20250606"
    finfos = []
    L = 18
    pdType = 2
    for run in range(1014):
        finfos.append(f"L_{L:.0f}_pdType_{pdType:.0f}_run_{run:.0f}")

    svd_analysis(folder, finfos)

if __name__ == "__main__":
    main()
