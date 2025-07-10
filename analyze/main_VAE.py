# TODO: copied from semiflexible polymer codebase, need to be modified
# analyze 2D vol surface using VAE
from VAE_model import *
from torch.utils.data import Subset
import time

# TODO: need to modify for the Iq analysis
def main():
    folder = "../data/data_pack"  # for polymer data
    #folder = "../pd_data" # for neutron
    ld = 3
    for label in ["L_18_pdType_1", "L_18_pdType_2"]:
        print(f"Processing label: {label}")

        if 1:
            train_and_save_VAE_alone(folder, label, latent_dim=ld, num_epochs=1000)

        if 1:
            train_and_save_generator(folder, label, vae_path=f"{folder}/{label}_vae_state_dict.pt", input_dim=2, latent_dim=ld, num_epochs=300, fine_tune_epochs=300)

        if 1:
            train_and_save_inferrer(folder, label, vae_path=f"{folder}/{label}_vae_state_dict.pt", input_dim=2, latent_dim=ld, num_epochs=300, fine_tune_epochs=200)

        visualize_param_in_latent_space(f"{folder}/{label}_vae_state_dict.pt", folder, label, latent_dim=ld, save_path=f"{folder}/{label}_latent_distribution.png")

        plot_loss_curves(folder, label)

        show_vae_random_reconstructions(folder, label, f"{folder}/{label}_vae_state_dict.pt", latent_dim=ld)

        show_gen_random_reconstruction(folder, label, f"{folder}/{label}_gen_state_dict.pt", latent_dim=ld)

        show_inf_random_analysis(folder, label, f"{folder}/{label}_inf_state_dict.pt", latent_dim=ld)

if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    print(f"Total runtime: {end_time - start_time:.2f} seconds")
