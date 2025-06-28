# TODO: copied from semiflexible polymer codebase, need to be modified
# analyze 2D vol surface using VAE
from VAE_model import *
from torch.utils.data import Subset
import time

# TODO: need to modify for the Iq analysis
def main():
    folder = "../data/20250615"
    label = "L_18_pdType_1"
    ld = 3

    if 0:
        train_and_save_VAE_alone(folder, label, latent_dim=ld, num_epochs=100)

        #visualize_latent_distribution(f"{folder}/{label}_vae_state_dict.pt", folder, label, latent_dim=ld, save_path=f"{folder}/{label}_latent_distribution.png")

    if 0:
        train_and_save_generator(folder, label, vae_path=f"{folder}/{label}_vae_state_dict.pt", input_dim=2, latent_dim=ld, num_epochs=50)

    plot_loss_curves(folder, label)

    show_vae_random_reconstructions(folder, label, f"{folder}/{label}_vae_state_dict.pt", latent_dim=ld)

    show_gen_random_reconstruction(folder, label, f"{folder}/{label}_gen_state_dict.pt", latent_dim=ld)


if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    print(f"Total runtime: {end_time - start_time:.2f} seconds")
