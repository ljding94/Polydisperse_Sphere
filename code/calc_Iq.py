import numpy as np
from scipy.spatial.distance import pdist
import argparse
import sys
import time


def read_file(filename):
    data = np.loadtxt(filename, skiprows=9, delimiter=" ")
    x, y, z, diam = data[:, 2], data[:, 3], data[:, 4], data[:, 5]
    return x, y, z, diam


def calc_Pq(q, diam):
    radius = diam * 0.5
    Pq = 3 * (np.sin(q * radius) - q * radius * np.cos(q * radius)) / (q * radius) ** 3
    return Pq


def calc_Iq(filename, qmin=0.1, qmax=10.0, nq=100):
    x, y, z, diam = read_file(filename)
    V = diam**3 * np.pi / 6
    q_vals = np.linspace(qmin, qmax, nq)
    Pq = []
    # find Pq of each particle basd on the distance
    for i in range(len(x)):
        Pq.append(calc_Pq(q_vals, diam[i]))

    # find the 3D Iq from Aq
    N_theta = 50  # number of latitude bands
    N_phi = 100  # number of longitudes

    u = (np.arange(N_theta) + 0.5) / N_theta
    qtheta_vals = np.arccos(1 - 2 * u)
    qphi_vals = 2 * np.pi * np.arange(N_phi) / N_phi

    sum_V2 = np.sum(V)**2
    A_Re_q2 = np.zeros(len(q_vals))
    A_Im_q2 = np.zeros(len(q_vals))
    for qtheta in qtheta_vals:
        for qphi in qphi_vals:
            qx = q_vals * np.sin(qtheta) * np.cos(qphi)
            qy = q_vals * np.sin(qtheta) * np.sin(qphi)
            qz = q_vals * np.cos(qtheta)
            A_Re_theta_phi = np.zeros(len(q_vals))
            A_Im_theta_phi = np.zeros(len(q_vals))
            for i in range(len(x)):
                A_Re_theta_phi += V[i] * Pq[i] * np.cos(qx * x[i] + qy * y[i] + qz * z[i])
                A_Im_theta_phi += V[i] * Pq[i] * np.sin(qx * x[i] + qy * y[i] + qz * z[i])
            A_Re_q2 += A_Re_theta_phi*A_Re_theta_phi
            A_Im_q2 += A_Im_theta_phi*A_Im_theta_phi
            print(f"Processing qtheta: {qtheta:.2f}/{qtheta_vals[-1]}, qphi: {qphi:.2f}/{qphi_vals[-1]}")

    A_Re_q2 /= N_theta * N_phi
    A_Im_q2 /= N_theta * N_phi
    Iq_vals = (A_Re_q2 + A_Im_q2) / sum_V2
    return q_vals, Iq_vals


def write_Iq(filename, q_vals, Iq_vals):
    with open(filename, "w", newline="") as f:
        f.write("q,I(q)\n")
        for q, Iq in zip(q_vals, Iq_vals):
            f.write(f"{q:.6f},{Iq:.6f}\n")
    print(f"Data written to {filename}")


def main():
    if len(sys.argv) != 5:
        print("Usage: python calc_Iq.py pdTpe N sigma folder")
        sys.exit(1)
    pdTpe = int(sys.argv[1])
    N = float(sys.argv[2])
    sigma = float(sys.argv[3])
    folder = sys.argv[4]

    # Construct the input filename based on the provided arguments.
    input_filename = f"{folder}/pdType_{pdTpe}_N_{N}_sigma_{sigma}/dump.000008000.txt"
    print(f"Processing file: {input_filename}")

    # Calculate I(q) values using the input filename.
    q_vals, Iq_vals = calc_Iq(input_filename)

    # Construct an output filename.
    output_filename = input_filename.replace(".txt", "_Iq.txt")
    write_Iq(output_filename, q_vals, Iq_vals)


if __name__ == "__main__":
    start_time = time.time()
    main()
    print(f"Running time: {time.time() - start_time:.2f} seconds")
