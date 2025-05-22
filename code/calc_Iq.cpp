#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <cmath>
#include <filesystem>
#include <regex>
#include <chrono> // for timing

namespace fs = std::filesystem;

struct Atom
{
    double x, y, z, diam;
};

// Function 1: read_atom_positions
std::vector<Atom> read_atom_positions(const std::string &dump_file)
{
    std::vector<Atom> atoms;
    std::ifstream infile(dump_file);
    if (!infile)
    {
        std::cerr << "Error opening file: " << dump_file << std::endl;
        return atoms;
    }

    std::string line;
    bool reading_atoms = false;

    while (std::getline(infile, line))
    {
        // Look for the header that indicates the start of atom data.
        if (line.find("ITEM: ATOMS") != std::string::npos)
        {
            reading_atoms = true;
            continue;
        }

        // Once the header is found, read each subsequent line as an atom's data.
        if (reading_atoms)
        {
            std::istringstream iss(line);
            int id, type;
            double x, y, z, diam;
            // Now expect: id, type, x, y, z, diam (and ignore the rest if present).
            if (!(iss >> id >> type >> x >> y >> z >> diam))
            {
                // Skip lines that don't match the expected format.
                continue;
            }
            atoms.push_back({x, y, z, diam});
        }
    }

    if (atoms.empty())
    {
        std::cerr << "No atom data found in file: " << dump_file << std::endl;
    }
    return atoms;
}
// function 2: calc Pq
std::vector<double> calc_Pq(Atom atom, std::vector<double> q_vec)
{
    std::vector<double> Pq(q_vec.size(), 0.0);
    // radius = diam * 0.5
    // Pq = 3 * (np.sin(q * radius) - q * radius * np.cos(q * radius)) / (q * radius) ** 3
    // return Pq
    double radius = atom.diam * 0.5;
    for (size_t i = 0; i < q_vec.size(); ++i)
    {
        double q = q_vec[i];
        if (q == 0.0)
        {
            Pq[i] = 1.0; // j0(0) = 1
        }
        else
        {
            double qr = q * radius;
            Pq[i] = (3.0 * (std::sin(qr) - qr * std::cos(qr))) / (qr * qr * qr);
        }
    }
    return Pq;
}

// Function 3: calc_Iq
// Given a dump file and a q vector, calculate S(q) for that file.
// For each atom, we compute r = Iqrt(x^2+y^2+z^2) and use j0(q*r) = sin(q*r)/(q*r) (with j0=1 for r=0).
// Returns a vector of S(q) values (one for each q in q_vec).
std::vector<double> calc_Iq(std::vector<Atom> atoms, std::vector<double> q_vec, int N_theta, int N_phi)
{
    // 1, find Pq for each atom
    std::vector<std::vector<double>> Pq(atoms.size());
    for (int i = 0; i < atoms.size(); ++i)
    {
        Pq[i] = calc_Pq(atoms[i], q_vec);
    }

    // Compute theta and phi values.
    std::vector<double> qtheta_vals, qphi_vals;
    for (int i = 0; i < N_theta; ++i)
    {
        double u = (i + 0.5) / static_cast<double>(N_theta);
        qtheta_vals.push_back(std::acos(1 - 2 * u));
    }
    for (int j = 0; j < N_phi; ++j)
    {
        double phi = 2.0 * M_PI * j / static_cast<double>(N_phi);
        qphi_vals.push_back(phi);
    }

    // Compute V for each atom assuming spherical volume: V = (pi/6)*diam^3
    std::vector<double> V(atoms.size(), 0.0);
    double sum_V2 = 0.0;
    for (size_t i = 0; i < atoms.size(); ++i)
    {
        V[i] = (M_PI / 6.0) * std::pow(atoms[i].diam, 3);
        sum_V2 += V[i]*V[i];
    }

    // Initialize the accumulators for the real and imaginary parts.
    std::vector<std::vector<std::vector<double>>> A_Re_q_theta_phi(q_vec.size(), std::vector<std::vector<double>>(N_theta, std::vector<double>(N_phi, 0.0)));
    std::vector<std::vector<std::vector<double>>> A_Im_q_theta_phi(q_vec.size(), std::vector<std::vector<double>>(N_theta, std::vector<double>(N_phi, 0.0)));
    std::vector<std::vector<std::vector<double>>> I_q_theta_phi(q_vec.size(), std::vector<std::vector<double>>(N_theta, std::vector<double>(N_phi, 0.0)));

    // Loop over all theta and phi values.
    for (int itheta = 0; itheta < N_theta; itheta++)
    {
        for (int iphi = 0; iphi < N_phi; iphi++)
        {
            // For each q value, compute the q-vector components.
            for (size_t iq = 0; iq < q_vec.size(); ++iq)
            {
                double q = q_vec[iq];
                double qx = q * std::sin(qtheta_vals[itheta]) * std::cos(qphi_vals[iphi]);
                double qy = q * std::sin(qtheta_vals[itheta]) * std::sin(qphi_vals[iphi]);
                double qz = q * std::cos(qtheta_vals[itheta]);

                // Sum over all atoms.
                for (size_t i = 0; i < atoms.size(); ++i)
                {
                    double phase = qx * atoms[i].x + qy * atoms[i].y + qz * atoms[i].z;
                    A_Re_q_theta_phi[iq][itheta][iphi] += V[i] * Pq[i][iq] * std::cos(phase);
                    A_Im_q_theta_phi[iq][itheta][iphi] += V[i] * Pq[i][iq] * std::sin(phase);
                }
                I_q_theta_phi[iq][itheta][iphi] = (A_Re_q_theta_phi[iq][itheta][iphi] * A_Re_q_theta_phi[iq][itheta][iphi] + A_Im_q_theta_phi[iq][itheta][iphi] * A_Im_q_theta_phi[iq][itheta][iphi])/(sum_V2);
            }
        }
    }
    // Compute 1D Iq .
    std::vector<double> Iq_vals(q_vec.size(), 0.0);
    for (int itheta = 0; itheta < N_theta; itheta++)
    {
        for (int iphi = 0; iphi < N_phi; iphi++)
        {
            for (size_t iq = 0; iq < q_vec.size(); ++iq)
            {
                Iq_vals[iq] += I_q_theta_phi[iq][itheta][iphi]/(N_theta * N_phi); // average over theta and phi
                // no need to sin(qtheta_vals[itheta]) here, because we have uniform cos(theta)
                //Iq_vals[iq] += I_q_theta_phi[iq][itheta][iphi]*std::sin(qtheta_vals[itheta])/(N_theta * N_phi); // average over theta and phi
            }
        }
    }
    return Iq_vals;
}
// Function 4: find dump-by-dump average Iq

std::vector<double> calc_average_Iq(const std::string &folder_path, const std::vector<double> &q_vec, int N_theta, int N_phi)
{
    std::vector<double> avg_Iq(q_vec.size(), 0.0);
    int file_count = 0;

    fs::path p(folder_path);
    // Regex for file names like: dump.000000000.txt
    std::regex pattern("^dump\\.[0-9]+\\.txt$");

    for (const auto &entry : fs::directory_iterator(p))
    {
        if (entry.is_regular_file())
        {
            std::string filename = entry.path().filename().string();
            if (std::regex_match(filename, pattern))
            {
                std::cout << "Processed file: " << filename << std::endl;
                std::string full_path = entry.path().string();
                std::vector<Atom> atoms = read_atom_positions(full_path);
                std::vector<double> file_Iq = calc_Iq(atoms, q_vec, N_theta, N_phi);
                // Check for error indicator (here if first value is -1.0, we skip)
                if (!file_Iq.empty() && file_Iq[0] < 0.0)
                    continue;
                for (size_t i = 0; i < avg_Iq.size(); ++i)
                {
                    avg_Iq[i] += file_Iq[i];
                }
                file_count++;
            }
        }
    }
    if (file_count == 0)
    {
        std::cerr << "No dump files matching the pattern were found in: " << folder_path << std::endl;
        return avg_Iq;
    }
    // Average the S(q) over all files.
    for (auto &val : avg_Iq)
    {
        val /= file_count;
    }
    std::cout << "Processed " << file_count << " files." << std::endl;
    return avg_Iq;
}

// Function 4: save_Iq
// Given a folder path, a q vector, and the averaged S(q) vector, save the data to an output file in the folder.
void save_Iq(const std::string &filename, const std::vector<double> &q_vec, const std::vector<double> &avg_Iq)
{
    std::ofstream ofs(filename);
    if (!ofs)
    {
        std::cerr << "Error opening output file: " << filename << std::endl;
        return;
    }

    // Write header lines (optional)
    ofs << "q";
    // Write q vector in one row
    for (size_t i = 0; i < q_vec.size(); ++i)
    {
        ofs << "," << q_vec[i];
    }
    ofs << "\n";
    ofs << "S(q)";
    // Write S(q) values in one row
    for (size_t i = 0; i < avg_Iq.size(); ++i)
    {
        ofs << "," << avg_Iq[i];
    }

    ofs.close();
    std::cout << "Averaged S(q) saved to " << filename << std::endl;
}
// Function 1: main
// Takes the input path (folder) as argument, constructs a q vector, calculates the average S(q),
// and then saves the averaged S(q) and q vector to an output file in the folder.
int main(int argc, char *argv[])
{
    if (argc != 2)
    {
        std::cerr << "Usage: " << argv[0] << " <folder_path>\n";
        return 1;
    }
    std::string folder_path = argv[1];

    std::string save_file = folder_path + "/Iq" + ".csv";
    std::string dump_folder = folder_path;

    // Start timer.
    auto start_time = std::chrono::steady_clock::now();

    // Define a q vector (for example, from 0.1 to 15.0 in 3 points on a log scale)
    int num_q = 100;
    int N_theta = 30;
    int N_phi = 60;
    std::vector<double> q_vec(num_q);
    // double qi = 1e-2;
    double qi = 2e-1;
    double qf = 13e0;
    for (int k = 0; k < num_q; k++)
    {
        //q_vec[k] = qi * std::pow(qf / qi, 1.0 * k / (num_q - 1)); // uniform in log scale;
        q_vec[k] = qi + (qf - qi) * k / (num_q - 1); // uniform in linear scale
    }

    // Calculate the average S(q) over all dump files in the folder.
    std::vector<double> avg_Iq = calc_average_Iq(dump_folder, q_vec, N_theta, N_phi);

    // Save the q vector and averaged S(q) to a file in the folder.
    save_Iq(save_file, q_vec, avg_Iq);

    // Stop timer.
    auto end_time = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed_seconds = end_time - start_time;
    std::cout << "Total running time: " << elapsed_seconds.count() << " seconds." << std::endl;

    return 0;
}