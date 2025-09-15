# IMPORT LIBRARIES
import numpy as np
import argparse
import random



class Parser:
    def __init__(self):
        """ inits Parser with the arguments to take into account"""
        self.parser = argparse.ArgumentParser(description='Generating 1D artificial data with PIC simulations.')

        self.parser.add_argument('-SEED', default = 1, type = int, dest = 'SEED',
                                help='Random seed.')        
        self.parser.add_argument('-FOLDER_DATA', default = 'data/', type = str, dest = 'FOLDER_DATA',
                               help='Folder in which data are saved.')
        self.parser.add_argument('-N_EXAMPLES', default = 32768, type = int, dest = 'N_EXAMPLES',
                                help='Number of examples to generate.')
        self.parser.add_argument('-N_PARTICLES', default = 256, type = int, dest = 'N_PARTICLES',
                                help='Number of particles on the grid per example.')
        self.parser.add_argument('-MIN_STD_Q', default = 0.5, type = float, dest = 'MIN_STD_Q',
                               help='Minimum of initial positions standard deviation.')
        self.parser.add_argument('-MAX_STD_Q', default = 1.5, type = float, dest = 'MAX_STD_Q',
                               help='Maximum of initial positions standard deviation.')
        self.parser.add_argument('-MIN_STD_P', default = 0.5, type = float, dest = 'MIN_STD_P',
                               help='Minimum of initial momenta standard deviation.')
        self.parser.add_argument('-MAX_STD_P', default = 1.5, type = float, dest = 'MAX_STD_P',
                               help='Maximum of initial momenta standard deviation.')
        self.args = self.parser.parse_args()



if __name__ == "__main__":
    parser = Parser()

    random_seed = parser.args.SEED
    folder_data = parser.args.FOLDER_DATA # folder in which to save the data 
    n = parser.args.N_EXAMPLES # Number of examples
    N = parser.args.N_PARTICLES  # Number of particles
    L = 128  # Length of the 1D domain
    Nx = 128  # Number of grid points
    dx = L/Nx # Grid spacing
    T = 1.0  # Total time
    dt = 0.04  # Time step
    q_p = 0.1 # Particles charge 

    np.random.seed(random_seed)

    # GENERATE ARTIFICIAL CONDITIONS
    min_mu_q, max_mu_q = 64.0, 64.0
    mu_q = np.random.uniform(low=min_mu_q, high=max_mu_q, size=(n,1))
    
    min_mu_p, max_mu_p = 0.0, 0.0
    mu_p = np.random.uniform(low=min_mu_p, high=max_mu_p, size=(n,1))
    
    min_std_q, max_std_q = parser.args.MIN_STD_Q, parser.args.MAX_STD_Q
    sigma_q = np.random.uniform(low=min_std_q, high=max_std_q, size=(n,1))
    
    min_std_p, max_std_p = parser.args.MIN_STD_P, parser.args.MAX_STD_P
    sigma_p = np.random.uniform(low=min_std_p, high=max_std_p, size=(n,1))
    
    Cond = np.concatenate((sigma_q, sigma_p), axis=1)
    np.save(folder_data+'Cond.npy', Cond)


    # FUNCTIONS FOR PIC SIMULATIONS
    # Normalize particle density later
    def normalize_density(density):
        total_charge = np.sum(density) * dx  # Charge is proportional to density
        return density / total_charge  # Normalize charge to 1

    # Compute the charge density rho(x) using particle positions
    def compute_charge_density(x):
        rho = np.zeros(Nx)

        for xi in x:
            i = int(xi / dx)
            frac = (xi - i * dx) / dx
            iL = i % Nx
            iR = (i + 1) % Nx
            rho[iL] += (1 - frac) * q_p
            rho[iR] += frac * q_p

        rho /= dx  # Convert to charge density
        rho -= np.mean(rho)  # Ensure neutrality

        return rho

    def compute_electric_field(rho):
        k = np.fft.fftfreq(Nx, d=dx) * 2 * np.pi

        # Forward FFT of charge density
        rho_k = np.fft.fft(rho)

        # Zero out the zero-frequency mode (net charge) for stability
        rho_k[0] = 0.0

        # Solve Poisson equation in Fourier space: -k² φ_k = -rho_k  ⇒  φ_k = rho_k / k²
        with np.errstate(divide='ignore', invalid='ignore'):
            phi_k = np.zeros_like(rho_k)
            nonzero_k = k != 0
            phi_k[nonzero_k] = rho_k[nonzero_k] / (k[nonzero_k]**2)

        # Compute electric field: E_k = -i k φ_k
        E_k = -1j * k * phi_k
        E_k[0] = 0.0  # Set explicitly, even though it's zero

        # Inverse FFT to get real-space field
        E = np.real(np.fft.ifft(E_k))

        return E


    # Function to interpolate electric field to particle positions
    def interpolate_field(x, E_grid):
        E_interp = np.zeros_like(x)
        for i, xi in enumerate(x):
            left = int(xi / dx) % Nx
            right = (left + 1) % Nx
            frac = (xi - left * dx) / dx
            E_interp[i] = (1 - frac) * E_grid[left] + frac * E_grid[right]
        return E_interp


    # LOOP OVER THE WHOLE DATASET TO GENERATE DATA
    Q00, P00 = np.zeros((n,N)), np.zeros((n,N))
    Q05, P05 = np.zeros((n,N)), np.zeros((n,N))
    Q10, P10 = np.zeros((n,N)), np.zeros((n,N))
    Q15, P15 = np.zeros((n,N)), np.zeros((n,N))
    Q20, P20 = np.zeros((n,N)), np.zeros((n,N))
    Q25, P25 = np.zeros((n,N)), np.zeros((n,N))

    for i in range(n):
        if (i+1)%500==0: print(str(i+1)+'/'+str(n))
        x = mu_q[i]*np.ones(N) + sigma_q[i]*np.random.randn(N)
        v = mu_p[i]*np.ones(N) + sigma_p[i]*np.random.randn(N)
        Q00[i] = x
        P00[i] = v
        # Compute initial electric field
        rho = compute_charge_density(x)
        E_grid = compute_electric_field(rho)
        E = interpolate_field(x, E_grid)
        #E = np.interp(x, x_grid, E_grid)

        # **First half-kick** (Kick)
        v += 0.5 * dt * E  

        # Main loop: (Drift -> Compute Field -> Kick)
        for step in range(1, 26):
            # **Drift step**
            x = (x + dt * v) % L  # Update positions with periodic boundary conditions

            # **Compute new electric field at updated positions**
            rho = compute_charge_density(x)
            E_grid = compute_electric_field(rho)
            E = interpolate_field(x, E_grid)
            #E = np.interp(x, x_grid, E_grid)

            # **Second half-kick**
            v += 0.5 * dt * E

            if step==5:
                Q05[i] = x
                P05[i] = v
            if step==10:
                Q10[i] = x
                P10[i] = v
            if step==15:
                Q15[i] = x
                P15[i] = v
            if step==20:
                Q20[i] = x
                P20[i] = v
            if step==25:
                Q25[i] = x
                P25[i] = v

    np.save(folder_data+'Q00.npy', Q00)
    np.save(folder_data+'P00.npy', P00)
    np.save(folder_data+'Q05.npy', Q05)
    np.save(folder_data+'P05.npy', P05)
    np.save(folder_data+'Q10.npy', Q10)
    np.save(folder_data+'P10.npy', P10)
    np.save(folder_data+'Q15.npy', Q15)
    np.save(folder_data+'P15.npy', P15)
    np.save(folder_data+'Q20.npy', Q20)
    np.save(folder_data+'P20.npy', P20)
    np.save(folder_data+'Q25.npy', Q25)
    np.save(folder_data+'P25.npy', P25)





