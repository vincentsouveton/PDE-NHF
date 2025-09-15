# IMPORT LIBRARIES

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
from torch.autograd import grad
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
import argparse




class Parser:
    def __init__(self):
        """ inits Parser with the arguments to take into account"""
        self.parser = argparse.ArgumentParser(description='Training PDE-NHF on 1D data.')

        self.parser.add_argument('-SEED', default = 1, type = int, dest = 'SEED',
                                help='Random seed.') 
        self.parser.add_argument('-FOLDER_EXP', default = 'models/model/', type = str, dest = 'FOLDER_EXP',
                               help='Folder in which model is saved.')
        self.parser.add_argument('-FOLDER_DATA', default = 'data/', type = str, dest = 'FOLDER_DATA',
                               help='Folder in which data are saved.')
        self.parser.add_argument('-N_TRAINING', default = 20000, type = int, dest = 'N_TRAINING',
                                help='Number of training examples.')
        self.parser.add_argument('-N_VALIDATION', default = 6384, type = int, dest = 'N_VALIDATION',
                                help='Number of validation examples.')
        self.parser.add_argument('-L', default = 25, type = int, dest = 'L',
                                help='Number of steps in Leapfrog integrator.')
        self.parser.add_argument('-DT', default = 0.04, type = float, dest = 'DT',
                               help='Timestep in Leapfrog integrator.')
        self.parser.add_argument('-N_EPOCHS', default = 200, type = int, dest = 'N_EPOCHS',
                                help='Number of training epochs.')
        self.parser.add_argument('-BATCH_SIZE', default = 128, type = int, dest = 'BATCH_SIZE',
                                help='Minibatch size during training.')
        self.parser.add_argument('-LR', default = 0.0003, type = float, dest = 'LR',
                               help='Learning rate for Adam optimizer.')
        self.args = self.parser.parse_args()


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    parser = Parser()    
    

    # LOAD DATA, DEFINE FOLDER, DEFINE HYPERPARAMETERS

    # Hyperparameters
    random_seed = parser.args.SEED
    N = parser.args.N_TRAINING # number of examples in training dataset
    N2 = parser.args.N_VALIDATION # number of examples in validation dataset
    learning_rate = parser.args.LR # learning rate for Adam optimizer
    num_epochs = parser.args.N_EPOCHS # number of training epochs
    batch_size = parser.args.BATCH_SIZE # minibatch size during training 
    folder_exp = parser.args.FOLDER_EXP # folder in which model is saved
    folder_data = parser.args.FOLDER_DATA # folder in which data should be loaded

    Q = np.load(folder_data+'Q25.npy')
    P = np.load(folder_data+'P25.npy')
    Cond = np.load(folder_data+'Cond.npy')
    Q_train_numpy, P_train_numpy, Cond_train_numpy = Q[0:N], P[0:N], Cond[0:N]
    Q_val_numpy, P_val_numpy, Cond_val_numpy = Q[N:N+N2], P[N:N+N2], Cond[N:N+N2]

    Q_train = torch.tensor(Q_train_numpy,dtype=float)
    Q_val = torch.tensor(Q_val_numpy,dtype=float)
    P_train = torch.tensor(P_train_numpy,dtype=float)
    P_val = torch.tensor(P_val_numpy,dtype=float)
    Cond_train = torch.tensor(Cond_train_numpy,dtype=float)
    Cond_val = torch.tensor(Cond_val_numpy,dtype=float)

    class QPCondDataset(Dataset):
        def __init__(self, q_maps, p_maps, conds, transform=None):
            self.q_maps = q_maps
            self.p_maps = p_maps
            self.conds = conds
            self.transform = transform

        def __len__(self):
            return len(self.q_maps)

        def __getitem__(self, idx):
            q = self.q_maps[idx]
            p = self.p_maps[idx]
            cond = self.conds[idx]

            if self.transform:
                q = self.transform(q)
                p = self.transform(p)

            return q, p, cond


    training_dataset = QPCondDataset(Q_train, P_train, Cond_train) # Training dataset
    validation_dataset = QPCondDataset(Q_val, P_val, Cond_val) # Validation dataset




    # DEFINE MODEL

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Hopefully runs on GPU

    class Potential(nn.Module):
        def __init__(self, hidden_dim=256):
            super().__init__()
            self.phi = nn.Sequential(
                nn.Linear(1, hidden_dim),
                nn.Softplus(),
                nn.Linear(hidden_dim, hidden_dim)
            )
            self.rho = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.Softplus(),
                nn.Linear(hidden_dim, 1)
            )

        def forward(self, q):  # q: (B, N)
            q_centered = q - q.mean(dim=1, keepdim=True)  
            phi_q = self.phi(q_centered.unsqueeze(-1))   
            pooled = phi_q.sum(dim=1)                     
            return self.rho(pooled).squeeze(-1)           


    class NeuralHamiltonianFlow(nn.Module):
        def __init__(self, L, dt):
            super().__init__()
            self.L = L
            self.dt = dt
            self.V_net = Potential()  # Scalar potential energy
            self.register_parameter(name='a', param=torch.nn.Parameter(1.0*torch.ones(1))) # mass matrix scalar coefficient as M^-1 = a^2 Id

        def potential_energy(self, q):
            return self.V_net(q)

        def leapfrog_integrator(self, q, p, L, dt):
            # Compute initial grad_y (gradient of V)
            V = self.potential_energy(q)
            grad_q, = grad(V.sum(), q, create_graph=True)

            for step in range(L):
                # Half-step for momentum (kick)
                p = p - 0.5 * dt * grad_q

                # Full-step for position (drift)
                q = q + self.a**2 * p * dt
                #q = q + p * dt

                # Compute new grad_y for the next iteration
                V = self.potential_energy(q)
                grad_q, = grad(V.sum(), q, create_graph=True)

                # Final half-step for momentum (kick)
                p = p - 0.5 * dt * grad_q

            return q, p

        def forward(self, q, p, cond):
            sigma_q, sigma_p = cond[:,0].unsqueeze(1), cond[:,1].unsqueeze(1)

            q.requires_grad, p.requires_grad = True, True  # Ensure gradients

            # Perform Leapfrog steps
            q, p = self.leapfrog_integrator(q, p, self.L, self.dt)
            return q, p, sigma_q, sigma_p

        def loss(self, q, p, cond):
            q0, p0, sigma_q, sigma_p = self.forward(q, p, cond)

            # Prior: Negative log of Gaussian base distribution
            # Create a standard normal distribution
            prior_q = D.Normal(loc=64.0, scale=sigma_q)
            prior_p = D.Normal(loc=0.0, scale=sigma_p)
            log_pi_q0 = (prior_q.log_prob(q0)).sum(dim=1)
            log_pi_p0 = (prior_p.log_prob(p0)).sum(dim=1)  

            # KL Loss
            return -(log_pi_q0 + log_pi_p0).mean()

        def sample(self, q0, p0, nsteps, delta_t):
            q0.requires_grad, p0.requires_grad = True, True
            q0, p0 = q0.unsqueeze(0), p0.unsqueeze(0)

            q, p = self.leapfrog_integrator(q0, p0, nsteps, -delta_t)
            return q.detach(), p.detach()


    set_seed(random_seed)

    L = parser.args.L  # Number of Leapfrog steps
    dt = -parser.args.DT # Negative for training (reversed Hamiltonian dynamics)
    model = NeuralHamiltonianFlow(L=L, dt=dt)
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("The model is made of " + str(n_parameters) + " parameters.")
    model.to(device)
    model.double()



    # TRAINING PHASE

    # Initialize optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Create dataloader
    train_loader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=True)

    # Training loop
    training_loss = []
    validation_loss = []

    for epoch in range(num_epochs):
        total_loss = 0
        model.train()
        for q, p, cond in train_loader:
            q, p, cond = q.to(device), p.to(device), cond.to(device)
            loss = model.loss(q, p, cond) # Calculate training loss
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        training_loss.append(float(total_loss/len(train_loader)))
        print("Epoch " + str(epoch+1))
        print(float(total_loss/len(train_loader)))
        q, p, cond = next(iter(valid_loader))
        q, p, cond = q.to(device), p.to(device), cond.to(device)
        vloss = model.loss(q, p, cond) # Calculate validation loss
        validation_loss.append(float(vloss))
        if (epoch+1)%50==0: 
            torch.save(model.state_dict(), folder_exp+'model-'+str(epoch+1)) # Save trained model
            np.save(folder_exp+'training_loss'+str(epoch+1), np.array(torch.tensor(training_loss).to("cpu"))) # Save training loss values
            np.save(folder_exp+'validation_loss'+str(epoch+1), np.array(torch.tensor(validation_loss).to("cpu"))) # Save validation loss values
            torch.save(optimizer.state_dict(), folder_exp + 'optimizer'+str(epoch+1)) # Save optimizer state
        print(q.shape)

    np.save(folder_exp+'training_loss_final', np.array(torch.tensor(training_loss).to("cpu"))) # Save final training loss values
    np.save(folder_exp+'validation_loss_final', np.array(torch.tensor(validation_loss).to("cpu"))) # Save final validation loss values
    torch.save(model.state_dict(), folder_exp+'model_final') # Save final trained model
    torch.save(optimizer.state_dict(), folder_exp + 'optimizer_final') # Save final optimizer state



