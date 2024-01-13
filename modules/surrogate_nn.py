# load necessary modules
import numpy as np 
import os, sys, time
from pathlib import Path
from os.path import dirname, realpath
script_dir = Path(dirname(realpath('.')))
module_dir = str(script_dir)
sys.path.insert(0, module_dir + '/modules')
import utility as ut
import sample as sm
# from scipy.linalg import eigh
import pandas as pd
import json
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter 
import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as tfn
import torch.optim.lr_scheduler as lr_scheduler

torch.set_default_dtype(torch.float32)


class SurrogateModel_NN:
    """
    Description: A class for computing NN surrogate models for deterministic dynamical systems
    Args: 
            D: dimension of the underlying dynamical system
            D_r: dimension of the reservoir
            W_in_fn: a function for generating W_in
            b_in_fn: a function for generating b_in 
            W: value of the W matrix, default = None
            name: name of the surrogate model
            save_folder: folder for saving train log and parameters
        """
    def __init__(self, D, D_r, name='nn_model', save_folder='.'):        
        self.D = D
        self.D_r = D_r
        self.device = ("cuda"
                    if torch.cuda.is_available()
                    else "mps"
                    if torch.backends.mps.is_available()
                    else "cpu")
        self.net = nn.Sequential(nn.Linear(D, D_r, bias=True), nn.Tanh(), nn.Linear(D_r, D, bias=False))
        self.name = name
        self.save_folder = save_folder
        if not os.path.exists(self.save_folder):
            os.makedirs(self.save_folder)



    def forecast(self, u):
        """
        Description: forecasts for a single time step

        Args:
            u: state at current time step 

        Returns: forecasted state
        """
        return self.net(torch.from_numpy(u)).detach().numpy()
    
    
    
    def multistep_forecast(self, u, n_steps):
        """
        Description: forecasts for multiple time steps

        Args:
            u: state at current time step 
            n_steps: number of steps to propagate u

        Returns: forecasted state
        """    
        trajectory = np.zeros((self.D, n_steps))
        trajectory[:, 0] = u
        for step in range(n_steps - 1):
            u = self.forecast(u)
            trajectory[:, step+1] = u
        return trajectory

    def loss_fn(self, x, y, beta):
        return torch.norm(self.net(x) - y)**2 + beta*torch.norm(self.net.state_dict()['2.weight'])**2
    


    @ut.timer
    def learn(self, trajectory, epochs=100, learning_rate=1e-4, beta=4e-5, log_interval=100, save_interval=100,\
              milestones=[1000, 2000], drop=0.1):
        train_size = trajectory.shape[1]
        self.train_log = self.save_folder + '/train_log.csv'
        self.config = self.save_folder + '/config.json'
        log_row = {'iteration': [], 'loss': [], 'runtime': []}
        config = {'device': self.device, 'epochs': epochs, 'initial_rate':learning_rate,\
                   'milestones': milestones, 'drop': drop, 'train_size': train_size, 'D':self.D,\
                   'D_r':self.D_r, 'name': self.name, 'beta': beta}
        with open(self.config, 'w') as fp:
            json.dump(config, fp)
        if not os.path.exists(self.train_log):
            pd.DataFrame(log_row).to_csv(self.train_log, index=None)
        self.beta = beta
        x = torch.tensor(trajectory.T[:-1])
        y = torch.tensor(trajectory.T[1:])
        # Set the model to training mode - important for batch normalization and dropout layers
        # Unnecessary in this situation but added for best practices
        self.net.train()
        optimizer = torch.optim.Adam(self.net.parameters(), lr=learning_rate)
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=drop)
        start = time.time()

        for epoch in range(epochs):
            # Compute prediction and loss
            optimizer.zero_grad()
            loss = self.loss_fn(x, y, beta)
            # Backpropagation
            loss.backward()
            optimizer.step()
            scheduler.step()
                

            if epoch % log_interval == 0:
                loss = loss.item()
                end = time.time()
                print(f"epoch: {epoch}    loss: {loss:>7f}     time elapsed={end-start:.4f}")
                log_row['iteration'] = epoch
                log_row['loss'] = [loss]
                log_row['runtime'] = [end-start]
                pd.DataFrame(log_row).to_csv(self.train_log, mode='a', index=False, header=False)
            
            if epoch % save_interval == 0:
                torch.save(self.net, self.save_folder + f'/{self.name}_{epoch}')
                

    
    @ut.timer
    def compute_tau_f(self, test, error_threshold=0.05, dt=0.02, Lyapunov_time=1/0.91):
        """
        Description: computes forecast time tau_f for the computed surrogate model

        Args:
            test: list of test trajectories
        """
        tau_f_se, tau_f_rmse = np.zeros(len(test)), np.zeros(len(test))
        self.validation_points = test.shape[-1]
        self.error_threshold = error_threshold
        self.dt = dt
        self.Lyapunov_time = Lyapunov_time
        se, rmse = np.zeros(len(test)), np.zeros(len(test))
        for validation_index in range(len(test)):
            validation_ = test[validation_index]
            prediction = self.multistep_forecast(validation_[:, 0], self.validation_points)
            se_ = np.linalg.norm(validation_ - prediction, axis=0)**2 / np.linalg.norm(validation_, axis=0)**2
            mse_ = np.cumsum(se_) / np.arange(1, len(se_)+1)
    
            
            l = np.argmax(mse_ > self.error_threshold)
            if l == 0:
                tau_f_rmse[validation_index] = self.validation_points
            else:
                tau_f_rmse[validation_index] = l-1


            l = np.argmax(se_ > self.error_threshold)
            if l == 0:
                tau_f_se[validation_index] = self.validation_points
            else:
                tau_f_se[validation_index] = l-1
            
            rmse[validation_index] = np.sqrt(mse_[-1])
            se[validation_index] = se_.mean()
 
            
        
        tau_f_rmse *= (self.dt / self.Lyapunov_time)
        tau_f_se *= (self.dt / self.Lyapunov_time)

        return tau_f_rmse, tau_f_se, rmse, se




class SurrogateModel_NN_multi(SurrogateModel_NN):
    def __init__(self, D, D_r):
        super().__init__(D, D_r)
        self.net = nn.Sequential(nn.Linear(D, D_r, bias=True), nn.Tanh(),\
                                 nn.Linear(D_r, D_r, bias=True), nn.Tanh(),\
                                 nn.Linear(D_r, D_r, bias=True), nn.Tanh(),\
                                 nn.Linear(D_r, D_r, bias=True), nn.Tanh(),\
                                 nn.Linear(D_r, D, bias=False))
    def loss_fn(self, x, y, beta):
        return torch.norm(self.net(x) - y)**2 + beta*torch.norm(self.net.state_dict()['8.weight'])**2


