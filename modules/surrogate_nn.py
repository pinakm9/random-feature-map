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
import surrogate as sr
import json
import sample as sm
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter 
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, RandomSampler
import torch.nn.functional as tfn
import torch.optim.lr_scheduler as lr_scheduler
from collections import OrderedDict
from learning_rate import AdaptiveRateBS

DTYPE = 'float64'
torch.set_default_dtype(torch.float64)
INFINITY = int(1e100)

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
    def __init__(self, D, D_r, name='nn', save_folder='.'):        
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
        self.config = {'name': name}



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
        return torch.sum((self.net(x) - y)**2) + beta*torch.sum(self.net[2].weight**2)
    

    def init_with_rf(self, L0, L1, beta, train, partition):
        self.sampler = sm.MatrixSampler(L0, L1, train.T)
        W_in, b_in = self.sampler.sample_(partition)
        rf = sr.SurrogateModel_LR(self.D, self.D_r, W_in, b_in)
        rf.compute_W(train, beta=beta)
        W_in = torch.Tensor(rf.W_in.astype(DTYPE))
        b_in = torch.Tensor(rf.b_in.astype(DTYPE))
        W = torch.Tensor(rf.W.astype(DTYPE))
        new_state_dict = OrderedDict({'0.weight': W_in, '0.bias': b_in, '2.weight': W})
        self.net.load_state_dict(new_state_dict, strict=False)
        self.config |= {'L0': L0, 'L1': L1, 'partition': partition}


    @ut.timer
    def learn(self, trajectory, steps=100, learning_rate=1e-4, beta=4e-5, log_interval=100, save_interval=100,\
              batch_size=1, **rate_params):
        train_size = trajectory.shape[1] - 1
        self.train_log = self.save_folder + '/train_log.csv'
        self.config_file = self.save_folder + '/config.json'
        log_row = {'iteration': [], 'loss': [], 'learning_rate': [], 'runtime': [], 'change': []}
        self.config |= {'device': self.device, 'steps': steps, 'initial_rate':learning_rate,\
                    'train_size': train_size, 'D':self.D,\
                   'D_r':self.D_r, 'name': self.name, 'beta': beta, 'batch_size': batch_size, 'log_interval':log_interval,\
                    'save_interval': save_interval}
        self.config |= rate_params
        with open(self.config_file, 'w') as fp:
            json.dump(self.config, fp)
        if not os.path.exists(self.train_log):
            pd.DataFrame(log_row).to_csv(self.train_log, index=None)
        self.beta = beta
        x = torch.Tensor(trajectory.T[:-1])
        y = torch.Tensor(trajectory.T[1:])
        
        if batch_size != 'GD':
            dataset = TensorDataset(x, y)
            sampler = RandomSampler(dataset, replacement=False, num_samples=INFINITY)
            self.dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size)
        # Set the model to training mode - important for batch normalization and dropout layers
        # Unnecessary in this situation but added for best practices
        self.net.train()
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=learning_rate)
        # scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=drop)
        lr_scheduler = AdaptiveRateBS(self, **rate_params)
        start = time.time()
        last_loss = np.inf
        step = 0
        while step < steps:
            # Compute prediction and loss
            self.optimizer.zero_grad()
            if batch_size != 'GD':
                x, y = next(self.dataloader.__iter__())
            loss = self.loss_fn(x, y, beta)
            # Backpropagation
            loss.backward()
            self.optimizer.step()
                
            # log training 
            if step % log_interval == 0: 
                loss_ = loss.item()
                end = time.time()
                lr = self.optimizer.param_groups[0]['lr']
                change = (loss_ - last_loss) / last_loss
                log_row['iteration'] = step
                log_row['loss'] = [loss_]
                log_row['learning_rate'] = [lr]
                log_row['change'] = [change]
                log_row['runtime'] = [end-start]
                print(f"step: {step} loss: {loss_:>7f} time: {end-start:.4f} lr: {lr:.6f},  change: {change*100:.2f}%")
                pd.DataFrame(log_row).to_csv(self.train_log, mode='a', index=False, header=False)
                
                if step % lr_scheduler.update_frequency == 0 and change > lr_scheduler.min_change:
                    if change < 0.:
                        sign = +1.
                    else:
                        sign = -1.
                    lr_scheduler.step(sign, last_loss, x, y, beta)
                else:
                    last_loss = loss_ + 0.
            
            if step % save_interval == 0:
                torch.save(self.net, self.save_folder + f'/{self.name}_{step}')
            
            
            step += 1
               
            

            
      


    
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
    

    def get_config(self):
        self.config = json.loads(self.save_folder + '/config.json')
    

    def get_save_idx(self):
        return sorted([int(f.split('_')[-1]) for f in os.listdir(self.save_folder) if f.startswith(self.name)])
    
    def load(self, idx):
        self.net = torch.load(self.save_folder + f'/{self.name}_{idx}')

    
    @ut.timer
    def count_row_types(self, m, M, data):
        idx = self.get_save_idx()
        df = pd.read_csv(self.save_folder + '/train_log.csv')
        good_rows_W_in = np.full(len(df['iteration']), np.nan)
        linear_rows_W_in = np.full(len(df['iteration']), np.nan)
        extreme_rows_W_in = np.full(len(df['iteration']), np.nan)
        gs = sm.GoodRowSampler(m, M, data)
        ls = sm.BadRowSamplerLinear(m, data)
        es = sm.BadRowSamplerExtreme(M, data)
        log_interval = df['iteration'][1] - df['iteration'][0] 
        for i in idx:
            self.load(i)
            rows = self.net.state_dict()['0.weight'].numpy()
            bs = self.net.state_dict()['0.bias'].numpy()
            j = int(i / log_interval)
            good_rows_W_in[j] = gs.are_rows(rows, bs).sum()
            linear_rows_W_in[j] = ls.are_rows(rows, bs).sum()
            extreme_rows_W_in[j] = es.are_rows(rows, bs).sum()
        df['good_rows_W_in'] = good_rows_W_in / float(self.D_r)
        df['linear_rows_W_in'] = linear_rows_W_in /  float(self.D_r)
        df['extreme_rows_W_in'] = extreme_rows_W_in /  float(self.D_r)
        df.to_csv(f'{self.save_folder}/train_log.csv', index=None) 

        
    def compute_training_error(self, train, train_index, length):
        """
        Description: computes forecast time tau_f for the computed surrogate model

        Args:
            train: training data
            W_norm: norm of W
        """

        prediction = self.multistep_forecast(train[:, train_index], length)
        train_sq_err = (np.linalg.norm(train[:, train_index:train_index+length] - prediction, axis=0)**2 / np.linalg.norm(train[:, train_index:train_index+length], axis=0)**2).mean()
        return train_sq_err


    @ut.timer
    def compute_train_error(self, train, train_index, length):
        idx = self.get_save_idx()
        df = pd.read_csv(self.save_folder + '/train_log.csv')
        train_sq_err = np.full(len(df['iteration']), np.nan)
        log_interval = df['iteration'][1] - df['iteration'][0] 
        for i in idx:
            self.load(i)
            j = int(i / log_interval)
            train_sq_err[j] = self.compute_training_error(train, train_index, length)
            # print(train_sq_err[j])
   
        df['train_sq_err'] = train_sq_err
        df.to_csv(f'{self.save_folder}/train_log.csv', index=None) 




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


