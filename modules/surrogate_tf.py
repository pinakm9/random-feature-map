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
import tensorflow as tf

# DTYPE = tf.float64

class SurrogateModel_NN:
    """
    Description: A class for computing NN surrogate models for deterministic dynamical systems
    Args: 
            D: dimension of the underlying dynamical system
            D_r: dimension of the reservoir
            W_in_fn: a function for generating W_in
            b_in_fn: a function for generating b_in 
            W: value of the W matrix, default = None
        """
    def __init__(self, D, D_r):        
        self.D = D
        self.D_r = D_r
        # self.device = ("cuda"
        #             if torch.cuda.is_available()
        #             else "mps"
        #             if torch.backends.mps.is_available()
        #             else "cpu")
        self.net = tf.keras.Sequential()
        self.net.add(tf.keras.layers.Dense(D_r, activation='tanh'))
        self.net.add(tf.keras.layers.Dense(D, activation=None))



    def forecast(self, u):
        """
        Description: forecasts for a single time step

        Args:
            u: state at current time step 

        Returns: forecasted state
        """
        return self.net(u.reshape(1, self.D)).numpy()[0]
    
    
    @tf.function
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

    def loss_fn(self, x, y):
        return tf.reduce_mean((self.net(x) - y)**2) #+ self.beta*torch.mean(self.net.state_dict()['2.weight']**2)
    
    @tf.function
    def train_step(self, x, y):
        with tf.GradientTape() as tape:
            loss = self.loss_fn(x, y)
        grads = tape.gradient(loss, self.net.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.net.trainable_weights))
        return loss


    @ut.timer
    def learn(self, trajectory, epochs=100, learning_rate=1e-4, beta=4e-5):
        self.beta = beta
        self.optimizer = tf.keras.optimizers.Adam(lr=learning_rate)
        start = time.time()
        for epoch in range(epochs):
            # for batch, (x, y) in enumerate(dataloader):
            # Compute prediction and loss
            loss = self.train_step(trajectory.T[:-1], trajectory.T[1:])
                
            if epoch % 100 == 0:
                loss = loss.numpy()
                end = time.time()
                print(f"epoch: {epoch}    loss: {loss:>7f}      time elapsed={end-start:.4f}")

    
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
            print(f'Working with index = {validation_index}')
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







