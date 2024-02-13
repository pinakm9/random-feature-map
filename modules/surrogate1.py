# load necessary modules
import numpy as np 
import os, sys 
from pathlib import Path
from os.path import dirname, realpath
script_dir = Path(dirname(realpath('.')))
module_dir = str(script_dir)
sys.path.insert(0, module_dir + '/modules')
import utility as ut
# from scipy.linalg import eigh
import pandas as pd
import json
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import sample as sm
import surrogate as sr
import time
from joblib import Parallel, delayed, parallel_config
from scipy.interpolate import splrep, BSpline


class BatchStrategy_SMLR:
    """
    Description: runs batches of surrogate models using uniformly random features. Initialization involves picking 
    either good, linear or extreme rows randomly.
    """
    def __init__(self, save_folder, D, D_r, n_repeats, beta, error_theshold, train, test, dt, Lyapunov_time=1.,\
                 L0=0.4, L1=3.5, percents=3, row_selection='good_linear_extreme', train_option='constant',\
                 limits_W_in=[-0.1, 0.1], limits_W=[-0.5, 0.5]):
        """
        Args:
            save_folder: folder where all the generated data will be saved
            D: dimension of the underlying dynamical system
            D_r: dimension of the reservoir
            n_repeats: the number of repeats for every good row percentage 
            beta: regularization parameters for ridge regression
            train: training data
            test: trajectories for calculating tau_f
            error_threshold: threshold to determine forecast time
            dt: time step of the surrogate model
            Lyapunov_time: Lyapunov time for the underlying dynamical system, default=1
            L0: lower bound for determining good rows
            L1: upper bound for determining good rows
            percents: an integer determining the allowed percentages (uniform, starting at 0. and ending at 100.)
                      for any type of row
            row_selection: row selection strategy
            train_option: training data selection strategy
            limits_W_in: limits for counting zero rows of W_in 
            limits_W: limits for counting zero columns of W
        """
        self.D = D
        self.D_r = D_r

        self.n_models = (percents + 1) * n_repeats
        self.beta = beta
        self.train = train
        self.test = test
        self.bad_feature_count_trajectory = test.reshape((test.shape[1], -1))[:, :1600]
        self.error_threshold = error_theshold
        self.dt = dt 
        self.Lyapunov_time = Lyapunov_time
        dx = 100./percents
        self.percents = np.arange(0., 100. + dx, dx)
        self.remaining = ((100. + dx - self.percents) / dx).astype(int)
        self.L0 = L0
        self.L1 = L1
        self.sampler = sm.MatrixSampler(L0, L1, train.T)
        self.n_repeats = n_repeats
        self.row_selection = row_selection
        self.train_option = train_option
        if self.train_option == 'constant':
            self.training_points = self.train.shape[1]
        elif self.train_option.startswith('random'):
            self.training_points = int(self.train_option.split('_')[-1])
        self.validation_points = self.test.shape[-1]
        self.limits_W_in = limits_W_in
        self.limits_W = limits_W

        folders = [save_folder, save_folder + '/W_in', save_folder + '/b_in', save_folder + '/W']

        attributes = ['save_folder', 'W_in_folder', 'b_in_folder', 'W_folder']

        for i, folder in enumerate(folders):
            if not os.path.exists(folder):
                os.makedirs(folder)
            setattr(self, attributes[i], folder)

        # self.data = {'l':[], 'train_index': [], 'test_index': [], '||W_in||':[], '||b_in||':[],\
        #              '||W||':[], 'good_rows_W_in':[], 'linear_rows_W_in':[],\
        #              'extreme_rows_W_in': [], 'tau_f_se': [], 'tau_f_rmse': [], 'se': [], 'rmse': [],\
        #              '0_cols_W': [], '0_rows_W_in': [], 'avg_bad_features': [], 'avg_abs_col_sum_W': []}    
        
        config = {'D': D, 'D_r': D_r, 'Lyapunov_time': Lyapunov_time, 'dt': dt, 'beta': beta,\
                  'error_threshold': error_theshold, 'n_repeats': n_repeats, 'n_models': self.n_models, 'percents': list(self.percents),\
                  'L0':L0, 'L1':L1, 'training_points': self.training_points, 'row_selection': row_selection,\
                  'train_option': train_option, 'validation_points': self.validation_points,\
                  'possible_validation_trajectories': len(self.test), 'full_train_trajectory_length': self.train.shape[1],\
                  'bad_feature_count_trajectory_length': self.bad_feature_count_trajectory.shape[-1],\
                  'limits_W_in': list(limits_W_in), 'limits_W': list(limits_W)}

        with open('{}/config.json'.format(self.save_folder), 'w') as f:
            f.write(json.dumps(config))
    


    def get_row_partitions_50_50(self, percent):
        good = int(percent * self.D_r / 100.)
        linear = int((100. - percent) * self.D_r / 200.)
        extreme = self.D_r - good - linear
        return [good, linear, extreme]



    def count_zero_rows_entry(self, matrix, limits):
        new_matrix = matrix.copy()
        new_matrix[(new_matrix > limits[0]) & (new_matrix < limits[1])] = 0. 
        return np.sum(~(new_matrix.any(1)))
    
    
    def count_bad_features(self, model, threshold=0.998):
        features = np.abs(np.tanh(model.W_in@self.bad_feature_count_trajectory + model.b_in[:,np.newaxis]))
        bad = np.sum(features > threshold, axis=0)
        return bad.mean()


    # @ut.timer
    def count(self, model, threshold=3.5):
            return self.count_zero_rows_entry(model.W_in, self.limits_W_in) / self.D_r,\
                   self.count_zero_rows_entry(model.W.T, self.limits_W) / self.D_r,\
                   self.count_bad_features(model, np.tanh(threshold)) / self.D_r,\
                   np.sum(np.abs(model.W), axis=0).mean() / self.D_r
        

    # @ut.timer
    def run_single(self, l, i, j, W_in, b_in, partition, save_data=False):
        """
        Description: runs experiments for a single value of (w, b) multiple times 

        Args: 
            l: index/identifier for the single experiment
            i: index to select training data
            j: index to select validation data
            W_in: parameters for W_in
            b_in: parameters for b_in
            partition: partition of row types of W_in
            save_data: boolean determing wheather to save model
        """
        model = sr.SurrogateModel_LR(self.D, self.D_r, W_in, b_in)
        model.compute_W(self.train[:, i:i+self.training_points], beta=self.beta);
   
        if save_data:
            np.save(self.W_in_folder + '/W_in_{}.npy'.format(l), model.W_in)
            np.save(self.b_in_folder + '/b_in_{}.npy'.format(l), model.b_in)
            np.save(self.W_folder + '/W_{}.npy'.format(l), model.W)
        
        results = [l, i, j, np.linalg.norm(model.W_in)/self.D_r,\
                   np.linalg.norm(model.b_in)/self.D_r, np.linalg.norm(model.W)/self.D_r] +\
                   [part/float(self.D_r) for part in partition]
        results += self.compute_error(model, j)
        results += self.count(model)

        return results



    @ut.timer
    def run(self, save_data=False):
        """
        Description: runs all the experiments, documents W_in, b_in, W, forecast times and errors
        """
        l = 0
        file_path = '{}/batch_data.csv'.format(self.save_folder)
        if os.path.exists(file_path):
            os.remove(file_path)
        columns = ['l', 'train_index', 'test_index', '||W_in||', '||b_in||', '||W||', 'good_rows_W_in', 'linear_rows_W_in', 'extreme_rows_W_in',\
                   'tau_f_rmse', 'tau_f_se', 'rmse', 'se',\
                   '0_rows_W_in', '0_cols_W', 'avg_bad_features', 'avg_abs_col_sum_W']
        
        for percent in self.percents:
            print(f'Generating parameters for {percent}% good rows ...')
            partition = self.get_row_partitions_50_50(percent)
            W_ins, b_ins = self.sampler.sample_parallel(partition, self.n_repeats)
            print('Running experiments ...')
            experiment_indices = list(range(l, l+self.n_repeats))
            train_indices = np.random.randint(self.train.shape[1] - self.training_points, size=self.n_repeats)
            test_indices = np.random.randint(len(self.test), size=self.n_repeats)
            start = time.time()
            data = Parallel(n_jobs=-1)(delayed(self.run_single)\
                                (experiment_indices[k], train_indices[k], test_indices[k], W_ins[k], b_ins[k], partition, save_data)\
                                for k in range(self.n_repeats))
            print('Documenting results ...')#, len(columns), len(data[0]))
            pd.DataFrame(data, columns=columns, dtype=float)\
                        .to_csv(file_path, mode='a', index=False, header=not os.path.exists(file_path))
            del data         
            # for k in range(self.n_repeats):
            #     self.run_single(experiment_indices[k], train_indices[k], test_indices[k], W_ins[k], b_ins[k], partition)
            end = time.time()
            print(f'Time taken for batch of experiments = {end-start:.2f}s')
            l += self.n_repeats

    @ut.timer
    def run_single_percent(self, percent_id, save_data=False):
        """
        Description: runs all the experiments, documents W_in, b_in, W, forecast times and errors
        """
        l = percent_id * self.n_repeats
        file_path = '{}/batch_data_{}.csv'.format(self.save_folder, percent_id)
        if os.path.exists(file_path):
            os.remove(file_path)
        columns = ['l', 'train_index', 'test_index', '||W_in||', '||b_in||', '||W||', 'good_rows_W_in', 'linear_rows_W_in', 'extreme_rows_W_in',\
                   'tau_f_rmse', 'tau_f_se', 'rmse', 'se',\
                   '0_rows_W_in', '0_cols_W', 'avg_bad_features', 'avg_abs_col_sum_W']

        percent = self.percents[percent_id]
        print(f'Generating parameters for {percent}% good rows ...')
        partition = self.get_row_partitions_50_50(percent)
        W_ins, b_ins = self.sampler.sample_parallel(partition, self.n_repeats)
        print('Running experiments ...')
        experiment_indices = list(range(l, l+self.n_repeats))
        train_indices = np.random.randint(self.train.shape[1] - self.training_points, size=self.n_repeats)
        test_indices = np.random.randint(len(self.test), size=self.n_repeats)
        data = Parallel(n_jobs=-1)(delayed(self.run_single)\
                            (experiment_indices[k], train_indices[k], test_indices[k], W_ins[k], b_ins[k], partition, save_data)\
                            for k in range(self.n_repeats))
        print('Documenting results ...')#, len(columns), len(data[0]))
        pd.DataFrame(data, columns=columns, dtype=float)\
                    .to_csv(file_path, mode='a', index=False, header=not os.path.exists(file_path))
        del data         


         
            




class BatchStrategyAnalyzer_SMLR:
    """
    Description: A class for analyzing the results of a batch run
    """
    def __init__(self, save_folder) -> None:
        """
        Args: 
            save_folder: folder where the results of batch run experiments were stored
        """
        self.save_folder = save_folder
        folders = [save_folder, save_folder + '/W_in', save_folder + '/b_in', save_folder + '/W', save_folder + '/tau_f',\
                    save_folder + '/errors', save_folder + '/plots']
        attributes = ['save_folder', 'W_in_folder', 'b_in_folder', 'W_folder', 'tau_f_folder', 'errors_folder', 'plot_folder']
        for i, attribute in enumerate(attributes):
            setattr(self, attribute, folders[i])
        
        if not os.path.exists(self.plot_folder):
            os.makedirs(self.plot_folder)
        
        self.get_config()

    # @ut.timer
    def compute_error(self, l, test, good, linear):
        """
        Description: computes forecast time tau_f for the computed surrogate model

        Args:
            l: surrogate model index
            test: list of test trajectories
            good: fraction of good rows
            linear: fraction of linear rows
        """
        model = self.get_model(l)
        m = test.shape[-1]
        def err(path):
            prediction = model.multistep_forecast(path[:, 0], m)
            return (np.linalg.norm(path - prediction, axis=0)**2 / np.linalg.norm(path, axis=0)**2).mean()
        results = Parallel(n_jobs=-1)(delayed(err)(path) for path in test)
        sq_err = np.array(results).mean()
        
        ng, nl = int(good*self.D_r), int(linear*self.D_r)

        good = model.W[:, :ng]
        linear = model.W[:, ng:ng+nl]
        extreme = model.W[:, ng+nl:]

        zc_g = self.count_zero_rows_entry(good.T, self.limits_W)
        zc_l = self.count_zero_rows_entry(linear.T, self.limits_W)
        zc_e = self.count_zero_rows_entry(extreme.T, self.limits_W)

        return l, sq_err, zc_g, zc_l, zc_e


    def compute_training_error(self, l, train, train_index, length, W_norm, mse):
        """
        Description: computes forecast time tau_f for the computed surrogate model

        Args:
            l: surrogate model index
            train: training data
            W_norm: norm of W
            mse: mean square test error
    
        """
        model = self.get_model(l)
        train_index = int(train_index)

        prediction = model.multistep_forecast(train[:, train_index], length)
        train_sq_err = (np.linalg.norm(train[:, train_index:train_index+length] - prediction, axis=0)**2 / np.linalg.norm(train[:, train_index:train_index+length], axis=0)**2).mean()
       
        penalty = (W_norm)**2 * self.beta
        return l, train_sq_err, train_sq_err + penalty, mse + penalty 
    
    def stat(self, func, arr):
        arr = arr.flatten()
        if len(arr) == 0:
            return np.nan
        else:
            return getattr(np, func)(arr)
        


    def compute_max_mean(self, l, i, j):
        W = np.abs(self.get_W(l))
        # W /= np.max(W, axis=1)[:, np.newaxis]
        W_g = W[:, :i]
        W_l = W[:, i:i+j]
        W_e = W[:, i+j:]
        return self.stat('max', W), self.stat('min', W), self.stat('mean', W), self.stat('median', W),\
               self.stat('max', W_g), self.stat('min', W_g), self.stat('mean', W_g), self.stat('median', W_g),\
               self.stat('max', W_l), self.stat('min', W_l), self.stat('mean', W_l), self.stat('median', W_l),\
               self.stat('max', W_e), self.stat('min', W_e), self.stat('mean', W_e), self.stat('median', W_e),\
               self.count_zero_rows_entry(W_g.T, self.limits_W) / self.D_r,\
               self.count_zero_rows_entry(W_l.T, self.limits_W) / self.D_r,\
               self.count_zero_rows_entry(W_e.T, self.limits_W) / self.D_r
               

                                               


    def count_zero_rows_entry(self, matrix, limits):
        new_matrix = matrix.copy()
        new_matrix[(new_matrix > limits[0]) & (new_matrix < limits[1])] = 0. 
        return np.sum(~(new_matrix.any(1)))
    
    @ut.timer
    def compute_err_zc(self, test, percents=50):
        l = 0
        data = self.get_data()
        new_data = {}
        new_data['mean_sq_err'] = []
        new_data['zc_good'] = []
        new_data['zc_linear'] = []
        new_data['zc_extreme'] = [] 
        dx = 100./percents
        percents = np.arange(0., 100. + dx, dx) / 100.
        for percent in percents:
            print(f'Computing error for {percent} good rows ...')
            experiment_indices = list(range(l, l+self.n_repeats))
            # linear = data['linear_rows_W_in'][l: l+self.n_repeats]
            start = time.time()
            # with parallel_config(backend='threading', n_jobs=-1):
            #     results = Parallel(n_jobs=-1)(delayed(self.compute_error)\
            #                     (experiment_indices[k], test, good[k], linear[k]) for k in range(self.n_repeats))
            results = [self.compute_error(experiment_indices[k], test, percent, 0.5*(1.-percent)) for k in range(self.n_repeats)]
            results = np.array(results).T
            new_data['mean_sq_err'] += list(results[1])
            new_data['zc_good'] += list(results[2])
            new_data['zc_linear'] += list(results[3])
            new_data['zc_extreme'] += list(results[4])   
            end = time.time()
            print(f'Time taken for batch of error computations = {end-start:.2f}s')
            l += self.n_repeats
        data['mean_sq_err'] = new_data['mean_sq_err']
        data['zc_good'] = new_data['zc_good']
        data['zc_linear'] = new_data['zc_linear']
        data['zc_extreme'] = new_data['zc_extreme'] 
        data.to_csv(f'{self.save_folder}/batch_data.csv', index=False)

        
    @ut.timer
    def compute_train_err(self, train, length=220, percents=50):
        l = 0
        data = self.get_data()
        new_data = {}
        new_data['train_sq_err'] = []
        new_data['train_sq_err_penalty'] = []
        new_data['mean_sq_err_penalty'] = []
    
        dx = 100./percents
        percents = np.arange(0., 100. + dx, dx) / 100.
        for percent in percents:
            print(f'Computing error for {percent} good rows ...')
            experiment_indices = list(range(l, l+self.n_repeats))
            train_indices = data['train_index'].to_numpy()[l: l+self.n_repeats]
            W_norm = data['||W||'].to_numpy()[l: l+self.n_repeats] * self.D_r     
            mse = data['mean_sq_err'].to_numpy()[l: l+self.n_repeats]
            start = time.time()
            with parallel_config(backend='threading', n_jobs=-1):
                results = Parallel(n_jobs=-1)(delayed(self.compute_training_error)\
                                (experiment_indices[k], train, train_indices[k], length, W_norm[k], mse[k]) for k in range(self.n_repeats))
            # results = [self.compute_training_error(experiment_indices[k], train, train_indices[k], length, W_norm[k], mse[k])\
            #             for k in range(self.n_repeats)]
            results = np.array(results).T
            new_data['train_sq_err'] += list(results[1])
            new_data['train_sq_err_penalty'] += list(results[2])
            new_data['mean_sq_err_penalty'] += list(results[3])
         
            end = time.time()
            print(f'Time taken for batch of error computations = {end-start:.2f}s')
            l += self.n_repeats
            
        data['train_sq_err'] = new_data['train_sq_err']
        data['train_sq_err_penalty'] = new_data['train_sq_err_penalty']
        data['mean_sq_err_penalty'] = new_data['mean_sq_err_penalty']
        data.to_csv(f'{self.save_folder}/batch_data.csv', index=False)

    
    def train_loss(self, l, train, train_index, length):
        model = self.get_model(l)
        train_index = int(train_index)
        return l, np.sum((model.forecast_m(train[:, train_index:train_index+length]) - train[:, train_index+1:train_index+length+1])**2) + self.beta * np.sum(model.W**2)
    

    def test_loss(self, l, test, dt=0.02, Lyapunov_time=1./0.91, penalty=True):
        model = self.get_model(l)
        half = round(0.5*Lyapunov_time/dt)
        one = round(Lyapunov_time/dt)
        two = round(2.0*Lyapunov_time/dt)
        loss_half = 0.
        loss_one = 0.
        loss_two = 0.
        n = len(test)
        if penalty:
            penalty = self.beta * np.sum(model.W**2)
        else:
            penalty = 0.
        for trajectory in test:
            path = model.forecast_m(trajectory[:, :two+1])
            loss_half += np.sum((path[:, :half] - trajectory[:, 1:half+1])**2) + penalty
            loss_one += np.sum((path[:, :one] - trajectory[:, 1:one+1])**2) + penalty
            loss_two += np.sum((path[:, :two] - trajectory[:, 1:two+1])**2) + penalty
        return l, loss_half/n, loss_one/n, loss_two/n


    @ut.timer
    def compute_train_loss(self, train, percents=50):
        l = 0
        data = self.get_data()
        new_data = {}
        new_data['train_loss'] = []
        dx = 100./percents
        percents = np.arange(0., 100. + dx, dx) / 100.
        for percent in percents:
            print(f'Computing loss for {percent} good rows ...')
            experiment_indices = list(range(l, l+self.n_repeats))
            train_indices = data['train_index'].to_numpy()[l: l+self.n_repeats]
            start = time.time()
            with parallel_config(backend='threading', n_jobs=-1):
                results = Parallel(n_jobs=-1)(delayed(self.train_loss)(experiment_indices[k], train, train_indices[k], self.training_points) for k in range(self.n_repeats))
            results = np.array(results).T
            new_data['train_loss'] += list(results[1])
            end = time.time()
            print(f'Time taken for batch of training loss computations = {end-start:.2f}s')
            l += self.n_repeats
        
        data['train_loss'] = new_data['train_loss']
        data.to_csv(f'{self.save_folder}/batch_data.csv', index=False)


    @ut.timer
    def compute_test_loss(self, test, percents=50, dt=0.02, Lyapunov_time=1/0.91, penalty=True):
        l = 0
        data = self.get_data()
        new_data = {}
        
        new_data['test_loss_half'] = []
        new_data['test_loss_one'] = []
        new_data['test_loss_two'] = []

        dx = 100./percents
        percents = np.arange(0., 100. + dx, dx) / 100.

        for percent in percents: 
            print(f'Computing loss for {percent} good rows ...')
            experiment_indices = list(range(l, l+self.n_repeats))
            start = time.time()
            with parallel_config(backend='threading', n_jobs=-1):
                results = Parallel(n_jobs=-1)(delayed(self.test_loss)(experiment_indices[k], test, dt, Lyapunov_time, penalty) for k in range(self.n_repeats))
            results = np.array(results).T
            print(len(results[0]))
            new_data['test_loss_half'] += list(results[1])
            new_data['test_loss_one'] += list(results[2])
            new_data['test_loss_two'] += list(results[3])
            end = time.time()
            print(f'Time taken for batch of testing loss computations = {end-start:.2f}s')
            l += self.n_repeats
        
        data['test_loss_half'] = new_data['test_loss_half']
        data['test_loss_one'] = new_data['test_loss_one']
        data['test_loss_two'] = new_data['test_loss_two']
        data.to_csv(f'{self.save_folder}/batch_data.csv', index=False)



    @ut.timer
    def compute_mmmm(self, percents=50):
        l = 0
        data = self.get_data()
        keys = ['|W|_max',
                '|W|_min', 
                '|W|_mean',
                '|W|_median',     
                '|W_g|_max',
                '|W_g|_min', 
                '|W_g|_mean',
                '|W_g|_median',  
                '|W_l|_max',
                '|W_l|_min', 
                '|W_l|_mean',
                '|W_l|_median', 
                '|W_e|_max',
                '|W_e|_min', 
                '|W_e|_mean',
                '|W_e|_median', 
                'zc_good',
                'zc_linear',
                'zc_extreme'] 
        
        new_data = {}
        for key in keys:
            new_data[key] = []   
    
        dx = 100./percents
        percents = np.arange(0., 100. + dx, dx) / 100.
        for percent in percents:
            print(f'Computing error for {percent} good rows ...')
            experiment_indices = list(range(l, l+self.n_repeats))
            good_indices = (data['good_rows_W_in'].to_numpy()[l: l+self.n_repeats] * self.D_r).astype(int)
            linear_indices = (data['linear_rows_W_in'].to_numpy()[l: l+self.n_repeats] * self.D_r).astype(int)
            start = time.time()
            with parallel_config(backend='threading', n_jobs=-1):
                results = Parallel(n_jobs=-1)(delayed(self.compute_max_mean)\
                                (experiment_indices[k], good_indices[k], linear_indices[k]) for k in range(self.n_repeats))
            # results = [self.compute_training_error(experiment_indices[k], train, train_indices[k], length, W_norm[k], mse[k])\
            #             for k in range(self.n_repeats)]
            results = np.array(results).T

            for i, key in enumerate(keys):
                new_data[key] += list(results[i])
        
            end = time.time()
            print(f'Time taken for batch of error computations = {end-start:.2f}s')
            l += self.n_repeats
        for key in keys:
            data[key] = new_data[key]
        data.to_csv(f'{self.save_folder}/batch_data.csv', index=False)


    def get_config(self):
        with open('{}/config.json'.format(self.save_folder), 'r') as f:
            config = json.loads(f.read())
        for key in config:
            setattr(self, key, config[key])
        self.percents = np.unique(self.get_data()['good_rows_W_in'])

                    

    @ut.timer
    def get_mean_std(self, quantity):
        mean, std = np.zeros(self.n_models), np.zeros(self.n_model) 
        data = pd.read_csv('{}/batch_data.csv'.format(self.save_folder))
        for l in range(self.n_model):
            array = data.loc[(data['l'] == l)][quantity]
            mean[l], std[l] = np.mean(array), np.std(array)
        return mean, std
    
    def get_W_in(self, l):
        return np.load(self.W_in_folder + '/W_in_{}.npy'.format(l))
    
    def get_b_in(self, l):
        return np.load(self.b_in_folder + '/b_in_{}.npy'.format(l))
    
    def get_W(self, l):
        return np.load(self.W_folder + '/W_{}.npy'.format(l))
    
    
    def get_model(self, l):
        W_in_fn = lambda d, d_r: np.load(self.W_in_folder + '/W_in_{}.npy'.format(l))
        b_in_fn  = lambda d_r: np.load(self.b_in_folder + '/b_in_{}.npy'.format(l))
        W = np.load(self.W_folder + '/W_{}.npy'.format(l))
        model = sr.SurrogateModel_LR(self.D, self.D_r, W_in_fn, b_in_fn, W)
        return model
    

    @ut.timer
    def get_grid(self, quantity):
        m = len(self.percents)
        mean, std = np.zeros((m, m)), np.zeros((m, m))
        data = pd.read_csv('{}/batch_data.csv'.format(self.save_folder))
        for i in range(m):
            good = self.percents[i]
            for j in range(m): 
                extreme = self.percents[j]
                array = data.loc[(data['good_rows_W_in'] == good) & (data['extreme_rows_W_in'] == extreme)][quantity]
                mean[i][j], std[i][j] = np.mean(array), np.std(array)
        return mean, std 
    
    @ut.timer
    def get_grid_bad(self, quantity):
        m = len(self.percents)
        mean, std = np.zeros((m, m)), np.zeros((m, m))
        data = pd.read_csv('{}/batch_data.csv'.format(self.save_folder))
        for i in range(m):
            linear = self.percents[i]
            for j in range(m): 
                extreme = self.percents[j]
                array = data.loc[(data['linear_rows_W_in'] == linear) & (data['extreme_rows_W_in'] == extreme)][quantity]
                mean[i][j], std[i][j] = np.mean(array), np.std(array)
        return mean, std 


    def get_grid_abc(self, a, b, c, arr_a, arr_b):
        m, n = len(arr_a), len(arr_b)
        num, mean, std = np.zeros((m, n)), np.zeros((m, n)), np.zeros((m, n))
        data = pd.read_csv('{}/batch_data.csv'.format(self.save_folder))
        for i in range(m):
            
            for j in range(n): 
                
                array = data.loc[(data[a] <= arr_a[i]) & (data[b] <= arr_b[j])][c]
                mean[i][j], std[i][j] = np.mean(array), np.std(array)
                num[i][j] = len(array)
        return mean, std, num 


    def get_line(self, quantity, spline=0):
        m = len(self.percents)
        mean, std = np.zeros(m), np.zeros(m)
        data = pd.read_csv('{}/batch_data.csv'.format(self.save_folder))
        for i in range(m):
            good = self.percents[i]
            array = data.loc[data['good_rows_W_in'] == good][quantity]
            mean[i], std[i] = array.mean(), array.std()
        if spline > 0:
            tck = splrep(self.percents, mean, s=spline)
            mean = BSpline(*tck)(self.percents)
            tck = splrep(self.percents, std, s=spline)
            std = BSpline(*tck)(self.percents)
        return mean, std
    
    def get_ratio_line(self, quantity1, quantity2):
        m = len(self.percents)
        mean, std = np.zeros(m), np.zeros(m)
        data = pd.read_csv('{}/batch_data.csv'.format(self.save_folder))
        for i in range(m):
            good = self.percents[i]
            array = data.loc[data['good_rows_W_in'] == good][quantity1] / data.loc[data['good_rows_W_in'] == good][quantity2]
            mean[i], std[i] = array.mean(), array.std()
        return mean, std
    

    def get_line_gos(self, quantity, gen=4., over=1.5, super=5.):
        m = len(self.percents)
        mean_g, std_g = np.zeros(m), np.zeros(m)
        mean_o, std_o = np.zeros(m), np.zeros(m)
        mean_s, std_s = np.zeros(m), np.zeros(m)
        num_g, num_o, num_s = np.zeros(m), np.zeros(m), np.zeros(m)
        data = pd.read_csv('{}/batch_data.csv'.format(self.save_folder))
        for i in range(m):
            good = self.percents[i]
            array_g = data.loc[(data['good_rows_W_in'] == good) & (data['tau_f_se'] >= gen)][quantity]
            array_o = data.loc[(data['good_rows_W_in'] == good) & (data['tau_f_se'] <= over)][quantity]
            array_s = data.loc[(data['good_rows_W_in'] == good) & (data['tau_f_se'] >= super)][quantity]
            mean_g[i], std_g[i] = array_g.mean(), array_g.std()
            mean_o[i], std_o[i] = array_o.mean(), array_o.std()
            mean_s[i], std_s[i] = array_s.mean(), array_s.std()
            num_g[i] = len(array_g)
            num_o[i] = len(array_o)
            num_s[i] = len(array_s)
        return [mean_g, std_g, num_g], [mean_o, std_o, num_o], [mean_s, std_s, num_s]


    def get_data(self):
        data = pd.read_csv('{}/batch_data.csv'.format(self.save_folder))
        return data
    

    def compute_optimal_p(self, gen=4., threshold=0.5, spline=0):
        data = pd.read_csv('{}/batch_data.csv'.format(self.save_folder))
        data_g = data.loc[data['tau_f_se'] >= gen]
        m = len(self.percents)
        probs = np.zeros(m)
        for i in range(m):
            a = len(data_g.loc[data_g['good_rows_W_in']==self.percents[i]]['tau_f_se'])
            A = len(data.loc[data['good_rows_W_in']==self.percents[i]]['tau_f_se'])
            probs[i] = a/float(A)
        if spline > 0:
            tck = splrep(self.percents, probs, s=spline)
            probs = BSpline(*tck)(self.percents)

        m = np.argmax(probs >= threshold)
        if m == 0:
            m = -1

        arr = self.get_line('0_cols_W', spline=spline)[0]
        n = max(np.argmax(arr), len(arr)-1-np.argmax(arr[::-1]))

        return m, n
    

    def write_ftle(self, ftle_file):
        ftle = np.load(ftle_file)
        data = self.get_data()
        ftle_ = np.zeros(len(data['l']))
        for i, j in enumerate(data['test_index'].to_numpy().astype(int)):
            ftle_[i] = ftle[j]
        data['ftle'] = ftle_
        data.to_csv(f'{self.save_folder}/batch_data.csv', index=False)

    
    
    @ut.timer
    def plot(self, reduction_factor=10, s=20):
        xlabel_size = 13
        ylabel_size = 13
        title_size = 13
        tick_size = 5

        # plot grid quantities
        data = self.get_data()
        total = self.n_models
        random_idx = np.random.choice(total, size=int(total/reduction_factor), replace=False)
        data_list = []
        data_list.append(data['tau_f_se'][random_idx])
        data_list.append(data['0_cols_W'][random_idx])
        # data_list.append(data['avg_abs_col_sum_W'][random_idx])
        data_list.append(data['||W||'][random_idx])
        data_list.append(data['avg_bad_features'][random_idx])
        data_list.append(data['good_rows_W_in'][random_idx])
        data_list.append(data['linear_rows_W_in'][random_idx])
        data_list.append(data['extreme_rows_W_in'][random_idx])
   
        label_list = []
        label_list.append(r'$\tau_f$')
        label_list.append(r'zero cols $W$')
        # label_list.append(r'$E|\text{col|sum } W$')
        label_list.append(r'$\|W\|$')
        label_list.append(r'bad features')
        label_list.append(r'good rows $W_{\rm in}$')
        label_list.append(r'linear rows $W_{\rm in}$')
        label_list.append(r'extreme rows $W_{\rm in}$')

        pairs = []
        for i in range(len(data_list)):
            for j in range(len(data_list)):
                if i < j:
                    pairs.append([j, i])
        
        n_rows, n_cols = 6, 4
        fig_name = '{}/relationships.png'.format(self.plot_folder)
        # s = 20
        figsize = 4
        wspace = 0.3
        hspace = 0.3 
        color_list = [data_list[0]] * len(pairs)
        size_list = [data_list[4]] * len(pairs)
        ut.grid_scatter_plotter(data_list, label_list, color_list, pairs, n_rows, n_cols, fig_name,\
						 xlabel_size, ylabel_size, title_size, tick_size,\
						 wspace, hspace, s, figsize, size_list)

       
       
       
        grid_data_list = []
        grid_data_list.append(self.get_grid('tau_f_se')[0])
        grid_data_list.append(self.get_grid('0_cols_W')[0])
        grid_data_list.append(self.get_grid('||W||')[0])
        grid_data_list.append(self.get_grid('avg_bad_features')[0])
 
        x = np.array(self.percents) / 100.
        xlabel = 'good rows'
        ylabel = 'extreme rows'

        n_rows, n_cols = 2, 2
        fig_name = '{}/good-bad.png'.format(self.plot_folder)

       
        ut.grid_heat_plotter(x, x, grid_data_list, xlabel, ylabel, label_list, n_rows, n_cols, fig_name,\
						xlabel_size, ylabel_size, title_size, tick_size,\
						wspace, hspace, figsize, show=False)
        
        bad_grid_data_list = []
        bad_grid_data_list.append(self.get_grid('tau_f_se')[0])
        bad_grid_data_list.append(self.get_grid('0_cols_W')[0])
        bad_grid_data_list.append(self.get_grid('||W||')[0])
        bad_grid_data_list.append(self.get_grid('avg_bad_features')[0])

        xlabel = 'linear rows'
        fig_name = '{}/bad-bad.png'.format(self.plot_folder)
        ut.grid_heat_plotter(x, x, bad_grid_data_list, xlabel, ylabel, label_list, n_rows, n_cols, fig_name,\
						xlabel_size, ylabel_size, title_size, tick_size,\
						wspace, hspace, figsize, show=False)

        plt.hist(data_list[0], weights=np.ones(len(data_list[0])) / len(data_list[0]))
        plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
        plt.xlabel(r'$\tau_f$')
        plt.savefig('{}/distribution_of_realizations.png'.format(self.plot_folder))
        plt.close()

        self.plot_extras(x, 'good rows')

        quantities = ['tau_f_se', '0_cols_W', '||W||', 'avg_bad_features']
        for i, quantity in enumerate(quantities):
            self.plot_against(quantity, quantities[:i]+quantities[i+1:])
       
     

    def get_generalize_overfit_superb(self, quantity, gen=4., over=1.5, super=5.):
        data = self.get_data()
        generalize = np.zeros(len(self.percents))
        overfit = np.zeros(len(self.percents))
        superb = np.zeros(len(self.percents)) 

        generalize_std = np.zeros(len(self.percents))
        overfit_std = np.zeros(len(self.percents))
        superb_std = np.zeros(len(self.percents))

        num_gen = np.zeros(len(self.percents))
        num_over = np.zeros(len(self.percents))
        num_super = np.zeros(len(self.percents))

        for i, p in enumerate(self.percents):
            subset = data.loc[data['good_rows_W_in'] == p]
            arr_gen = subset.loc[subset['tau_f_se'] >= gen][quantity]
            arr_over = subset.loc[subset['tau_f_se'] <= over][quantity]
            arr_super = subset.loc[subset['tau_f_se'] >= super][quantity]

            generalize[i] = arr_gen.mean()
            overfit[i] = arr_over.mean()
            superb[i] = arr_super.mean()

            generalize_std[i] = arr_gen.std()
            overfit_std[i] = arr_over.std()
            superb_std[i] = arr_super.std()

            num_gen[i] = len(arr_gen)
            num_over[i] = len(arr_over)
            num_super[i] = len(arr_super)


        return [[generalize, overfit, superb], [generalize_std, overfit_std, superb_std], [num_gen, num_over, num_super]]
    


    @ut.timer
    def plot_extras(self, x, xlabel):
        n_rows, n_cols = 5, 2
        quantities = ['tau_f_se', '0_cols_W', '||W||', 'avg_bad_features']
        labels = [r'$\tau_f$', 'zero cols W', '||W||', 'bad features']
        plot_funcs = ['plot', 'semilogy', 'loglog']

        for func in plot_funcs:
            fig = plt.figure(figsize=(4*n_cols, 4*n_rows))
            axes = [fig.add_subplot(n_rows, n_cols, j+1) for j in range(9)]
            for i in range(0, 8, 2):
                j = int(i/2)
                mean, std, num = self.get_generalize_overfit_superb(quantities[j])
                if i > 0:
                    mean = [e for e in mean]
                    std = [e for e in std]

                getattr(axes[i], func)(x, mean[0], label='generalize') 
                getattr(axes[i], func)(x, mean[1], label='overfit') 
                getattr(axes[i], func)(x, mean[2], label='superb')

                axes[i].set_xlabel(xlabel)
                axes[i].set_ylabel(labels[j])
                axes[i].legend()

                getattr(axes[i+1], func)(x, std[0], label='generalize') 
                getattr(axes[i+1], func)(x, std[1], label='overfit') 
                getattr(axes[i+1], func)(x, std[2], label='superb') 

                axes[i+1].set_xlabel(xlabel)
                axes[i+1].set_ylabel('std ' + labels[j])
                axes[i+1].legend()

            getattr(axes[-1], func)(x, num[0], label='generalize') 
            getattr(axes[-1], func)(x, num[1], label='overfit') 
            getattr(axes[-1], func)(x, num[2], label='superb') 

            axes[-1].set_xlabel(xlabel)
            axes[-1].set_ylabel('number of occurences')
            axes[-1].legend()

            fig.subplots_adjust(wspace=0.3, hspace=0.5)
            plt.savefig('{}/gos-{}-{}.png'.format(self.plot_folder, xlabel, func))
            plt.close()



    def plot_against_std(self, xlabel, ylabels):
        fig = plt.figure(figsize=(4*3, 4*2))
        x = self.get_line(xlabel)[1]
        for i, ylabel in enumerate(ylabels):
            ax = fig.add_subplot(2, 3, i+1) 
            y = self.get_line(ylabel)[0]
            ax.scatter(x, y)
            ax.set_xlabel('std ' + xlabel)
            ax.set_ylabel(ylabel)

        y = self.get_line_gos(ylabel)
        for i, ylabel in enumerate(['# generalize', '# overfit', '# superb']):
            ax = fig.add_subplot(2, 3, i+4)
            ax.scatter(x, y[i][-1])
            ax.set_xlabel('std ' + xlabel)
            ax.set_ylabel(ylabel)

        fig.subplots_adjust(wspace=0.3, hspace=0.5)
        plt.savefig(f'{self.plot_folder}/against_std_{xlabel}.png')
        plt.close()

    
    def plot_against_mean(self, xlabel, ylabels):
        fig = plt.figure(figsize=(4*3, 4*2))
        x = self.get_line(xlabel)[0]
        for i, ylabel in enumerate(ylabels):
            ax = fig.add_subplot(2, 3, i+1) 
            y = self.get_line(ylabel)[0]
            ax.scatter(x, y)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)

        y = self.get_line_gos(ylabel)
        for i, ylabel in enumerate(['# generalize', '# overfit', '# superb']):
            ax = fig.add_subplot(2, 3, i+4)
            ax.scatter(x, y[i][-1])
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)

        fig.subplots_adjust(wspace=0.3, hspace=0.5)
        plt.savefig(f'{self.plot_folder}/against_{xlabel}.png')
        plt.close()

    @ut.timer
    def plot_against(self, xlabel, ylabels):
        self.plot_against_mean(xlabel, ylabels)
        self.plot_against_std(xlabel, ylabels)


    