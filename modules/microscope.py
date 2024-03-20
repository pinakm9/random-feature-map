import surrogate as sr 
import sample as sm
import os, json
import numpy as np
import utility as ut
import pandas as pd
import ipywidgets as widgets
from ipywidgets import interact
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy import stats

class MicroscopeExtremeToGood:

    """
    Description: runs experiments for fixed D_r, starting with all extreme rows and then switching off 
                 extreme rows one by one
    """
    def __init__(self, save_folder, D, D_r, beta, error_theshold, train, test, dt, Lyapunov_time=1.,\
                 L0=0.4, L1=3.5, limits_W_in=[-0.1, 0.1], limits_W=[-0.5, 0.5]):
        """
        Args:
            save_folder: folder where all the generated data will be saved
            D: dimension of the underlying dynamical system
            D_r: dimension of the reservoir
            beta: regularization parameters for ridge regression
            train: training data
            test: trajectories for calculating tau_f
            error_threshold: threshold to determine forecast time
            dt: time step of the surrogate model
            Lyapunov_time: Lyapunov time for the underlying dynamical system, default=1
            L0: lower bound for determining good rows
            L1: upper bound for determining good rows
            limits_W_in: limits for counting zero rows of W_in 
            limits_W: limits for counting zero columns of W
        """
        self.D = D
        self.D_r = D_r

        self.n_models = D_r
        self.beta = beta
        self.train = train
        self.test = test
        self.bad_feature_count_trajectory = test.reshape((test.shape[1], -1))[:, :1600]
        self.error_threshold = error_theshold
        self.dt = dt 
        self.Lyapunov_time = Lyapunov_time
        
        self.L0 = L0
        self.L1 = L1
        self.sampler = sm.MatrixSampler(L0, L1, train.T)
        self.good_row_sampler = sm.GoodRowSampler(L0, L1, train.T)
       
        
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
                  'error_threshold': error_theshold,  'n_models': self.n_models, \
                  'L0':L0, 'L1':L1, 'training_points': self.train.shape[-1], \
                  'validation_points': self.test.shape[-1],\
                  'n_validation_trajectories': len(self.test), 'full_train_trajectory_length': self.train.shape[1],\
                  'bad_feature_count_trajectory_length': self.bad_feature_count_trajectory.shape[-1],\
                  'limits_W_in': list(limits_W_in), 'limits_W': list(limits_W)}

        with open('{}/config.json'.format(self.save_folder), 'w') as f:
            f.write(json.dumps(config))

        self.W_in, self.b_in = self.sampler.sample_([0, 0, self.D_r])
    


    def compute_error(self, model):
        """
        Description: computes forecast time tau_f for the computed surrogate model

        Args:
            validation_index: index for selecting validation trajectory
            rmse_threshold: threshold to determine forecast time
        """
        se, mse = 0., 0.
        for validation_index in range(len(self.test)):
            validation_ = self.test[validation_index]
            prediction = model.multistep_forecast(validation_[:, 0], self.validation_points)
            se_ = np.linalg.norm(validation_ - prediction, axis=0)**2 / np.linalg.norm(validation_, axis=0)**2
            mse_ = np.cumsum(se_) / np.arange(1, len(se_)+1)
            se += se_
            mse += mse_
        
        se /= len(self.test)
        mse /= len(self.test)
   
        
        l = np.argmax(mse > self.error_threshold)
        if l == 0:
            tau_f_rmse = self.validation_points
        else:
            tau_f_rmse = l-1


        l = np.argmax(se > self.error_threshold)
        if l == 0:
            tau_f_se = self.validation_points
        else:
            tau_f_se = l-1
        
        rmse = np.sqrt(mse[-1])
        se = se[-1]
 
            
        
        tau_f_rmse *= (self.dt / self.Lyapunov_time)
        tau_f_se *= (self.dt / self.Lyapunov_time)

        return tau_f_rmse, tau_f_se, rmse, se

        


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

    @ut.timer
    def run_single(self, l):
        """
        Description: runs experiments for a single value of (w, b) multiple times 

        Args: 
            l: index/identifier for the single experiment
        """
        if l > 0:
            self.W_in[l], self.b_in[l] = self.good_row_sampler.sample_()
        model = sr.SurrogateModel_LR(self.D, self.D_r, self.W_in, self.b_in)
        model.compute_W(self.train, beta=self.beta);
   
        np.savetxt(self.W_in_folder + '/W_in_{}.csv'.format(l), model.W_in, delimiter=',')
        np.savetxt(self.b_in_folder + '/b_in_{}.csv'.format(l), model.b_in, delimiter=',')
        np.savetxt(self.W_folder + '/W_{}.csv'.format(l), model.W, delimiter=',')
        
        results = [l, np.linalg.norm(model.W_in)/self.D_r,\
                   np.linalg.norm(model.b_in)/self.D_r, np.linalg.norm(model.W)/self.D_r] +\
                   [l/float(self.D_r), 0., 1. - l/float(self.D_r)]
        results += self.compute_error(model)
        results += self.count(model)

        return results
    

    @ut.timer
    def run(self):
        """
        Description: runs all the experiments, documents W_in, b_in, W, forecast times and errors
        """
        file_path = '{}/batch_data.csv'.format(self.save_folder)
        # if os.path.exists(file_path):
        #     os.remove(file_path)
        columns = ['l', '||W_in||', '||b_in||', '||W||', 'good_rows_W_in', 'linear_rows_W_in', 'extreme_rows_W_in',\
                   'tau_f_rmse', 'tau_f_se', 'rmse', 'se',\
                   '0_rows_W_in', '0_cols_W', 'avg_bad_features', 'avg_abs_col_sum_W']
        data = []
        for l in range(self.D_r):
            print(f'Running experiment #{l} ...')
            data += [self.run_single(l)]
            # print('Documenting results ...')#, len(columns), len(data[0]))
        pd.DataFrame(data, columns=columns, dtype=float).to_csv(file_path, mode='w', index=False)
       




class MicroscopeExtremeToGoodViewer:
    """
    Description: A class for analyzing the results of a batch run
    """
    def __init__(self, save_folder, n_plot_rows) -> None:
        """
        Args: 
            save_folder: folder where the results of batch run experiments were stored
            n_plot_rows: number of rows to plot
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
        self.m = n_plot_rows
        self.n = int(np.ceil(self.D_r/float(self.m)))

        self.idx = -np.ones((self.m, self.n), dtype=int)

        l = 0
        for i in range(self.m):
            for j in range(self.n):
                self.idx[i][j] = l
                l += 1 
        
        self.data = self.get_data()
        
    
    def get_config(self):
        with open('{}/config.json'.format(self.save_folder), 'r') as f:
            config = json.loads(f.read())
        for key in config:
            setattr(self, key, config[key])

                    

    @ut.timer
    def get_mean_std(self, quantity):
        data = pd.read_csv('{}/batch_data.csv'.format(self.save_folder))
        array = data[quantity]
        mean, std = np.mean(array), np.std(array)
        return mean, std
    
    def get_W_in(self, l):
        return np.genfromtxt(self.W_in_folder + '/W_in_{}.csv'.format(l), delimiter=',')
    
    def get_b_in(self, l):
        return np.genfromtxt(self.b_in_folder + '/b_in_{}.csv'.format(l), delimiter=',')
    
    def get_W(self, l):
        return np.genfromtxt(self.W_folder + '/W_{}.csv'.format(l), delimiter=',')
    
    
    def get_model(self, l):
        W_in_fn = lambda d, d_r: np.genfromtxt(self.W_in_folder + '/W_in_{}.csv'.format(l), delimiter=',')
        b_in_fn  = lambda d_r: np.genfromtxt(self.b_in_folder + '/b_in_{}.csv'.format(l), delimiter=',')
        W = np.genfromtxt(self.W_folder + '/W_{}.csv'.format(l), delimiter=',')
        model = sr.SurrogateModel_LR(self.D, self.D_r, W_in_fn, b_in_fn, W)
        return model


    def get_data(self):
        data = pd.read_csv('{}/batch_data.csv'.format(self.save_folder))
        return data
    
    def get_nonzero_cols(self, l, limits):
        W = self.get_W(l)
        W[(W > limits[0]) & (W < limits[1])] = 0. 
        return W.T.any(1)

    def get_zero_sep(self, l, cols):
        zc = ~cols
        gc = 0
        for i in range(l):
            if zc[i]:
                gc += 1
        return gc, zc.sum()-gc
       
    def plot1(self, l, limits):
        fig = plt.figure(figsize=(13, 5))
        ax_W_in = fig.add_subplot(121)
        ax_W = fig.add_subplot(122)

        lights = np.array([1] * l + [0] * (self.D_r - l)).reshape(self.m, self.n)
        ax_W_in.matshow(lights, cmap=ListedColormap(['white']))
        ax_W_in.axis('off')
        h = 1.8
        for (i, j), z in np.ndenumerate(self.idx):
            ax_W_in.text(h*j, h*i, '{}'.format(z), ha='center', va='center',\
                     bbox=dict(boxstyle='round', facecolor='None' if lights[i][j] else 'red', edgecolor='0.3', pad=0.1), fontsize=10)
        ax_W_in.set_xticks([])
        ax_W_in.set_yticks([])
        ax_W_in.set_title(r'rows of $W_{in}$', fontsize=12)

        ax_W.matshow(lights, cmap=ListedColormap(['white']))
        ax_W.axis('off')

        cols = self.get_nonzero_cols(l, limits).reshape(self.m, self.n)
        for (i, j), z in np.ndenumerate(self.idx):
            ax_W.text(h*j, h*i, '{}'.format(z), ha='center', va='center',\
                     bbox=dict(boxstyle='circle', facecolor='None' if cols[i][j] else 'red', edgecolor='0.3', pad=0.1), fontsize=10)
        ax_W.set_xticks([])
        ax_W.set_yticks([])
        bf = self.data['avg_bad_features'][l]
        er = self.data['rmse'][l]**2
        wn = self.data['||W||'][l]
        ax_W.set_title(r'columns of $W$', fontsize=12)
        fig.suptitle(f'avg bad features={bf:.2f}, error={er:.2f}, # zero cols={self.D_r-cols.flatten().sum()}, ||W||={wn:.2f}')
        plt.show()


    def view1(self, limits=None):
        if limits is None:
            limits = self.limits_W
        good_rows = widgets.IntSlider(min=0, max=self.D_r-1, step=1, layout=widgets.Layout(width='1100px'),\
                                         description='good rows')
        interact(lambda l: self.plot1(l, limits), l=good_rows)
        
   
    def plot2(self, l, limits):
        fig = plt.figure(figsize=(15, 5))
        ax_1 = fig.add_subplot(311)
        ax_2 = fig.add_subplot(312)
        ax_3 = fig.add_subplot(313)
        axs = [ax_1, ax_2, ax_3]


        W = np.abs(self.get_W(l))
        cols = self.get_nonzero_cols(l, limits)
        colors = ['None' if e else 'red' for e in cols]
        gc, bc = self.get_zero_sep(l, cols)
        
        for i in range(3):
            c = np.max(W[i, :])
            axs[i].plot(range(self.D_r), W[i, :]/c)
            axs[i].scatter(range(self.D_r), W[i, :]/c, c=colors)
            axs[i].set_ylabel(f'row {i+1} of |W|')

        
        bf = self.data['avg_bad_features'][l]
        er = self.data['rmse'][l]**2
        wn = self.data['||W||'][l]
        zc = self.D_r-cols.sum()

        fig.suptitle(f'avg bad features={bf:.2f}, error={er:.2f}, # zero cols={zc}, ||W||={wn:.2f}\n\
                      # zero cols in good part={gc}, # zero cols in bad part={bc}')
        plt.show()

    def plot3(self, l, c0, c1, l0, l1, limits=[-0.5, 0.5], file_path=None):
        fig = plt.figure(figsize=(15, 5))
        ax_1 = fig.add_subplot(311)
        ax_2 = fig.add_subplot(312)
        ax_3 = fig.add_subplot(313)
        axs = [ax_1, ax_2, ax_3]


        W = np.abs(self.get_W(l))
        cols = self.get_nonzero_cols(l, limits)
        colors = ['None' if e else 'red' for e in cols]
        gc, bc = self.get_zero_sep(l, cols)
        
        for i in range(3):
            c = np.max(W[i, :])
            axs[i].axvline(l, c='black')
            axs[i].plot(range(1, l+1), W[i, :l]/c, c=c0, label=l0)
            axs[i].plot(range(l, self.D_r+1), W[i, l-1:]/c, c=c1, label=l1)
            axs[i].scatter(range(1, self.D_r+1), W[i, :]/c, c=colors, s=10)
            axs[i].set_ylabel(r'$\frac{{|\mathbf{{W}}_{{{0}}}^{{(s)}}|}}{{\|\mathbf{{W}}_{{{0}}}^{{(s)}}\|_\infty}}$'.format(i+1), fontsize=18)
            axs[i].set_xlim((0, self.D_r+1))
            
        
        # locs, labels = axs[-1].get_xticks(), axs[-1].get_xticklabels()
        # locs = list(locs) + [l]
        # labels = list(labels) + ['s']
        # axs[-1].set_xticks(locs)
        # axs[-1].set_xticklabels(labels)
        # #(f's={l}', x=float(l)/self.D_r)
        axs[-1].set_xlabel('column')
        axs[0].set_title(f's={l}')
        axs[0].legend()
        bf = self.data['avg_bad_features'][l]
        er = self.data['rmse'][l]**2
        wn = self.data['||W||'][l]
        # zc = self.D_r-cols.sum()

        # fig.suptitle(f'avg bad features={bf:.2f}, error={er:.2f}, # zero cols={zc}, ||W||={wn:.2f}\n\
        #               # zero cols in good part={gc}, # zero cols in bad part={bc}')
        if file_path is not None:
            plt.savefig(file_path, dpi=300, bbox_inches='tight')
        plt.show()

    


    def view2(self, limits=None):
        if limits is None:
            limits = self.limits_W
        good_rows = widgets.IntSlider(min=0, max=self.D_r-1, step=1, layout=widgets.Layout(width='1100px'),\
                                         description='good rows', continuous_update=False)
        out = widgets.interactive(lambda l: self.plot2(l, limits), l=good_rows)
        output = out.children[-1]
        output.layout.height = '600px'
        return out


    def zc_vs_good(self, limits):
        fig = plt.figure(figsize=(12, 4))
        ax = fig.add_subplot(131)
        ax2 = fig.add_subplot(132)
        ax1 = fig.add_subplot(133)
        zc = np.ones(self.D_r)
        gc = np.ones(self.D_r)
        bc = np.ones(self.D_r)
        for i in range(self.D_r):
            cols = self.get_nonzero_cols(i, limits)
            zc[i] = self.D_r - cols.sum()
            gc[i], bc[i] = self.get_zero_sep(i, cols) 
        
        
        x = np.array(list(range(self.D_r))) / self.D_r
        zc /= self.D_r
        gc /= self.D_r 
        bc /= self.D_r 
        sg = stats.linregress(x, gc).slope
        p, q = np.argmax(zc), np.argmax(zc[::-1])
        k = max(p, len(zc)-1-q)
        sbl = stats.linregress(x[:k], bc[:k]).slope
        sbr = stats.linregress(x[k:], bc[k:]).slope

        ax.plot(x, zc)
        ax.set_xlabel('# good rows')
        ax.set_ylabel('# zero columns')

        ax2.plot(x, gc, label=f'# zc in good, slope={sg:.2f}')
        ax2.plot(x, bc, label=f'# zc in bad, slopes={sbl:.2f}, {sbr:.2f}')
        ax2.set_xlabel('# good rows')
        ax2.set_ylabel('# zero columns')
        ax2.legend()

        ax1.scatter(zc[10:], (self.data['rmse'][10:])**2)
        ax1.set_ylabel('error')
        ax1.set_xlabel('# zero columns')

        line1 = fr'Before maxima # zc in bad = {bc[k-30]:.2f}, $p_b$={1.-x[k-30]:.2f}'
        line2 = fr'At maxima # zc in bad = {bc[k]:.2f}, $p_b$={1.-x[k]:.2f}'
        line3 = fr'After maxima # zc in bad = {bc[k+30]:.2f}, $p_b$={1.-x[k+30]:.2f}'
        fig.text(0., -0.2, line1+ '\n' + line2 + '\n'+ line3)
     
        plt.show()



class MicroscopeLinearToGood(MicroscopeExtremeToGood):
    def __init__(self, save_folder, D, D_r, beta, error_theshold, train, test, dt, Lyapunov_time=1, L0=0.4, L1=3.5, limits_W_in=[-0.1, 0.1], limits_W=[-0.5, 0.5]):
        super().__init__(save_folder, D, D_r, beta, error_theshold, train, test, dt, Lyapunov_time, L0, L1, limits_W_in, limits_W)
        self.W_in, self.b_in = self.sampler.sample_([0, self.D_r, 0])

    @ut.timer
    def run_single(self, l):
        """
        Description: runs experiments for a single value of (w, b) multiple times 

        Args: 
            l: index/identifier for the single experiment
        """
        if l > 0:
            self.W_in[l], self.b_in[l] = self.good_row_sampler.sample_()
        model = sr.SurrogateModel_LR(self.D, self.D_r, self.W_in, self.b_in)
        model.compute_W(self.train, beta=self.beta);
   
        np.savetxt(self.W_in_folder + '/W_in_{}.csv'.format(l), model.W_in, delimiter=',')
        np.savetxt(self.b_in_folder + '/b_in_{}.csv'.format(l), model.b_in, delimiter=',')
        np.savetxt(self.W_folder + '/W_{}.csv'.format(l), model.W, delimiter=',')
        
        results = [l, np.linalg.norm(model.W_in)/self.D_r,\
                   np.linalg.norm(model.b_in)/self.D_r, np.linalg.norm(model.W)/self.D_r] +\
                   [l/float(self.D_r), 1. - l/float(self.D_r), 0.]
        results += self.compute_error(model)
        results += self.count(model)

        return results
    

class MicroscopeLinearToGoodViewer(MicroscopeExtremeToGoodViewer):
    """
    Description: A class for analyzing the results of a batch run
    """
    def __init__(self, save_folder, n_plot_rows) -> None:
        super().__init__(save_folder, n_plot_rows)
    

    

   
    def plot2(self, l, limits):
        fig = plt.figure(figsize=(15, 5))
        ax_1 = fig.add_subplot(311)
        ax_2 = fig.add_subplot(312)
        ax_3 = fig.add_subplot(313)
        axs = [ax_1, ax_2, ax_3]


        W = np.abs(self.get_W(l))
        cols = self.get_nonzero_cols(l, limits)
        colors = ['None' if e else 'red' for e in cols]
        gc, bc = self.get_zero_sep(l, cols)
        
        for i in range(3):
            c = np.max(W[i, :])
            axs[i].plot(range(self.D_r), W[i, :]/c)
            axs[i].scatter(range(self.D_r), W[i, :]/c, c=colors)
            axs[i].set_ylabel(f'row {i+1} of |W|')

        
        bf = self.data['avg_bad_features'][l]
        er = self.data['rmse'][l]**2
        wn = self.data['||W||'][l]
        zc = self.D_r-cols.sum()

        fig.suptitle(f'avg bad features={bf:.2f}, error={er:.2f}, # zero cols={zc}, ||W||={wn:.2f}\n\
                      # zero cols in good part={gc}, # zero cols in linear part={bc}')
        plt.show()




    def zc_vs_good(self, limits):
        fig = plt.figure(figsize=(12, 4))
        ax = fig.add_subplot(131)
        ax2 = fig.add_subplot(132)
        ax1 = fig.add_subplot(133)
        zc = np.ones(self.D_r)
        gc = np.ones(self.D_r)
        bc = np.ones(self.D_r)
        for i in range(self.D_r):
            cols = self.get_nonzero_cols(i, limits)
            zc[i] = self.D_r - cols.sum()
            gc[i], bc[i] = self.get_zero_sep(i, cols) 
        
        
        x = np.array(list(range(self.D_r))) / self.D_r
        zc /= self.D_r
        gc /= self.D_r 
        bc /= self.D_r 
        sg = stats.linregress(x, gc).slope
        p, q = np.argmax(zc), np.argmax(zc[::-1])
        k = max(p, len(zc)-1-q)
        sbl = stats.linregress(x[:k], bc[:k]).slope
        sbr = stats.linregress(x[k:], bc[k:]).slope

        ax.plot(x, zc)
        ax.set_xlabel('# good rows')
        ax.set_ylabel('# zero columns')

        ax2.plot(x, gc, label=f'# zc in good, slope={sg:.2f}')
        ax2.plot(x, bc, label=f'# zc in linear, slopes={sbl:.2f}, {sbr:.2f}')
        ax2.set_xlabel('# good rows')
        ax2.set_ylabel('# zero columns')
        ax2.legend()

        ax1.scatter(zc[10:], (self.data['rmse'][10:])**2)
        ax1.set_ylabel('error')
        ax1.set_xlabel('# zero columns')

        line1 = fr'Before maxima # zc in linear = {bc[k-30]:.2f}, $p_b$={1.-x[k-30]:.2f}'
        line2 = fr'At maxima # zc in linear = {bc[k]:.2f}, $p_b$={1.-x[k]:.2f}'
        line3 = fr'After maxima # zc in linear = {bc[k+30]:.2f}, $p_b$={1.-x[k+30]:.2f}'
        fig.text(0., -0.2, line1+ '\n' + line2 + '\n'+ line3)
     
        plt.show() 



class MicroscopeMildToGood(MicroscopeLinearToGood):
    def __init__(self, save_folder, D, D_r, beta, error_theshold, train, test, dt, Lyapunov_time=1, L0=0.4, L1=3.5, limits_W_in=[-0.1, 0.1], limits_W=[-0.5, 0.5]):
        super().__init__(save_folder, D, D_r, beta, error_theshold, train, test, dt, Lyapunov_time, L0, L1, limits_W_in, limits_W)
        mild_matrix_sampler = sm.MatrixSampler(L1, L1, train.T)
        self.W_in, self.b_in = mild_matrix_sampler.sample_([0, self.D_r, 0])

    @ut.timer
    def run_single(self, l):
        """
        Description: runs experiments for a single value of (w, b) multiple times 

        Args: 
            l: index/identifier for the single experiment
        """
        if l > 0:
            self.W_in[l], self.b_in[l] = self.good_row_sampler.sample_()
        model = sr.SurrogateModel_LR(self.D, self.D_r, self.W_in, self.b_in)
        model.compute_W(self.train, beta=self.beta);

        np.savetxt(self.W_in_folder + '/W_in_{}.csv'.format(l), model.W_in, delimiter=',')
        np.savetxt(self.b_in_folder + '/b_in_{}.csv'.format(l), model.b_in, delimiter=',')
        np.savetxt(self.W_folder + '/W_{}.csv'.format(l), model.W, delimiter=',')
        
        results = [l, np.linalg.norm(model.W_in)/self.D_r,\
                    np.linalg.norm(model.b_in)/self.D_r, np.linalg.norm(model.W)/self.D_r] +\
                    [l/float(self.D_r), 1. - l/float(self.D_r), 0.]
        results += self.compute_error(model)
        results += self.count(model)

        return results


class MicroscopeMildToGoodViewer(MicroscopeLinearToGoodViewer):
    """
    Description: A class for analyzing the results of a batch run
    """
    def __init__(self, save_folder, n_plot_rows) -> None:
        super().__init__(save_folder, n_plot_rows)
    

    def plot2(self, l, limits):
        fig = plt.figure(figsize=(15, 5))
        ax_1 = fig.add_subplot(311)
        ax_2 = fig.add_subplot(312)
        ax_3 = fig.add_subplot(313)
        axs = [ax_1, ax_2, ax_3]


        W = np.abs(self.get_W(l))
        cols = self.get_nonzero_cols(l, limits)
        colors = ['None' if e else 'red' for e in cols]
        gc, bc = self.get_zero_sep(l, cols)
        
        for i in range(3):
            c = np.max(W[i, :])
            axs[i].plot(range(self.D_r), W[i, :]/c)
            axs[i].scatter(range(self.D_r), W[i, :]/c, c=colors)
            axs[i].set_ylabel(f'row {i+1} of |W|')

        
        bf = self.data['avg_bad_features'][l]
        er = self.data['rmse'][l]**2
        wn = self.data['||W||'][l]
        zc = self.D_r-cols.sum()

        fig.suptitle(f'avg bad features={bf:.2f}, error={er:.2f}, # zero cols={zc}, ||W||={wn:.2f}\n\
                      # zero cols in good part={gc}, # zero cols in mild part={bc}')
        plt.show()




    def zc_vs_good(self, limits):
        fig = plt.figure(figsize=(12, 4))
        ax = fig.add_subplot(131)
        ax2 = fig.add_subplot(132)
        ax1 = fig.add_subplot(133)
        zc = np.ones(self.D_r)
        gc = np.ones(self.D_r)
        bc = np.ones(self.D_r)
        for i in range(self.D_r):
            cols = self.get_nonzero_cols(i, limits)
            zc[i] = self.D_r - cols.sum()
            gc[i], bc[i] = self.get_zero_sep(i, cols) 
        
        
        x = np.array(list(range(self.D_r))) / self.D_r
        zc /= self.D_r
        gc /= self.D_r 
        bc /= self.D_r 
        sg = stats.linregress(x, gc).slope
        p, q = np.argmax(zc), np.argmax(zc[::-1])
        k = max(p, len(zc)-1-q)
        sbl = stats.linregress(x[:k], bc[:k]).slope
        sbr = stats.linregress(x[k:], bc[k:]).slope

        ax.plot(x, zc)
        ax.set_xlabel('# good rows')
        ax.set_ylabel('# zero columns')

        ax2.plot(x, gc, label=f'# zc in good, slope={sg:.2f}')
        ax2.plot(x, bc, label=f'# zc in mild, slopes={sbl:.2f}, {sbr:.2f}')
        ax2.set_xlabel('# good rows')
        ax2.set_ylabel('# zero columns')
        ax2.legend()

        ax1.scatter(zc[10:], (self.data['rmse'][10:])**2)
        ax1.set_ylabel('error')
        ax1.set_xlabel('# zero columns')

        line1 = fr'Before maxima # zc in mild = {bc[k-30]:.2f}, $p_b$={1.-x[k-30]:.2f}'
        line2 = fr'At maxima # zc in mild = {bc[k]:.2f}, $p_b$={1.-x[k]:.2f}'
        line3 = fr'After maxima # zc in mild = {bc[k+30]:.2f}, $p_b$={1.-x[k+30]:.2f}'
        fig.text(0., -0.2, line1+ '\n' + line2 + '\n'+ line3)
     
        plt.show()