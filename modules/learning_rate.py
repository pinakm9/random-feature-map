import numpy as np 
from scipy import stats
from scipy.signal import argrelextrema
from sklearn.linear_model import HuberRegressor
from sklearn.preprocessing import StandardScaler


class AdaptiveRate:
    def __init__(self, optimizer, initial_rate, initial_scenario, update_interval=100, del_loss=0.1, fluc_lim=0.2, del_rate=0.1):
        self.optimizer = optimizer
        self.del_loss = del_loss
        self.fluc_lim = fluc_lim
        self.interval = update_interval
        self.del_rate = del_rate
        self.past_rate = initial_rate
        self.past_scenario = initial_scenario
        self.slope = -1.
        self.fluctuations = 0
        # self.x_scaler, self.y_scaler = StandardScaler(), StandardScaler()




    def get_scenario(self, loss_history):
        # x_train = self.x_scaler.fit_transform(np.arange(self.interval)[..., None])
        # y_train = self.y_scaler.fit_transform(loss_history[..., None])
        # # fit model
        # model = HuberRegressor(epsilon=1)
        # model.fit(np.arange(self.interval).reshape(-1, 1), loss_history)
        
        maxima = argrelextrema(loss_history, np.greater)[0]

        self.fluctuations = len(maxima) / len(loss_history) 
        x, y = [], []
        for i in range(len(loss_history)):
            if i not in maxima:
                y.append(loss_history[i])
                x.append(i)

        self.slope = stats.linregress(np.array(x), np.array(y)).slope

        self.fluc_ok = self.fluctuations < self.fluc_lim 
        self.slope_ok = -self.slope > self.del_loss * y[0] / self.interval
        if self.slope_ok:
            return 0 # keep the current learning rate
        elif self.slope > 0:
            return -1
        elif self.fluc_ok:
            return 1 # increase the the current learning rate
        else:
            return -1 # decrease the current learning rate

        

    def step(self, loss_history):
        curr_rate = self.optimizer.param_groups[0]['lr']
        scenario = self.get_scenario(loss_history)
        if scenario == 0:
            new_rate = curr_rate
        elif scenario == 1:
            # if self.past_scenario == -1:
            #     new_rate = (curr_rate + self.past_rate) / 2.
            # else:
            new_rate = (1. + self.del_rate) * curr_rate
        else:
            if self.past_scenario == 1:
                new_rate = (curr_rate + self.past_rate) / 2.
            else:
                new_rate = (1 - self.del_rate) * curr_rate
        self.optimizer.param_groups[0]['lr'] = new_rate
        self.past_rate = curr_rate
        self.past_scenario = scenario
            
 

        

class AdaptiveRateBS:
    def  __init__(self, model, max_delta=0.1, min_change=-0.01, constant_rate=True, update_frequency=100, mode='BS'):
        self.model = model
        self.max_delta = max_delta
        self.min_change = min_change
        self.constant_rate = constant_rate
        self.update_frequency = update_frequency
        self.iter = 0
        self.mode = mode
    

    def descent(self, rate, loss_0, *loss_params):
        self.model.optimizer.param_groups[0]['lr'] = rate
        self.model.optimizer.zero_grad()
        loss_ = self.model.loss_fn(*loss_params)
        # Backpropagation
        loss_.backward()
        self.model.optimizer.step()
        loss_ = self.model.loss_fn(*loss_params)
        self.iter += 1
        return (loss_.item()-loss_0) / loss_0


    def increase(self, loss_0, *loss_params):
        left = self.model.optimizer.param_groups[0]['lr']
        if self.mode == 'BS':
            right = left * (1. + 0.)
            change = -1.
            i = 0
            while change < 0. and i < 100:
                right *= (1 + 1.*self.max_delta)
                change = self.descent(right, loss_0, *loss_params)
                i += 1
                print(f'Attempting to reverse change: attempt #{i}: new_rate: {right:.6f}, change: {change:.6f}')
            i = 0
            while change > self.min_change:
                middle = (left + right) / 2.
                change = self.descent(middle, loss_0, *loss_params)
                if change < 0.:
                    left = middle + 0.
                else:
                    right = middle + 0.
                i += 1
                print(f'Performing bisection: attempt #{i}: new_rate: {middle:.6f}, change: {change:.6f}')
                if np.abs(left-right) < 1e-7:
                    break
            return middle
        elif self.mode == 'simple':
            return left*(1.+ self.max_delta)
        elif self.mode == 'simple-random':
            return left*(1.+np.random.uniform()*self.max_delta)
        
        
    def decrease(self, loss_0, *loss_params):
        left = self.model.optimizer.param_groups[0]['lr']
        if self.mode == 'BS':
            right = left * (1. - 0.)
            change = +1.
            i = 0
            while change > 0. and i < 100:
                right *= (1 - 1.*self.max_delta)
                change = self.descent(right, loss_0, *loss_params)
                i += 1
                print(f'Attempting to reverse change: attempt #{i}: new_rate: {right:.6f}, change: {change:.6f}')
            i = 0
            if change <= self.min_change:
                return right
            while change > self.min_change:
                middle = (left + right) / 2.
                change = self.descent(middle, loss_0, *loss_params)
                if change < 0.:
                    right = middle + 0.
                else:
                    left = middle + 0.
                i += 1
                print(f'Performing bisection: attempt #{i}: new_rate: {middle:.6f}, change: {change:.6f}')
                if np.abs(left-right) < 1e-7:
                    break
            return middle
        elif self.mode == 'simple':
            return left*(1-self.max_delta)
        elif self.mode == 'simple-random':
            return left*(1-np.random.uniform()*self.max_delta)

    def step(self, sign, loss_0, *loss_params):
        if not self.constant_rate:
            if sign > 0.:
                print("--------------------------------------------------------------")
                print("             Attempting to increase learning rate             ")
                print("--------------------------------------------------------------")
                new_rate = self.increase(loss_0, *loss_params)
                self.model.optimizer.param_groups[0]['lr'] = new_rate
            if sign < 0.:
                print("--------------------------------------------------------------")
                print("             Attempting to decrease learning rate             ")
                print("--------------------------------------------------------------")
                new_rate = self.decrease(loss_0, *loss_params)
                self.model.optimizer.param_groups[0]['lr'] = new_rate
            # self.model.load(idx)
        
        
        