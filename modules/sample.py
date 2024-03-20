import os, sys 
from pathlib import Path
from os.path import dirname, realpath
script_dir = Path(dirname(realpath('.')))
module_dir = str(script_dir)
sys.path.insert(0, module_dir + '/modules')
import numpy as np
import matplotlib.pyplot as plt
import utility as ut
from joblib import Parallel, delayed

class InequalitySampler0:
    """
    Description: Class for sampling convex regions determined by  b0 < Ax < b1, condition: b0 < 0 < b1
    """
    def __init__(self, A, b0, b1) -> None:
        self.A = np.array(A)
        self.b0 = np.array(b0)
        self.b1 = np.array(b1)
        self.dim = self.A.shape[-1]
        self.x0 = np.zeros(self.dim)

    def is_feasible(self, x):
        """
        Description: Determines if an input belongs to the feasible region
        """
        y = self.A@x
        return ((y < self.b1) & (y > self.b0)).all()
    
    def is_feasible_m(self, X):
        """
        Description: Determines if multiple inputs belong to the feasible region
        """
        return [self.is_feasible(x) for x in X]

    
    def intersection_with_bisection(self, x0, d, tol=1e-2, max_iters=100):
        """
        Description: Finds the intersection of a line originating from x0 (lying inside the feasible region) having direction d with the feasible region
                    within the specified tolerance within the max_iters iterations
        """
        
        self.x0 = x0 - np.dot(x0, d) * d
        
        if self.is_feasible(self.x0):
            self.int_pts, self.weights = np.zeros((2, self.dim)), np.zeros(2)
            for i, t in enumerate([-1., 1.]):
                while self.is_feasible(self.x0 + t*d):
                    t *= 2 

                l, r = 0., t
                iter = 0

                while abs(r-l) > tol and iter < max_iters:
                    m = (r+l)/2.
                    if not self.is_feasible(self.x0 + m*d):
                        r = m
                    else:
                        l = m
                    iter += 1
                
                self.int_pts[i, :] = self.x0 + l*d
                self.weights[i] = l
            
            return 2
        else:
           self.int_pts, self.weights = [], []
           return 0
    

    def single_sample(self, x, steps, **kwargs):
        for _ in range(steps):
            # flag = True
            # while flag:
                # generate a random direction 
            d = np.random.normal(size=self.dim)
            d /= np.linalg.norm(d)
            # find the intersection points 
            while self.intersection_with_bisection(x, d, **kwargs) == 0:
                d = np.random.normal(size=self.dim)
                d /= np.linalg.norm(d)
            t = np.random.uniform(*np.sort(self.weights))
            x = self.x0 + t*d
                # if self.is_feasible(y):
                #     x = y
                #     flag = False
        return x
            

    @ut.timer
    def sample(self, n, steps=10, **kwargs):
        """
        Description: Samples n points from the feasible region with kwargs specified for intersection_with_bisection
        """
        self.X = np.zeros((n, self.dim))
        x = np.zeros(self.dim)
        for i in range(n):
            x = self.single_sample(x, steps, **kwargs)
            self.X[i] = x
        return self.X 
    
    def show_feasible(self, i=0, j=1, lims=(-10., 10.), resolution=300, s=5, alpha=1.0):
        """
        Description: Shows feasible region projected on to i < j th coordinates 
        """
        lims = np.array(lims)
        if len(lims.shape) < 2:
            grid = np.meshgrid(*([np.linspace(lims[0], lims[1], resolution)] * self.dim))
            xmin, xmax = lims
            ymin, ymax = lims 
        else:
            grid = np.meshgrid(*[np.linspace(lims[i, 0], lims[i, 1], resolution) for i in range(self.dim)])
            xmin, xmax = lims[i]
            ymin, ymax = lims[j]
        expression = True
        for l, row in enumerate(self.A):
            expr = 0. 
            for k, e in enumerate(row):
                expr += e*grid[k]
            expression &= (expr < self.b1[l]) & (expr > self.b0[l])
        
        
        for k in range(self.dim):
            if k!=i and k!=j:
                expression = np.any(expression, axis=k)

        points = []
        for l in range(resolution):
            for m in range(resolution):
                if expression[l][m]:
                    points.append([grid[i][l][m], grid[j][l][m]])
        points = np.array(points)
        fig = plt.figure(figsize=(4, 4))
        ax = fig.add_subplot(111)
        ax.scatter(points[:, 0], points[:, 1], s=s, alpha=alpha, c='maroon', label='feasible region')
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.axis('equal')
        ax.legend()
        plt.show()
        return ax
    
    def show_intersection(self, i=0, j=1, lims=(-10., 10.), resolution=300, s=5, alpha=1.0):
        """
        Description: Shows feasible region projected on to i < j th coordinates with the current intersection
        """
        lims = np.array(lims)
        if len(lims.shape) < 2:
            grid = np.meshgrid(*([np.linspace(lims[0], lims[1], resolution)] * self.dim))
            xmin, xmax = lims
            ymin, ymax = lims 
        else:
            grid = np.meshgrid(*[np.linspace(lims[i, 0], lims[i, 1], resolution) for i in range(self.dim)])
            xmin, xmax = lims[i]
            ymin, ymax = lims[j]
        expression = True
        for l, row in enumerate(self.A):
            expr = 0. 
            for k, e in enumerate(row):
                expr += e*grid[k]
            expression &= (expr < self.b1[l]) & (expr > self.b0[l])
        
        for k in range(self.dim):
            if k!=i and k!=j:
                expression = np.any(expression, axis=k)
        points = []
        for l in range(resolution):
            for m in range(resolution):
                if expression[l][m]:
                    points.append([grid[i][l][m], grid[j][l][m]])
        points = np.array(points)
        fig = plt.figure(figsize=(4, 4))
        ax = fig.add_subplot(111)
        ax.scatter(points[:, 0], points[:, 1], s=s, alpha=alpha, c='maroon', label='feasible region')
        ax.scatter(self.x0[i], self.x0[j], s=20, label=r'initial point $x_0$', c='deeppink')
        ax.plot(self.int_pts[:, i], self.int_pts[:, j], label=r'line segment along random direction $d$')
        ax.scatter(self.int_pts[:, i], self.int_pts[:, j], s=20, label='intersection points', c='orange')
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.axis('equal')
        ax.legend()
        plt.show()
        return ax

    def show_sample(self, i=0, j=1, lims=(-10., 10.), resolution=300, s=5, alpha=1.0):
        """
        Description: Shows feasible region projected on to i < j th coordinates with the current sample
        """
        lims = np.array(lims)
        if len(lims.shape) < 2:
            grid = np.meshgrid(*([np.linspace(lims[0], lims[1], resolution)] * self.dim))
            xmin, xmax = lims
            ymin, ymax = lims 
        else:
            grid = np.meshgrid(*[np.linspace(lims[i, 0], lims[i, 1], resolution) for i in range(self.dim)])
            xmin, xmax = lims[i]
            ymin, ymax = lims[j]
        expression = True
        for l, row in enumerate(self.A):
            expr = 0. 
            for k, e in enumerate(row):
                expr += e*grid[k]
            expression &= (expr < self.b1[l]) & (expr > self.b0[l])
        
        for k in range(self.dim):
            if k!=i and k!=j:
                expression = np.any(expression, axis=k)
        points = []
        for l in range(resolution):
            for m in range(resolution):
                if expression[l][m]:
                    points.append([grid[i][l][m], grid[j][l][m]])
        points = np.array(points)
        fig = plt.figure(figsize=(4, 4))
        ax = fig.add_subplot(111)
        ax.scatter(points[:, 0], points[:, 1], s=s, alpha=alpha, c='maroon', label='feasible region')
        ax.scatter(self.X[:, i], self.X[:, j], s=5, label='sample', c='orange')
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.axis('equal')
        ax.legend()
        plt.show()
        return ax

    
    


class InequalitySampler1(InequalitySampler0):
    """
    Description: Class for sampling convex regions determined by  |Ax + b| < M, condition: |b| < M
    """
    def __init__(self, A, b, M) -> None:
        b = np.array(b)
        b0 = - b - M
        b1 = -b + M
        super().__init__(A, b0, b1)


class InequalitySampler2(InequalitySampler0):
    """
    Description: Class for sampling regions determined by m < |Ax + b| < M, proper sampling condition: m < |b| < M although this condition is not necessary for
    visualization
    """
    def __init__(self, A, b, m , M) -> None:
        self.A = np.array(A)
        self.b = np.array(b)
        self.m = m
        self.M  = M 
        self.big = InequalitySampler1(A, b, M)
        self.small = InequalitySampler1(A, b, m)
        self.dim = self.big.dim


    def is_feasible(self, x):
        y = np.abs(self.A@x + self.b)
        return ((y < self.M) & (y > self.m)).all()
    

    def intersection_with_bisection(self, x, d, resolution=1000, tol=1e-2, max_iters=100):
        """
        Description: Finds the intersection of a line originating from x0 (lying inside the feasible region) having direction d with the feasible region
                    within the specified tolerance within the max_iters iterations
        """
        if self.big.intersection_with_bisection(x, d, tol=1e-2, max_iters=100) == 0:
            self.int_pts, self.weights = [], []
            return 0
        self.x0 = self.big.x0
        t_range = np.linspace(self.big.weights[0]-1., self.big.weights[1]+1., num=resolution, endpoint=True)
        isf = [self.is_feasible(self.big.x0 + t*d) for t in t_range]
        # isf[0], isf[-1] = True, True
        flips = []
        flag = False
        for i, sign in enumerate(isf):
            if sign != flag:
                flips.append(i)
                flag = not flag
        # print('self.big.weights', self.big.weights)
        # print('t_range', t_range)
        # print('isf', isf)
        # print('flips', flips)
        self.int_pts = []
        self.weights = []
        # self.int_pts[0], self.int_pts[-1] = self.big.int_pts[0], self.big.int_pts[-1]
        # self.weights[0], self.weights[-1] = self.big.weights[0], self.big.weights[-1]

        for i, flip in enumerate(flips):
            l, r = t_range[flip-1], t_range[flip]
            iter = 0
            
            while abs(r-l) > tol and iter < max_iters:
                m = (r+l)/2.
                if not self.is_feasible(self.x0 + m*d):
                    r = m
                else:
                    l = m
                iter += 1
                
            self.int_pts.append(self.x0 + l*d)
            self.weights.append(l)
        
        self.int_pts = np.array(self.int_pts)
        self.weights = np.array(self.weights)

        self.chords, self.probs = [], []
        for i, a in enumerate(self.weights):
            if i%2 == 0:
                b = self.weights[i+1]
                self.chords.append([a, b])
                self.probs.append((b-a))
        
        self.chords = np.array(self.chords)
        self.probs = np.array(self.probs)
        self.probs /= self.probs.sum()
        # print('self.int_pts', self.int_pts)
        # print('self.weights', self.weights)
        
        return len(self.weights)

    def single_sample(self, x, steps, **kwargs):
        for _ in range(steps):
            # flag = True
            # while flag:
                # generate a random direction 
            d = np.random.normal(size=self.dim)
            d /= np.linalg.norm(d)
            # find the intersection points 
            while self.intersection_with_bisection(x, d, **kwargs) == 0:
                d = np.random.normal(size=self.dim)
                d /= np.linalg.norm(d)
            j = np.random.choice(list(range(len(self.chords))), p=self.probs)
            t = np.random.uniform(*self.chords[j])
            x = self.x0 + t*d
                # if self.is_feasible(y):
                #     x = y
                #     flag = False
        return x
    
  
    def show_feasible(self, i=0, j=1, lims=(-10., 10.), resolution=300, s=5, alpha=1.0, components=False):
        """
        Description: Shows feasible region projected on to i < j th coordinates 
        """
        lims = np.array(lims)
        if len(lims.shape) < 2:
            grid = np.meshgrid(*([np.linspace(lims[0], lims[1], resolution)] * self.dim))
            xmin, xmax = lims
            ymin, ymax = lims 
        else:
            grid = np.meshgrid(*[np.linspace(lims[i, 0], lims[i, 1], resolution) for i in range(self.dim)])
            # print(grid)
            xmin, xmax = lims[i]
            ymin, ymax = lims[j]
        expression = True
        expression_big = True
        expression_removed = True
        for l, row in enumerate(self.A):
            expr = self.b[l] 
            for k, e in enumerate(row):
                expr += e*grid[k]
            expr = np.abs(expr)
            expression &= (expr < self.M) & (expr > self.m)
            expression_big &= (expr < self.M)
            expression_removed &= (expr > self.m)
        
        for k in range(self.dim):
            if k!=i and k!=j:
                expression = np.any(expression, axis=k)
        
        points = []
        points_big = []
        points_removed = []
        for l in range(resolution):
            for m in range(resolution):
                if expression[l][m]:
                    points.append([grid[i][l][m], grid[j][l][m]])
                if expression_big[l][m]:
                    points_big.append([grid[i][l][m], grid[j][l][m]])
                if not expression_removed[l][m]:
                    points_removed.append([grid[i][l][m], grid[j][l][m]])
        points = np.array(points)
        points_big = np.array(points_big)
        points_removed = np.array(points_removed)


        fig = plt.figure(figsize=(4, 4))
        ax = fig.add_subplot(111)
        if components:
            ax.scatter(points_big[:, 0], points_big[:, 1], s=s, alpha=0.01, c='red', label='big region')
            ax.scatter(points_removed[:, 0], points_removed[:, 1], s=s, alpha=0.01, c='grey', label='removed region')
        ax.scatter(points[:, 0], points[:, 1], s=s, alpha=alpha, c='maroon', label='feasible region')
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.axis('equal')
        lg = ax.legend()
        for lh in lg.legendHandles: 
            lh.set_alpha(1.)
        plt.show()
        return ax
    
    def show_intersection(self, i=0, j=1, lims=(-10., 10.), resolution=300, s=5, alpha=1.0, components=False):
        """
        Description: Shows feasible region projected on to i < j th coordinates with the current intersection
        """
        lims = np.array(lims)
        if len(lims.shape) < 2:
            grid = np.meshgrid(*([np.linspace(lims[0], lims[1], resolution)] * self.dim))
            xmin, xmax = lims
            ymin, ymax = lims 
        else:
            grid = np.meshgrid(*[np.linspace(lims[i, 0], lims[i, 1], resolution) for i in range(self.dim)])
            xmin, xmax = lims[i]
            ymin, ymax = lims[j]
        expression = True
        expression_big = True
        expression_removed = True

        for l, row in enumerate(self.A):
            expr = self.b[l] 
            for k, e in enumerate(row):
                expr += e*grid[k]
            expr = np.abs(expr)
            expression &= (expr < self.M) & (expr > self.m)
            expression_big &= (expr < self.M)
            expression_removed &= (expr > self.m)
        
        for k in range(self.dim):
            if k!=i and k!=j:
                expression = np.any(expression, axis=k)
        
        points = []
        points_big = []
        points_removed = []
        for l in range(resolution):
            for m in range(resolution):
                if expression[l][m]:
                    points.append([grid[i][l][m], grid[j][l][m]])
                if expression_big[l][m]:
                    points_big.append([grid[i][l][m], grid[j][l][m]])
                if not expression_removed[l][m]:
                    points_removed.append([grid[i][l][m], grid[j][l][m]])
        points = np.array(points)
        points_big = np.array(points_big)
        points_removed = np.array(points_removed)


        fig = plt.figure(figsize=(4, 4))
        ax = fig.add_subplot(111)
        if components:
            ax.scatter(points_big[:, 0], points_big[:, 1], s=s, alpha=0.01, c='red', label='big region')
            ax.scatter(points_removed[:, 0], points_removed[:, 1], s=s, alpha=0.01, c='grey', label='removed region')
        ax.scatter(points[:, 0], points[:, 1], s=s, alpha=alpha, c='maroon', label='feasible region')
        ax.scatter(self.x0[i], self.x0[j], s=20, label=r'initial point $x_0$', c='deeppink')
        ax.plot(self.int_pts[:, i], self.int_pts[:, j], label=r'line segment along random direction $d$')
        ax.scatter(self.int_pts[:, i], self.int_pts[:, j], s=20, label='intersection points', c='orange')
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.axis('equal')
        lg = ax.legend()
        for lh in lg.legendHandles: 
            lh.set_alpha(1.)
        plt.show()
        return ax


    def show_sample(self, i=0, j=1, lims=(-10., 10.), resolution=300, s=5, alpha=1.0, components=False):
        """
        Description: Shows feasible region projected on to i < j th coordinates with the current sample
        """
        lims = np.array(lims)
        if len(lims.shape) < 2:
            grid = np.meshgrid(*([np.linspace(lims[0], lims[1], resolution)] * self.dim))
            xmin, xmax = lims
            ymin, ymax = lims 
        else:
            grid = np.meshgrid(*[np.linspace(lims[i, 0], lims[i, 1], resolution) for i in range(self.dim)])
            xmin, xmax = lims[i]
            ymin, ymax = lims[j]
        expression = True
        expression_big = True
        expression_removed = True

        for l, row in enumerate(self.A):
            expr = self.b[l] 
            for k, e in enumerate(row):
                expr += e*grid[k]
            expr = np.abs(expr)
            expression &= (expr < self.M) & (expr > self.m)
            expression_big &= (expr < self.M)
            expression_removed &= (expr > self.m)
        
        for k in range(self.dim):
            if k!=i and k!=j:
                expression = np.any(expression, axis=k)
        
        points = []
        points_big = []
        points_removed = []
        for l in range(resolution):
            for m in range(resolution):
                if expression[l][m]:
                    points.append([grid[i][l][m], grid[j][l][m]])
                if expression_big[l][m]:
                    points_big.append([grid[i][l][m], grid[j][l][m]])
                if not expression_removed[l][m]:
                    points_removed.append([grid[i][l][m], grid[j][l][m]])
        points = np.array(points)
        points_big = np.array(points_big)
        points_removed = np.array(points_removed)


        fig = plt.figure(figsize=(4, 4))
        ax = fig.add_subplot(111)
        if components:
            ax.scatter(points_big[:, 0], points_big[:, 1], s=s, alpha=0.01, c='red', label='big region')
            ax.scatter(points_removed[:, 0], points_removed[:, 1], s=s, alpha=0.01, c='grey', label='removed region')
        ax.scatter(points[:, 0], points[:, 1], s=s, alpha=alpha, c='maroon', label='feasible region')
        ax.scatter(self.X[:, i], self.X[:, j], s=5, label='sample', c='orange')
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.axis('equal')
        lg = ax.legend()
        for lh in lg.legendHandles: 
            lh.set_alpha(1.)
        plt.show()
        return ax
    


class InequalitySampler3(InequalitySampler0):
    """
    A class for sampling from w.x_+ + b < l+ and w.x_- + b > l-
    """
    def __init__(self, x_minus, x_plus, l_minus, l_plus, b, signs) -> None:
        self.x_plus = x_plus
        self.x_minus = x_minus
        self.l_minus = l_minus
        self.l_plus = l_plus
        self.b = b
        self.dim = len(x_plus)
        self.signs = signs
        self.signs[self.signs == 0] = -1
    
    def is_feasible(self, x):
        """
        Description: Determines if an input belongs to the feasible region
        """
        y_plus = np.dot(self.x_plus, x)
        y_minus = np.dot(self.x_minus, x)
        return (y_plus + self.b < self.l_plus) & (y_minus + self.b > self.l_minus)

    
    
    def intersection_with_bisection(self, x0, d, tol=1e-2, max_iters=100):
        """
        Description: Finds the intersection of a line originating from x0 (lying inside the feasible region) having direction d with the feasible region
                    within the specified tolerance within the max_iters iterations
        """
        
        self.x0 = x0 - np.dot(x0, d) * d
        
        if self.is_feasible(self.x0):
            self.int_pts, self.weights = np.zeros((1, self.dim)), np.zeros(1)
            t = 1.
            while self.is_feasible(self.x0 + t*d):
                t *= 2 

            l, r = 0., t
            iter = 0

            while abs(r-l) > tol and iter < max_iters:
                m = (r+l)/2.
                if not self.is_feasible(self.x0 + m*d):
                    r = m
                else:
                    l = m
                iter += 1
            
            self.int_pts[0, :] = self.x0 + l*d
            self.weights[0] = l
            
            return 1
        else:
           self.int_pts, self.weights = [], []
           return 0
    
    
    def single_sample(self, x, steps, **kwargs):
        for _ in range(steps):
            d = np.random.normal(size=self.dim)
            d = np.abs(d) * self.signs
            d /= np.linalg.norm(d)
            # find the intersection points 
            while self.intersection_with_bisection(x, d, **kwargs) == 0:
                d = np.random.normal(size=self.dim)
                d = np.abs(d) * self.signs
                d /= np.linalg.norm(d)
            t = np.random.uniform(0., self.weights[0])
            x += t*d
        return x



class GoodRowSampler:
    """
    A class for sampling a row R such that R satisfies m<|R.x + b|<M for x in C = a convex set
    """
    def __init__(self, m, M, data):
        self.m = m 
        self.M = M 
        self.data = np.array(data)
        self.dim = self.data.shape[-1]
        self.lims = np.array([[min(self.data[:, d]), max(self.data[:, d])] for d in range(self.dim)])
        


    def get_vector(self, signs, option):
        if option:
            return np.array([self.lims[d, signs[d]] for d in range(self.dim)])
        else:
            return np.array([self.lims[d, (1+signs[d]) % 2] for d in range(self.dim)])
        

    def sample_(self):
        flag = np.random.randint(2)
        # assign signs for the entries of the row
        s = np.random.randint(2, size=self.dim)
        # set up inequalities
        if flag:
            lims = [self.m, self.M]
        else:
            lims = [-self.M, -self.m]
        x_plus = self.get_vector(s, True)
        x_minus = self.get_vector(s, False)

        b = np.random.uniform(*lims)
        sampler = InequalitySampler3(x_minus, x_plus, lims[0], lims[1], b, s)
        return sampler.single_sample(x=np.zeros(self.dim), steps=1), b
    
    # @ut.timer
    def sample(self, n_sample):
        rows, bs = np.zeros((n_sample, self.dim)), np.zeros(n_sample)
        for n in range(n_sample):
            rows[n, :], bs[n] = self.sample_()
        return rows, bs
    
    # @ut.timer
    def sample_parallel(self, n_sample):
        results = Parallel(n_jobs=-1)(delayed(self.sample_)() for _ in range(n_sample))
        rows = np.vstack([item[0] for item in results])
        bs = np.hstack([item[1] for item in results])
        return rows, bs


    def test_rows(self, rows, bs):
        Y = np.abs(self.data@(np.array(rows).T) + np.array(bs))
        return np.all((Y < self.M) & (Y > self.m), axis=0)
    
    
    def is_row(self, row, b):
        if b < self.M and b > self.m:
            sign = ((np.sign(row) + 1) / 2).astype(int)
            x_plus = self.get_vector(sign, True)
            x_minus = self.get_vector(sign, False)
            if x_plus @ row + b < self.M and x_minus @ row + b > self.m:
                return True
            else:
                return False
        elif b < -self.m and b > -self.M:
            sign = ((np.sign(row) + 1) / 2).astype(int)
            x_plus = self.get_vector(sign, True)
            x_minus = self.get_vector(sign, False)
            if x_plus @ row + b < -self.m and x_minus @ row + b > -self.M:
                return True
            else:
                return False
        else:
            return False
        
    def are_rows(self, rows, bs):
        flags = np.full(len(rows), True)
        for i in range(len(rows)):
            flags[i] = self.is_row(rows[i], bs[i])
        return flags


class BadRowSamplerLinear(GoodRowSampler):
    """
    A class for sampling a row R such that R satisfies |R.x + b| < m  for x in C = a convex set
    """
    def __init__(self, m, data):
        self.m = m 
        self.data = np.array(data)
        self.dim = self.data.shape[-1]
        self.lims = np.array([[min(self.data[:, d]), max(self.data[:, d])] for d in range(self.dim)])
        


    def sample_(self):
        # assign signs for the entries of the row
        s = np.random.randint(2, size=self.dim)
        # set up inequalities
        x_plus = self.get_vector(s, True)
        x_minus = self.get_vector(s, False)

        b = np.random.uniform(-self.m, self.m)
        sampler = InequalitySampler3(x_minus, x_plus, -self.m, self.m, b, s)
        return sampler.single_sample(x=np.zeros(self.dim), steps=1), b
    

    def test_rows(self, rows, bs):
        Y = np.abs(self.data@(np.array(rows).T) + np.array(bs))
        return np.all((Y < self.m), axis=0)
    
    def is_row(self, row, b):
        if np.abs(b) < self.m:
            sign = ((np.sign(row) + 1) / 2).astype(int)
            x_plus = self.get_vector(sign, True)
            x_minus = self.get_vector(sign, False)
            if x_plus @ row + b < self.m and x_minus @ row + b > -self.m:
                return True
            else:
                return False
        else:
            return False
        
    




class InequalitySampler4(InequalitySampler3):
    """
    A class for sampling from w.x_- + b > l+ 
    """
    def __init__(self, x_minus, l_plus, b, signs) -> None:
        self.x_minus = x_minus
        self.l_plus = l_plus
        self.b = b
        self.dim = len(x_minus)
        self.signs = signs
        self.signs[self.signs == 0] = -1
    
    
    def is_feasible(self, x):
        """
        Description: Determines if an input belongs to the feasible region
        """
        y_minus = np.dot(self.x_minus, x)
        return (y_minus + self.b > self.l_plus)

    

class InequalitySampler5(InequalitySampler3):
    """
    A class for sampling from w.x_+ + b < l-
    """
    def __init__(self, x_plus, l_minus, b, signs) -> None:
        self.x_plus = x_plus
        self.l_minus = l_minus
        self.b = b
        self.dim = len(x_plus)
        self.signs = signs
        self.signs[self.signs == 0] = -1
    
    
    def is_feasible(self, x):
        """
        Description: Determines if an input belongs to the feasible region
        """
        y_plus = np.dot(self.x_plus, x)
        return (y_plus + self.b < self.l_minus)

    

class BadRowSamplerExtreme(GoodRowSampler):
    """
    A class for sampling a row R such that R satisfies |R.x + b| > M  for x in C = a convex set
    """
    def __init__(self, M, data):
        self.M = M 
        self.data = np.array(data)
        self.dim = self.data.shape[-1]
        self.lims = np.array([[min(self.data[:, d]), max(self.data[:, d])] for d in range(self.dim)])
        


    def sample_(self):
        flag = np.random.randint(2)
        # assign signs for the entries of the row
        s = np.random.randint(2, size=self.dim)
        # set up inequalities
        x_plus = self.get_vector(s, True)
        x_minus = self.get_vector(s, False)
        
        if flag:
            b = np.random.uniform(self.M, 2.*self.M)
            sampler = InequalitySampler4(x_minus, self.M, b, s)
        else:
            b = np.random.uniform(-2.*self.M, -self.M)
            sampler = InequalitySampler5(x_plus, -self.M, b, s)
        
        return sampler.single_sample(x=np.zeros(self.dim), steps=1), b
    

    def test_rows(self, rows, bs):
        Y = np.abs(self.data@(np.array(rows).T) + np.array(bs))
        return np.all((Y > self.M), axis=0)
    
    def is_row(self, row, b):
        if b > self.M:
            sign = ((np.sign(row) + 1) / 2).astype(int)
            x_minus = self.get_vector(sign, False)
            if x_minus @ row + b > self.M:
                return True
            else:
                return False
        elif b < -self.M:
            sign = ((np.sign(row) + 1) / 2).astype(int)
            x_plus = self.get_vector(sign, True)
            if x_plus @ row + b < -self.M:
                return True
            else:
                return False
        else:
            return False
    

class MatrixSampler:
    """
    A class for selecting matrices with good/linear/extreme rows
    """
    def __init__(self, m, M, data) -> None:
        self.good = GoodRowSampler(m, M, data) 
        self.linear = BadRowSamplerLinear(m, data)
        self.extreme = BadRowSamplerExtreme(M, data)
        self.samplers = [self.good.sample, self.linear.sample, self.extreme.sample]
    
    def sample_(self, n_rows):
        good_matrix, good_b = self.good.sample(n_rows[0])
        linear_matrix, linear_b = self.linear.sample(n_rows[1])
        extreme_matrix, extreme_b = self.extreme.sample(n_rows[2])
        return np.vstack([good_matrix, linear_matrix, extreme_matrix]), np.hstack([good_b, linear_b, extreme_b])
    
    def sample_parallel_3_(self, n_rows):
        results = Parallel(n_jobs=3)(delayed(self.samplers[i])(row) for i, row in enumerate(n_rows))
        return np.vstack([item[0] for item in results]), np.hstack([item[1] for item in results])
    
    @ut.timer
    def sample(self, n_rows, n_sample):
        matrices = np.empty((n_sample, np.sum(n_rows), self.good.dim))
        bs = np.empty((n_sample, np.sum(n_rows)))
        for i in range(n_sample):
            matrices[i], bs[i] = self.sample_(n_rows) 
        return matrices, bs 
    
    @ut.timer
    def sample_parallel(self, n_rows, n_sample):
        results = Parallel(n_jobs=-1)(delayed(self.sample_)(n_rows) for _ in range(n_sample))
        matrices = np.stack([item[0] for item in results], axis=0)
        bs = np.stack([item[1] for item in results], axis=0)
        return matrices, bs

    @ut.timer
    def sample_parallel_3(self, n_rows, n_sample):
        results = Parallel(n_jobs=-1)(delayed(self.sample_parallel_3_)(n_rows) for _ in range(n_sample))
        matrices = np.stack([item[0] for item in results], axis=0)
        bs = np.stack([item[1] for item in results], axis=0)
        return matrices, bs


            
        
