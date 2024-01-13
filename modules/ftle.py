import numpy as np
from scipy.integrate import solve_ivp, odeint
import utility as ut
import scipy

class L63FTLE:
    def __init__(self, alpha=10., beta=8./3., rho=28.):
        self.alpha = alpha
        self.beta  = beta
        self.rho = rho
        self.dim = 3

    def func(self, X):
        x, y, z = X
        return [self.alpha * (y - x), (self.rho - z) * x - y, x * y - self.beta * z]

    # Jacobian matrix
    def Jacobian(self, X):
        x, y, z = X
        return np.array([[-self.alpha, self.alpha, 0], [self.rho - z, -1, -x], [y, x, -self.beta]])

    def assemble(self, X, Phi):
        return np.hstack((X,Phi.flatten()))

    def disassemble(self, XPhi):
        return XPhi[:self.dim], XPhi[self.dim:].reshape(self.dim, self.dim)

    def augmented_func(self, t, XPhi):
        X, Phi = self.disassemble(XPhi)
        dX = self.func(X)
        dPhi = self.Jacobian(X) @ Phi
        return self.assemble(dX, dPhi)
    

    # @ut.timer
    def compute(self, X0, T, n_iters=100, n_t_span=100):
        # initial states:
        Phi0 = np.identity(self.dim)
        dT = T/n_iters
        t_span = np.linspace(0., dT, num=n_t_span)
        t = dT
        norms = []
        for _ in range(n_iters):
            # sol = solve_ivp(
            #         fun=self.augmented_func,
            #         t_span=[0, dT],
            #         y0=self.assemble(X0, Phi0),
            #         t_eval=[dT],
            #         method='RK45'
            #     )
            # X0, PhiT = self.disassemble(np.array(sol.y).flatten())
            sol = odeint(func=self.augmented_func,
                         t=t_span,
                         y0=self.assemble(X0, Phi0),
                         tfirst=True
                        )
            X0, PhiT = self.disassemble(sol[-1].flatten())
            Phi0, R = scipy.linalg.qr(PhiT)
            # print(R.diagonal(), '------R______')
            # print(X0)
            # print(Phi0.T@Phi0, Phi0@Phi0.T)
            norms.append(np.log(np.abs(R.diagonal()))/dT)
            # t += dT
        
        return np.max(np.average(norms, axis=0))
        # return np.log(np.linalg.norm(Phi @ X0)) / T
       
