import numpy as np
from numpy.linalg import inv
from scipy.stats import multivariate_normal
import math


class ControllerFPD:
    """Class implementing the control from demonstartion algorithm"""

    def __init__(self, f_x, g_x, sys) -> None:
        """Initialize probabilistic descriptions of the system to be controlled and of the target system and the instance of the deterministic model.

        Args:
            f_x (numpy.ndarray[float64]): Probabilistic description of the system to be controlled.
            g_x (numpy.ndarray[float64]): Probabilistic description of the target system.
            sys (ToggleSwitchSimplified): Instance of the deterministic model.
        """

        self.f_x = f_x
        self.g_x = g_x
        self.sys = sys

    def dkl(self, f_x, g_x):
        """Compute analytically the Kullback-Leibler divergence.

        Args:
            f_x (numpy.ndarray[float64]): Probabilistic description of the system to be controlled.
            g_x (numpy.ndarray[float64]): Probabilistic description of the target system.

        Returns:
            float64: Value of the Kullback-Leibler divergence.
        """

        arr = np.zeros(f_x.shape[0] * f_x.shape[1])

        for i in range(f_x.shape[0]):
            for j in range(f_x.shape[1]):
                if f_x[(i, j)] != 0 and g_x[(i, j)] != 0:
                    arr[(i*f_x.shape[1]) + j] = f_x[(i, j)] * \
                        math.log(f_x[(i, j)]/g_x[(i, j)])

        return np.sum(arr)

    def gaussianDKL(self, mu1, mu2, cov1, cov2, dim):
        """Compute numerically the Kullback-Leibler divergence.

        Args:
            mu1 (float64): Mean of the next state of the system to be controlled.
            mu2 (float64): Mean of the next state of the target system.
            cov1 (numpy.ndarray[float64]): Covariance of the next state of the system to be controlled.
            cov2 (numpy.ndarray[float64]): Covariance of the next state of the target system.
            dim (int): Number of states.

        Returns:
            float64: Value of the Kullback-Leibler divergence.
        """

        el1 = np.trace(np.dot(inv(cov2), cov1))
        el2 = np.dot(np.dot(np.transpose(np.subtract(mu2, mu1)), inv(cov2)), (np.subtract(mu2, mu1)))
        el3 = np.log(((np.linalg.det(cov2))/(np.linalg.det(cov1)))) if np.linalg.det(cov1) != 0 else 0

        return 0.5*(el1 + el2 - dim + el3)

    def discretize(self, elm, dim, min, step, max):
        """Returns the discrete index associated with a continuous state or input value.

        Args:
            elm (numpy.ndarray[float64]): Array containing the continuous values of the states/inputs.
            dim (int): Number of states/inputs.
            min (numpy.ndarray[float64]): Array containing the minimum values of the states/inputs.
            step (numpy.ndarray[float64]): Array containing the discretization step of the states/inputs.
            max (numpy.ndarray[float64]): Array containing the maximum values of the states/inputs.

        Returns:
            tuple: Tuple of indices associated with states/inputs.
        """

        indices = [0]*dim
        for i in range(dim):
            value = max[i] - step[i] if elm[i] >= max[i] else elm[i]
            indices[i] = int((value - min[i])//step[i])

        return(tuple(indices))

    def makeFPDStep(self, uk, u_step, u_discr, u_axis, x_dim, x_min, x_step, x_max):
        """It calculates the optimal policy and samples the control input to be applied to the system.

        Args:
            uk (numpy.ndarray[float64]): Current target input
            u_step (numpy.ndarray[float64]): Array containing the discretization step of the inputs.
            u_discr (numpy.ndarray[float64]): Array containing the number of bins for each input.
            u_axis (numpy.ndarray[float64]): X-axis of possible state values. 
            x_dim (int): Number of states.
            x_min (numpy.ndarray[float64]): Array containing the minimum values of the states.
            x_step (numpy.ndarray[float64]): Array containing the discretization step of the states.
            x_max (numpy.ndarray[float64]): Array containing the maximum values of the states.

        Returns:
            numpy.ndarray[float64]: Input sampled by the optimal policy.
        """

        g_pf = multivariate_normal(uk, np.array([[u_step[0], 0], [0, 0.1*u_step[1]]])).pdf(u_axis)
        s = np.sum(g_pf)
        g_u = np.array([x/s for x in g_pf])

        f_u = np.zeros((g_u.shape))

        x_km1 = self.discretize(self.sys.getState(), x_dim, x_min, x_step, x_max)

        for i in range(u_discr[0]):
            for j in range(u_discr[1]):

                f_x = self.f_x[(x_km1[0], x_km1[1], i, j)]
                g_x = self.g_x[(x_km1[0], x_km1[1], i, j)]

                f_x_cov = np.array([[f_x[1][0], 0], [0, f_x[1][1]]])
                g_x_cov = np.array([[g_x[1][0], 0], [0, g_x[1][1]]])

                f_u[(j, i)] = g_u[(j, i)]*np.exp(-self.gaussianDKL(f_x[0], g_x[0], f_x_cov, g_x_cov, x_dim))

        s = np.sum(f_u)
        if s != 0:
            f_u = np.array([x/s for x in f_u])

        ind = np.random.choice(range(u_discr[0]*u_discr[1]), p=f_u.T.reshape(1, -1)[0])

        u2_ind = int(ind//u_discr[1])
        u1_ind = ind % u_discr[1]

        u = np.array([[u_axis[u1_ind, u2_ind][0]], [u_axis[u1_ind, u2_ind][1]]])

        return u
