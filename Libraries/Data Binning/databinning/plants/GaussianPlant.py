import numpy as np


class GaussianPlant:
    """Class that allows the gaussian probabilistic description of the system to be obtained through the technique of data binning."""

    def __init__(self, x_dim, x_discr, u_discr):

        # Initializing f(x_k| u_k, x_{k-1})
        self.conditional = np.zeros(np.concatenate((x_discr, u_discr, [x_dim], [x_dim])))

    def discretize(self, trajectory, dim, min, step):
        """Returns the discrete index associated with a continuous state or input value.

        Args:
            trajectory (numpy.ndarray[float64]): Array containing the continuous values of the states/inputs.
            dim (int): Dimension of the state space.
            min (numpy.ndarray[float64]): Array containing the minimum values of the states/inputs.
            step (numpy.ndarray[float64]): Array containing the discretization step of the states/inputs.

        Returns:
            tuple: Tuple of indices associated with states/inputs.
        """

        indices = [0]*dim
        for i in range(dim):
            indices[i] = int((trajectory[i] - min[i])//step[i])

        return(tuple(indices))

    def getConditional(self, x_discr, u_discr, x_step, u_step, x_min, u_min, sigma, sys, noise):
        """Returns Gaussian conditional PDF of the system.

        Args:
            x_discr (numpy.ndarray[float64]): Array containing the number of bins for each state.
            u_discr (numpy.ndarray[float64]): Array containing the number of bins for each input.
            x_step (numpy.ndarray[float64]): Array containing the discretization step of the states.
            u_step (numpy.ndarray[float64]): Array containing the discretization step of the inputs.
            x_min (numpy.ndarray[float64]): Array containing the minimum values of the states.
            u_min (numpy.ndarray[float64]): Array containing the minimum values of the inputs.
            sigma (numpy.ndarray[float64]): Covariance associated with each future state.
            sys (ToggleSwitchSimplified): Object of the system class to be modelled.
            noise (bool): Boolean value determining whether there is additional noise.

        Returns:
            numpy.ndarray[float64]: Conditional PDF of the system.
        """

        for i in range(x_discr[0]):
            for j in range(x_discr[1]):
                for k in range(u_discr[0]):
                    for z in range(u_discr[1]):

                        indXkm1 = np.array([i, j])
                        xkm1 = (indXkm1*x_step) + x_min

                        indUk = np.array([k, z])
                        uk = (indUk*u_step) + u_min

                        sys.initialState(xkm1)
                        xk = sys.make_step(uk)

                        if noise:
                            self.conditional[(i, j, k, z)] = np.array([xk + np.array([np.random.uniform(-2*x_step[0], 2*x_step[0]), np.random.uniform(
                                -2*x_step[1], 2*x_step[1])]), sigma + np.array([np.random.uniform(0, 2*x_step[0]), np.random.uniform(0, 2*x_step[1])])])
                        else:
                            self.conditional[(i, j, k, z)] = np.array([xk, sigma])

        return self.conditional

    def getConditionalFromTraj(self, data, x_dim, u_dim, x_step, u_step, x_max, u_max, x_min, u_min, sigma):
        """Compute the Gaussian conditional PDFs of the system using states and inputs trajectories.

        Args:
            data (numpy.ndarray[float64]): Array containing the continuous trajectories of the states/inputs.
            x_dim (numpy.ndarray[float64]): Number of states.
            u_dim (numpy.ndarray[float64]): Number of inputs
            x_step (numpy.ndarray[float64]): Array containing the discretization step of the states.
            u_step (numpy.ndarray[float64]): Array containing the discretization step of the inputs.
            x_max (numpy.ndarray[float64]): Array containing the maximum values of the states.
            u_max (numpy.ndarray[float64]): Array containing the maximum values of the inputs.
            x_min (numpy.ndarray[float64]): Array containing the minimum values of the states.
            u_min (numpy.ndarray[float64]): Array containing the minimum values of the inputs.
            sigma (numpy.ndarray[float64]): Covariance associated with each future state.

        Returns:
            numpy.ndarray[float64]: Conditional PDF of the system.
        """

        for traj in data:
            x_traj = traj[0]
            u_traj = traj[1]

            for i in range(len(x_traj) - 1):
                xkm1 = x_traj[i]   # x_{k-1}
                xk = x_traj[i+1]   # x_k
                uk = u_traj[i+1]   # u_k

                indXkm1 = self.discretize(xkm1, x_dim, x_min, x_step, x_max)
                indUk = self.discretize(uk, u_dim, u_min, u_step, u_max)

                xk_mean = self.conditional[indXkm1 + indUk][0]

                if not np.array_equal(xk_mean, np.zeros(x_dim)):
                    self.conditional[indXkm1 + indUk][0] = np.mean(np.array([xk_mean, xk]), axis=0)
                    self.conditional[indXkm1 + indUk][1] = np.std(np.array([xk_mean, xk]), axis=0)
                else:
                    self.conditional[indXkm1 + indUk] = np.array([xk, sigma])

        return self.conditional
