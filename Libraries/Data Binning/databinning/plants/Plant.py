import numpy as np


class Plant:
    """Class that allows the generic probabilistic description of the system to be obtained through the technique of data binning."""

    def __init__(self, x_discr, u_discr):

        # Initializing f(x_k, u_k, x_{k-1})
        self.full_joint = np.zeros(np.concatenate((x_discr, u_discr, x_discr)))

        # Initializing f(u_k, x_{k-1})
        self.reduced_joint = np.zeros(np.concatenate((u_discr, x_discr)))

        # Initializing f(x_k| u_k, x_{k-1})
        self.conditional = np.zeros(np.concatenate((self.x_discr, self.u_discr, self.x_discr)))

    def discretize(self, trajectory, dim, min, step):
        """Returns the discrete index associated with a continuous state or input value.

        Args:
            trajectory (numpy.ndarray[float64]): Array containing the continuous values of the states/inputs.
            dim (int): Number of states/inputs.
            min (numpy.ndarray[float64]): Array containing the minimum values of the states/inputs.
            step (numpy.ndarray[float64]): Array containing the discretization step of the states/inputs.

        Returns:
            tuple: Tuple of indices associated with states/inputs.
        """

        indices = [0]*dim

        for i in range(dim):
            indices[i] = int((trajectory[i] - min[i])//step[i])

        return(tuple(indices))

    def getJoints(self, data, x_dim, u_dim, x_step, u_step, x_max, u_max, x_min, u_min):
        """Compute the full joint and the reduced joint PDFs of the system using states and inputs trajectories.

        Args:
            data (numpy.ndarray[float64]): Array containing the continuous trajectories of the states/inputs.
            x_dim (numpy.ndarray[float64]): Number of states.
            u_dim (numpy.ndarray[float64]): Number of inputs.
            x_step (numpy.ndarray[float64]): Array containing the discretization step of the states.
            u_step (numpy.ndarray[float64]): Array containing the discretization step of the inputs.
            x_max (numpy.ndarray[float64]): Array containing the maximum values of the states.
            u_max (numpy.ndarray[float64]): Array containing the maximum values of the inputs.
            x_min (numpy.ndarray[float64]): Array containing the minimum values of the states.
            u_min (numpy.ndarray[float64]): Array containing the minimum values of the inputs.

        Returns:
            numpy.ndarray[float64], numpy.ndarray[float64]: Full joint and reduced joint.
        """

        for traj in data:
            x_traj = traj[0]
            u_traj = traj[1]

            for i in range(len(x_traj) - 1):
                xkm1 = x_traj[i]   # x_{k-1}
                xk = x_traj[i+1]   # x_k
                uk = u_traj[i+1]   # u_k

                indXkm1 = self.discretize(xkm1, x_dim, x_min, x_step, x_max)
                indXk = self.discretize(xk, x_dim, x_min, x_step, x_max)
                indUk = self.discretize(uk, u_dim, u_min, u_step, u_max)

                # Updating the full joint 'pmf'
                self.full_joint[indXk + indUk + indXkm1] += 1

                # Updating the partial 'pmf'
                self.reduced_joint[indUk + indXkm1] += 1

        self.full_joint = self.full_joint / np.sum(self.full_joint)
        self.reduced_joint = self.reduced_joint / np.sum(self.reduced_joint)

        return self.full_joint, self.reduced_joint

    def getConditional(self):
        """Compute the conditional PDF of the system starting from the full joint and the reduced joint.

        Returns:
            numpy.ndarray[float64]: Conditional PDF of the system.
        """

        for (index, x) in np.ndenumerate(self.full_joint):
            u_index = index[self.x_dim:]

            if self.reduced_joint[u_index] == 0:
                self.conditional[index] = 0
            else:
                self.conditional[index] = self.full_joint[index] / self.reduced_joint[u_index]

        return self.conditional
