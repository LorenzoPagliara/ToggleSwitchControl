import numpy as np


class Plant:

    def __init__(self, x_discr, u_discr):

        # Initializing f(x_k, u_k, x_{k-1})
        self.full_joint = np.zeros(np.concatenate((x_discr, u_discr, x_discr)))

        # Initializing f(u_k, x_{k-1})
        self.reduced_joint = np.zeros(np.concatenate((u_discr, x_discr)))

        # Initializing f(x_k| u_k, x_{k-1})
        self.conditional = np.zeros(np.concatenate((self.x_discr, self.u_discr, self.x_discr)))

    def discretize(self, trajectory, dim, min, step, max):

        indices = [0]*dim

        for i in range(dim):
            value = trajectory[i] - step[i] if trajectory[i] >= max[i] else trajectory[i]
            indices[i] = int((value - min[i])//step[i])

        return(tuple(indices))

    def getJoints(self, data, x_dim, u_dim, x_step, u_step, x_max, u_max, x_min, u_min):

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

        for (index, x) in np.ndenumerate(self.full_joint):
            u_index = index[self.x_dim:]

            if self.reduced_joint[u_index] == 0:
                self.conditional[index] = 0
            else:
                self.conditional[index] = self.full_joint[index] / self.reduced_joint[u_index]

        return self.conditional


class GaussianPlant:

    def __init__(self, x_dim, x_discr, u_discr):

        # Initializing f(x_k| u_k, x_{k-1})
        self.conditional = np.zeros(np.concatenate((x_discr, u_discr, [x_dim], [x_dim])))

    def discretize(self, trajectory, dim, min, step, max):

        indices = [0]*dim
        for i in range(dim):
            value = max[i] - step[i] if trajectory[i] >= max[i] else trajectory[i]
            indices[i] = int((value - min[i])//step[i])

        return(tuple(indices))

    def getPlant(self, data, x_dim, u_dim, x_step, u_step, x_max, u_max, x_min, u_min, sigma):

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

    def getIdealPlant(self, x_discr, u_discr, x_step, u_step, x_min, u_min, sigma, sys):

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

                        self.conditional[(i, j, k, z)] = np.array([xk, sigma])

        return self.conditional
