from .ToggleSwitchModel import *


class ToggleSwitchLinearModel(ToggleSwitchModel):
    """
    Class extending the model used by the MPC and representing a linear model.
    """

    def __init__(self, stochasticity, LacI_ref, TetR_ref, t_step, total_time, avg_period, *args) -> None:
        """Constructor of the class, set the model used by the controller and initialises the results dictionary.

        Args:
            stochasticity (bool): Boolean parameter determining whether the model is deterministic or stochastic.
            LacI_ref (float64): Reference for the LacI protein.
            TetR_ref (float64): Reference for the TetR protein.
            t_step (float64): Sampling time used by the controller.
            total_time (float64): Total simulation time used by the controller.
            avg_period (float64): Time interval for reading the average of state trajectories.
        """

        super().__init__(stochasticity, LacI_ref, TetR_ref, t_step, total_time, avg_period, *args)

    def set_model(self, stochasticity, *args):
        """Defines the model equations and the cost function. It receives the dynamic matrix A and the control matrix B of the linearised model as additional arguments.

        Args:
            stochasticity (bool, optional): Boolean parameter that adds or subtracts stochasticity to the model. Defaults to False.

        Returns:
            do_mpc.model.Model: Instance of the model used by the controller.
        """

        A = args[0]
        B = args[1]

        model = do_mpc.model.Model(model_type='continuous')

        _x = model.set_variable(var_type='states', var_name='x', shape=(4, 1))
        _u = model.set_variable(var_type='inputs', var_name='u', shape=(2, 1))

        x_next = A@_x+B@_u

        model.set_rhs('x', x_next)
        model.set_expression(expr_name='cost', expr=((model.x['x', 2] - self.LacI_ref)**2 + (model.x['x', 3] - self.TetR_ref)**2))

        model.setup()

        return model

    def set_cost(self, controller):
        """Defines the cost function that the controller uses as an objective function.

        Args:
            controller (do_mpc.controller.MPC): Controller instance for which to set the cost.

        Returns:
            do_mpc.controller.MPC: Instance of the controller.
        """
        mterm = self.model.aux['cost']
        lterm = self.model.aux['cost']

        controller.set_objective(mterm=mterm, lterm=lterm)
        controller.set_rterm(u=10)

        return controller

    def set_constraints(self, mpc):
        """Defines the constraints for the MPC controller.

        Args:
            controller (do_mpc.controller.MPC): Controller instance for which to set constraints.

        Returns:
            do_mpc.controller.MPC: Instance of the controller.
        """

        mpc.bounds['lower', '_x', 'x'] = np.array([[3.20e-2], [1.19e-1], [0], [0]])

        mpc.bounds['lower', '_u', 'u'] = np.array([[0], [0]])
        mpc.bounds['upper', '_u', 'u'] = np.array([[35], [0.35]])

        return mpc

    def set_trajectories(self, data):
        """It updates the dictionary of state trajectories based on closed-loop results obtained by the controller.

        Args:
            data (dict): Closed-loop simulation results.
        """

        states = {
            'mRNA_LacI': [x[0] for x in data['_x', 'x']],
            'mRNA_TetR': [x[1] for x in data['_x', 'x']],
            'LacI': [x[2] for x in data['_x', 'x']],
            'TetR': [x[3] for x in data['_x', 'x']]
        }

        u1 = [x[0] for x in data['_u', 'u']]
        u2 = [x[1] for x in data['_u', 'u']]

        for i in range(len(u1)):
            if i % 15 != 0:
                u1[i] = u1[i-1]
                u2[i] = u2[i-1]

        inputs = {
            'aTc': u1,
            'IPTG': u2
        }

        avg_samples_range = int(self.avg_period/self.t_step)
        avg_stop_time = self.total_time - ((self.total_time-1) % self.avg_period) - 1

        avg_LacI = [np.mean(states['LacI'][x:x + avg_samples_range]) for x in range(0, len(states['LacI']), avg_samples_range)]
        avg_TetR = [np.mean(states['TetR'][x:x + avg_samples_range]) for x in range(0, len(states['TetR']), avg_samples_range)]

        avg_x = np.arange(0, self.total_time, self.avg_period)
        favg_LacI = CubicSpline(avg_x, avg_LacI)
        avg_LacI = favg_LacI(np.arange(0, avg_stop_time, self.t_step))
        favg_TetR = CubicSpline(avg_x, avg_TetR)
        avg_TetR = favg_TetR(np.arange(0, avg_stop_time, self.t_step))

        avg_trajectories = {
            'LacI': [x for x in avg_LacI.tolist()],
            'TetR': [x for x in avg_TetR.tolist()]
        }

        ISE, ITAE = self.compute_performance_metrics(avg_trajectories)

        cost = [x[0] for x in data['_aux', 'cost'].tolist()]
        time = [x[0] for x in data['_time'].tolist()]

        self.trajectories = {
            'states': states,
            'inputs': inputs,
            'avg_traj': avg_trajectories,
            'cost': cost,
            'time': time,
            'ISE': ISE,
            'ITAE': ITAE
        }

    def plot_results(self):
        """Plot the results of the closed-loop simulation.

        Returns:
            numpy.ndarray, numpy.ndarray: Arrays of lines and figures.
        """

        fig_x = 20
        fig_y = 10

        plt.rcParams['axes.grid'] = True
        plt.rcParams['font.size'] = 16

        time = self.trajectories['time']
        avg_stop_time = self.total_time - ((self.total_time-1) % self.avg_period) - 1
        avg_time = np.arange(0, avg_stop_time, self.t_step)

        # -------------------- Proteins -------------------- #
        # --- LacI --- #
        figure_proteins, axes = plt.subplots(2, sharex=True, figsize=(fig_x, fig_y))

        axes[0].set_ylabel(r'$a.u.$')
        axes[0].set_title('LacI')
        line_LacI, = axes[0].plot(time, self.trajectories['states']['LacI'], color='b')
        line_avg_LacI, = axes[0].plot(avg_time, self.trajectories['avg_traj']['LacI'], color='b', linestyle='--')
        line_ref_LacI, = axes[0].plot(time, self.LacI_ref*np.ones(len(time)), color='k', linestyle='--')
        axes[0].legend(['LacI', 'Avg LacI traj', 'LacI target'], loc='upper right')

        # --- TetR --- #
        axes[1].set_ylabel(r'$a.u.$')
        axes[1].set_title('TetR')
        line_TetR, = axes[1].plot(time, self.trajectories['states']['TetR'], color='m')
        line_avg_TetR, = axes[1].plot(avg_time, self.trajectories['avg_traj']['TetR'], color='m', linestyle='--')
        line_ref_TetR, = axes[1].plot(time, self.TetR_ref*np.ones(len(time)), color='k', linestyle='--')
        axes[1].legend(['TetR', 'Avg TetR traj', 'TetR target'], loc='upper right')
        axes[1].set_xlabel('Time [min]')

        figure_proteins.set_facecolor("white")

        # -------------------- mRNAs -------------------- #
        # --- mRNA LacI --- #
        figure_mRNAs, axes1 = plt.subplots(2, sharex=True, figsize=(fig_x, fig_y))
        axes1[0].set_ylabel(r'$mRNA$')
        axes1[0].set_title('mRNA LacI')
        line_mRNA_LacI, = axes1[0].plot(time, self.trajectories['states']['mRNA_LacI'], color='b')
        axes1[0].legend(['mRNA LacI'], loc='upper right')

        # --- mRNA TetR --- #
        axes1[1].set_ylabel(r'$mRNA$')
        axes1[1].set_title('mRNA TetR')
        line_mRNA_TetR, = axes1[1].plot(time, self.trajectories['states']['mRNA_TetR'], color='m')
        axes1[1].legend(['mRNA TetR'], loc='upper right')
        axes1[1].set_xlabel('Time [min]')

        figure_mRNAs.set_facecolor("white")

        # -------------------- External inductors concentrations -------------------- #
        # --- aTc --- #
        figure_inductors, axes2 = plt.subplots(2, sharex=True, figsize=(fig_x, fig_y))
        axes2[0].set_ylabel(r'$ng/mL$')
        axes2[0].set_title('aTc')
        line_aTc, = axes2[0].plot(time, self.trajectories['inputs']['aTc'], color='y')
        axes2[0].legend(['aTc'], loc='upper right')

        # --- IPTG --- #
        axes2[1].set_ylabel(r'$mM$')
        axes2[1].set_title('IPTG')
        line_IPTG, = axes2[1].plot(time, self.trajectories['inputs']['IPTG'], color='r')
        axes2[1].legend(['IPTG'], loc='upper right')
        axes2[1].set_xlabel('Time [min]')

        figure_inductors.set_facecolor("white")
        # -------------------- Cost -------------------- #
        figure_cost, axes3 = plt.subplots(1, figsize=(fig_x, fig_y/2))
        axes3.set_title('Cost')
        line_cost, = axes3.plot(time, self.trajectories['cost'], color='g')
        axes3.legend(['Cost'], loc='upper right')
        axes3.set_xlabel('Time [min]')

        figure_cost.set_facecolor("white")

        figures = np.array([figure_proteins, figure_mRNAs, figure_inductors, figure_cost])
        lines = np.array([line_LacI, line_avg_LacI, line_ref_LacI, line_TetR, line_avg_TetR, line_ref_TetR,
                          line_mRNA_LacI, line_mRNA_TetR, line_aTc, line_IPTG, line_cost])

        return figures, lines
