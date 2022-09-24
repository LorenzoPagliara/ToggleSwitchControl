import do_mpc
from casadi import *

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.interpolate import CubicSpline
import numpy as np
import json


class ToggleSwitchModel:
    """
    Class representing the Toggle Switch model used by the MPC controller. 
    This defines the model equations, the cost function and constraints to be taken into account when applying the control strategy, 
    and a set of methods for formatting, storing, exporting and plotting closed-loop simulation results and performance evaluation.
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

        self.t_step = t_step
        self.total_time = total_time
        self.avg_period = avg_period
        self.LacI_ref = LacI_ref
        self.TetR_ref = TetR_ref

        self.model = self.set_model(stochasticity, *args)
        self.trajectories = {}

    def set_model(self, stochasticity=False, *args):
        """Defines the model equations and the cost function.

        Args:
            stochasticity (bool, optional): Boolean parameter that adds or subtracts stochasticity to the model. Defaults to False.

        Returns:
            do_mpc.model.Model: Instance of the model used by the controller.
        """

        model = do_mpc.model.Model(model_type='continuous')

        # Model states
        mRNA_LacI = model.set_variable(var_type='states', var_name='mRNA_LacI')
        mRNA_TetR = model.set_variable(var_type='states', var_name='mRNA_TetR')
        lacI = model.set_variable(var_type='states', var_name='LacI')
        tetR = model.set_variable(var_type='states', var_name='TetR')
        v1 = model.set_variable(var_type='states', var_name='v1')
        v2 = model.set_variable(var_type='states', var_name='v2')

        # Model inputs
        aTc = model.set_variable(var_type='inputs', var_name='aTc')
        iptg = model.set_variable(var_type='inputs', var_name='IPTG')

        # Model parameters
        k_m0_L = 3.20e-2
        k_m0_T = 1.19e-1
        k_m_L = 8.30
        k_m_T = 2.06
        k_p_L = 9.726e-1
        k_p_T = 9.726e-1
        g_m_L = 1.386e-1
        g_m_T = 1.386e-1
        g_p_L = 1.65e-2
        g_p_T = 1.65e-2
        theta_LacI = 31.94
        theta_TetR = 30.00
        theta_IPTG = 9.06e-2
        theta_aTc = 11.65
        eta_LacI = 2.00
        eta_TetR = 2.00
        eta_IPTG = 2.00
        eta_aTc = 2.00
        k_in_aTc = 2.75e-2
        k_out_aTc = 2.00e-2
        k_in_IPTG = 1.62e-1
        k_out_IPTG = 1.11e-1

        # Defining model's equations
        dmRNA_LacI = k_m0_L + k_m_L*(1 / (1 + ((tetR/theta_TetR) * (1 / (1 + (v1/theta_aTc)**eta_aTc)))**eta_TetR)) - g_m_L * mRNA_LacI
        dmRNA_TetR = k_m0_T + k_m_T*(1 / (1 + ((lacI/theta_LacI) * (1 / (1 + (v2/theta_IPTG)**eta_IPTG)))**eta_LacI)) - g_m_T * mRNA_TetR
        dLacI = k_p_L * mRNA_LacI - g_p_L * lacI
        dTetR = k_p_T * mRNA_TetR - g_p_T * tetR
        dv1 = (k_in_aTc * (aTc - v1)) * (aTc > v1) + (k_out_aTc * (aTc - v1)) * (aTc <= v1)
        dv2 = (k_in_IPTG * (iptg - v2)) * (iptg > v2) + (k_out_IPTG * (iptg - v2)) * (iptg <= v2)

        model.set_rhs('mRNA_LacI', dmRNA_LacI, process_noise=stochasticity)
        model.set_rhs('mRNA_TetR', dmRNA_TetR, process_noise=stochasticity)
        model.set_rhs('LacI', dLacI, process_noise=stochasticity)
        model.set_rhs('TetR', dTetR, process_noise=stochasticity)
        model.set_rhs('v1', dv1, process_noise=stochasticity)
        model.set_rhs('v2', dv2, process_noise=stochasticity)

        # The process noise w is used to simulate a disturbed system in the Simulator.

        # Measurement noise
        if stochasticity:
            model.n_v = np.random.randn(6, 1)

        # Cost function
        model.set_expression(expr_name='cost', expr=((lacI - self.LacI_ref)**2 + (tetR - self.TetR_ref)**2))

        model.setup()

        return model

    def get_model(self):
        """Returns the model instance used by the controller.

        Returns:
            do_mpc.model.Model: Instance of the model used by the controller.
        """
        return self.model

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
        controller.set_rterm(aTc=1, IPTG=1)

        return controller

    def set_constraints(self, controller):
        """Defines the constraints for the MPC controller.

        Args:
            controller (do_mpc.controller.MPC): Controller instance for which to set constraints.

        Returns:
            do_mpc.controller.MPC: Instance of the controller.
        """

        controller.bounds['lower', '_x', 'mRNA_LacI'] = 3.20e-2
        controller.bounds['lower', '_x', 'mRNA_TetR'] = 1.19e-1

        controller.bounds['lower', '_x', 'LacI'] = 0
        controller.bounds['lower', '_x', 'TetR'] = 0

        controller.bounds['lower', '_x', 'v1'] = 0
        controller.bounds['lower', '_x', 'v2'] = 0

        controller.bounds['lower', '_u', 'aTc'] = 0
        controller.bounds['upper', '_u', 'aTc'] = 35

        controller.bounds['lower', '_u', 'IPTG'] = 0
        controller.bounds['upper', '_u', 'IPTG'] = 0.35

        return controller

    def compute_performance_metrics(self, avg_trajectories):
        """Calculates the Integral Square Error (ISE) and the Integral Time-weighted Absolute Error (ITAE).

        Args:
            avg_trajectories (numpy.ndarray[float64]): Array of the average protein trajectory.

        Returns:
            float64, float64: Integral Square Error (ISE), Integral Time-weighted Absolute Error (ITAE).
        """

        e3_bar = np.array([(x - self.LacI_ref) / self.LacI_ref for x in avg_trajectories['LacI']])
        e4_bar = np.array([(x - self.TetR_ref) / self.TetR_ref for x in avg_trajectories['TetR']])

        e_bar = np.array([np.linalg.norm([e3_bar[i], e4_bar[i]]) for i in range(len(e3_bar))])

        ISE = np.sum((e_bar)**2)
        ITAE = np.sum([np.abs(e_bar[i])*(self.t_step*(i+1)) for i in range(len(e_bar))])

        return ISE, ITAE

    def set_trajectories(self, data):
        """It updates the dictionary of state trajectories based on closed-loop results obtained by the controller.

        Args:
            data (dict): Closed-loop simulation results.
        """

        states = {
            'mRNA_LacI': [x[0] for x in data['_x', 'mRNA_LacI'].tolist()],
            'mRNA_TetR': [x[0] for x in data['_x', 'mRNA_TetR'].tolist()],
            'LacI': [x[0] for x in data['_x', 'LacI'].tolist()],
            'TetR': [x[0] for x in data['_x', 'TetR'].tolist()],
            'v1': [x[0] for x in data['_x', 'v1'].tolist()],
            'v2': [x[0] for x in data['_x', 'v2'].tolist()],
        }

        u1 = [x[0] for x in data['_u', 'aTc'].tolist()]
        u2 = [x[0] for x in data['_u', 'IPTG'].tolist()]

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

    def export_results(self, type, name, mode):
        """Exports closed-loop simulation results.

        Args:
            type (str): Defines the type of simulation performed: deterministic, stochastic, randomic, uncertain.
            name (str): Defines the name of the file on which the results will be stored.
            mode (str): Defines the mode of opening the file.
        """

        f = open('./data/' + type + '/' + name + '.json', mode)
        data_json = json.dumps(self.trajectories)
        f.write(data_json)
        f.close()

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

        axes[0].set_ylabel('a.u.')
        axes[0].set_title('LacI')
        line_LacI, = axes[0].plot(time, self.trajectories['states']['LacI'], color='b')
        line_avg_LacI, = axes[0].plot(avg_time, self.trajectories['avg_traj']['LacI'], color='b', linestyle='--')
        line_ref_LacI, = axes[0].plot(time, self.LacI_ref*np.ones(len(time)), color='k', linestyle='--')
        axes[0].legend(['LacI', 'Avg LacI traj', 'LacI target'], loc='upper right')

        # --- TetR --- #
        axes[1].set_ylabel('a.u.')
        axes[1].set_title('TetR')
        line_TetR, = axes[1].plot(time, self.trajectories['states']['TetR'], color='m')
        line_avg_TetR, = axes[1].plot(avg_time, self.trajectories['avg_traj']['TetR'], color='m', linestyle='--')
        line_ref_TetR, = axes[1].plot(time, self.TetR_ref*np.ones(len(time)), color='k', linestyle='--')
        axes[1].legend(['TetR', 'Avg TetR traj', 'TetR target'], loc='upper right')
        axes[1].set_xlabel('time [min]')

        figure_proteins.set_facecolor("white")

        # -------------------- mRNAs -------------------- #
        # --- mRNA LacI --- #
        figure_mRNAs, axes1 = plt.subplots(2, sharex=True, figsize=(fig_x, fig_y))
        axes1[0].set_ylabel('')
        axes1[0].set_title('mRNA_LacI')
        line_mRNA_LacI, = axes1[0].plot(time, self.trajectories['states']['mRNA_LacI'], color='b')
        axes1[0].legend(['mRNA LacI'], loc='upper right')

        # --- mRNA TetR --- #
        axes1[1].set_ylabel('')
        axes1[1].set_title('mRNA_TetR')
        line_mRNA_TetR, = axes1[1].plot(time, self.trajectories['states']['mRNA_TetR'], color='m')
        axes1[1].legend(['mRNA TetR'], loc='upper right')
        axes1[1].set_xlabel('time [min]')

        figure_mRNAs.set_facecolor("white")

        # -------------------- Internal inducers concentrations -------------------- #
        # --- v1 --- #
        figure_int_inducers, axes2 = plt.subplots(2, sharex=True, figsize=(fig_x, fig_y))
        axes2[0].set_ylabel('a.u.')
        axes2[0].set_title('$v1$')
        line_int_aTc, = axes2[0].plot(time, self.trajectories['states']['v1'], color='y')
        axes2[0].legend(['$v1$'], loc='upper right')

        # --- v2 --- #
        axes2[1].set_ylabel('')
        axes2[1].set_title('$v2$')
        line_int_IPTG, = axes2[1].plot(time, self.trajectories['states']['v2'], color='r')
        axes2[1].legend(['$v2$'], loc='upper right')
        axes2[1].set_xlabel('time [min]')

        figure_int_inducers.set_facecolor("white")

        # -------------------- External inducers concentrations -------------------- #
        # --- aTc --- #
        figure_inducers, axes3 = plt.subplots(2, sharex=True, figsize=(fig_x, fig_y))
        axes3[0].set_ylabel('')
        axes3[0].set_title('aTc')
        line_aTc, = axes3[0].plot(time, self.trajectories['inputs']['aTc'], color='y')
        axes3[0].legend(['aTc'], loc='upper right')

        # --- IPTG --- #
        axes3[1].set_ylabel('')
        axes3[1].set_title('IPTG')
        line_IPTG, = axes3[1].plot(time, self.trajectories['inputs']['IPTG'], color='r')
        axes3[1].legend(['IPTG'], loc='upper right')
        axes3[1].set_xlabel('time [min]')

        figure_inducers.set_facecolor("white")

        # -------------------- Cost -------------------- #
        figure_cost, axes4 = plt.subplots(1, figsize=(fig_x, fig_y/2))
        axes4.set_ylabel('')
        axes4.set_title('Cost')
        line_cost, = axes4.plot(time, self.trajectories['cost'], color='g')
        axes4.legend(['Cost'], loc='upper right')
        axes4.set_xlabel('time [min]')

        figure_cost.set_facecolor("white")

        figures = np.array([figure_proteins, figure_mRNAs, figure_int_inducers, figure_inducers, figure_cost])
        lines = np.array([line_LacI, line_avg_LacI, line_ref_LacI, line_TetR, line_avg_TetR, line_ref_TetR,
                          line_mRNA_LacI, line_mRNA_TetR, line_int_aTc, line_int_IPTG, line_aTc, line_IPTG, line_cost])

        return figures, lines

    def update_mRNA(self, i, data, lines):
        lines[0].set_data(data['time'][:i], data['states']['mRNA_LacI'][:i])
        lines[1].set_data(data['time'][:i], data['states']['mRNA_TetR'][:i])
        return lines

    def update_protein(self, i, data, lines, avg_time):
        lines[0].set_data(data['time'][:i], data['states']['LacI'][:i])
        lines[1].set_data(avg_time[:i], data['avg_traj']['LacI'][:i])
        lines[2].set_data(data['time'][:i], self.LacI_ref*np.ones(len(data['time']))[:i])
        lines[3].set_data(data['time'][:i], data['states']['TetR'][:i])
        lines[4].set_data(avg_time[:i], data['avg_traj']['TetR'][:i])
        lines[5].set_data(data['time'][:i], self.TetR_ref*np.ones(len(data['time']))[:i])
        return lines

    def update_internal_inducers(self, i, data, lines):
        lines[0].set_data(data['time'][:i], data['states']['v1'][:i])
        lines[1].set_data(data['time'][:i], data['states']['v2'][:i])
        return lines

    def update_external_inducers(self, i, data, lines):
        lines[0].set_data(data['time'][:i], data['inputs']['aTc'][:i])
        lines[1].set_data(data['time'][:i], data['inputs']['IPTG'][:i])
        return lines

    def update_cost(self, i, data, lines):
        lines[0].set_data(data['time'][:i], data['cost'][:i])
        return lines

    def animate_results(self, type, name, funct, figures, fargs, steps):
        """Animate result plots.

        Args:
            type (str): Defines the type of simulation performed: deterministic, stochastic, randomic, uncertain.
            name (str): Defines the name of the file on which the plots will be stored.
            funct (funct): Defines the function for updating plots.
            figures (numpy.ndarray): Array of figures to be animated.
            fargs (): Arguments of the plots update functions.
            steps (int): Number of control loop steps.
        """
        anim = animation.FuncAnimation(figures, funct, fargs=fargs, frames=steps)
        anim.save('./simulations/' + type + '/' + name + '.mp4', fps=60)
