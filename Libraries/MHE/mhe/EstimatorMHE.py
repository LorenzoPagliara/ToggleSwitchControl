import numpy as np
import matplotlib.pyplot as plt
import do_mpc
from casadi import *


class EstimatorMHE:
    """
    Class representing the MHE estimator. 
    It contains the model instance whose parameters are to be estimated.

    This class defines the parameters of the estimator and is capable of executing the estimation loop.
    """

    def __init__(self, t_step, setup_mhe) -> None:
        """Constructor of the class, sets the parameters of the estimator and simulator.

        Args:
            t_step (float64): Estimator sampling time.
            setup_mhe (dict): Dictionary containing estimator parameters.
        """
        self.t_step = t_step
        self.model = self.set_model()
        self.estimator = self.estimator_mhe(setup_mhe)
        self.simulator = self.simulator_mhe()

    def set_model(self):
        """Defines the model equations and the parameters to be estimated.

        Returns:
            do_mpc.model.Model: Instance of the model used by the estimator.
        """

        model = do_mpc.model.Model(model_type='continuous')

        # Model states
        x = model.set_variable(var_type='states', var_name='x', shape=(4, 1))
        dx = model.set_variable(var_type='states', var_name='dx', shape=(4, 1))

        # Model inputs
        u = model.set_variable(var_type='inputs', var_name='u', shape=(2, 1))

        # State measurements
        x_meas = model.set_meas('x_meas', x, meas_noise=True)

        # Input measurements
        u_meas = model.set_meas('u_meas', u, meas_noise=False)

        # Model parameters
        k_m0_L = model.set_variable('parameter', 'k_m0_L')
        k_m0_T = model.set_variable('parameter', 'k_m0_T')
        k_m_L = model.set_variable('parameter', 'k_m_L')
        k_m_T = model.set_variable('parameter', 'k_m_T')
        k_p_L = model.set_variable('parameter', 'k_p_L')
        k_p_T = model.set_variable('parameter', 'k_p_T')
        g_m_L = model.set_variable('parameter', 'g_m_L')
        g_m_T = model.set_variable('parameter', 'g_m_T')
        g_p_L = model.set_variable('parameter', 'g_p_L')
        g_p_T = model.set_variable('parameter', 'g_p_T')
        theta_LacI = model.set_variable('parameter', 'theta_LacI')
        theta_TetR = model.set_variable('parameter', 'theta_TetR')
        theta_IPTG = model.set_variable('parameter', 'theta_IPTG')
        theta_aTc = model.set_variable('parameter', 'theta_aTc')
        eta_LacI = 2.00
        eta_TetR = 2.00
        eta_IPTG = 2.00
        eta_aTc = 2.00

        model.set_rhs('x', dx)

        dx_next = vertcat(
            k_m0_L + k_m_L*(1 / (1 + ((x[3]/theta_TetR) * (1 / (1 + (u[0]/theta_aTc)**eta_aTc)))**eta_TetR)) - g_m_L * x[0],
            k_m0_T + k_m_T*(1 / (1 + ((x[2]/theta_LacI) * (1 / (1 + (u[1]/theta_IPTG)**eta_IPTG)))**eta_LacI)) - g_m_T * x[1],
            k_p_L * x[0] - g_p_L * x[2],
            k_p_T * x[1] - g_p_T * x[3]
        )

        model.set_rhs('dx', dx_next, process_noise=False)

        model.setup()

        return model

    def estimator_mhe(self, setup_mhe):
        """Defines the parameters of the estimator and the bounds of the parameters to be estimated.

        Args:
            setup_mhe (dict): Dictionary containing estimator parameters.

        Returns:
            do_mpc.estimator.MHE: Instance of the estimator.
        """

        mhe = do_mpc.estimator.MHE(self.model, ['k_m0_L', 'k_m0_T', 'k_m_L', 'k_m_T', 'k_p_L', 'k_p_T', 'g_m_L', 'g_m_T', 'g_p_L', 'g_p_T', 'theta_LacI',
                                                'theta_TetR', 'theta_IPTG', 'theta_aTc'])

        mhe.set_param(**setup_mhe)

        Px = 50*np.eye(8)
        Pv = 10*np.diag(np.array([1, 1, 1, 1]))
        Pp = 1000*np.eye(14)
        mhe.set_default_objective(Px, Pv, Pp)

        mhe.bounds['lower', '_p_est', 'k_m0_L'] = 0
        mhe.bounds['lower', '_p_est', 'k_m0_T'] = 0
        mhe.bounds['lower', '_p_est', 'k_m_L'] = 0
        mhe.bounds['lower', '_p_est', 'k_m_T'] = 0
        mhe.bounds['lower', '_p_est', 'k_p_L'] = 0
        mhe.bounds['lower', '_p_est', 'k_p_T'] = 0
        mhe.bounds['lower', '_p_est', 'g_m_L'] = 0
        mhe.bounds['lower', '_p_est', 'g_m_T'] = 0
        mhe.bounds['lower', '_p_est', 'g_p_L'] = 0
        mhe.bounds['lower', '_p_est', 'g_p_T'] = 0
        mhe.bounds['lower', '_p_est', 'theta_LacI'] = 0
        mhe.bounds['lower', '_p_est', 'theta_TetR'] = 0
        mhe.bounds['lower', '_p_est', 'theta_IPTG'] = 0
        mhe.bounds['lower', '_p_est', 'theta_aTc'] = 0

        mhe.setup()

        return mhe

    def simulator_mhe(self):
        """Defines the parameters of the simulator.

        Returns:
            do_mpc.simulator.Simulator: Instance of the simulator.
        """

        simulator = do_mpc.simulator.Simulator(self.model)
        simulator.set_param(t_step=self.t_step)

        p_template_sim = simulator.get_p_template()

        def p_fun_sim(t_now):

            p_template_sim['k_m0_L'] = 3.20e-2
            p_template_sim['k_m0_T'] = 1.19e-1
            p_template_sim['k_m_L'] = 8.30
            p_template_sim['k_m_T'] = 2.06
            p_template_sim['k_p_L'] = 9.726e-1
            p_template_sim['k_p_T'] = 9.726e-1
            p_template_sim['g_m_L'] = 1.386e-1
            p_template_sim['g_m_T'] = 1.386e-1
            p_template_sim['g_p_L'] = 1.65e-2
            p_template_sim['g_p_T'] = 1.65e-2
            p_template_sim['theta_LacI'] = 31.94
            p_template_sim['theta_TetR'] = 30.00
            p_template_sim['theta_IPTG'] = 9.06e-2
            p_template_sim['theta_aTc'] = 11.65
            return p_template_sim

        simulator.set_p_fun(p_fun_sim)

        simulator.setup()

        return simulator

    def estimation_loop(self, x_0, p_est0, steps):
        """Defines the estimation loop.

        Args:
            x_0 (numpy.ndarray[float64]): Initial conditions.
            p_est0 (numpy.ndarray[float64]): Initial estimation of parameters
            steps (int): Number of estimation loop steps.

        Returns:
            dict: The parameter estimates.
        """

        self.estimator.reset_history()
        self.simulator.reset_history()

        self.simulator.x0 = x_0
        self.estimator.x0 = x_0
        self.estimator.p_est0 = p_est0

        self.estimator.set_initial_guess()

        for k in range(steps):

            u = np.random.randn(2, 1)
            v0 = 0.001*np.random.randn(self.model.n_v, 1)
            y_next = self.simulator.make_step(u, v0=v0)
            x_0 = self.estimator.make_step(y_next)

        self.plot_estimates(self.estimator.data, self.simulator.data)

        return self.estimator.data

    def plot_estimates(self, estimator_data, simulator_data):
        """Plot the parameter estimate against the actual value.

        Args:
            estimator_data (dict): Parameter estimation.
            simulator_data (dict): Actual parameters.
        """
        mhe_graphics = do_mpc.graphics.Graphics(estimator_data)
        sim_graphics = do_mpc.graphics.Graphics(simulator_data)

        fig_x = 20
        fig_y = 10

        plt.rcParams['axes.grid'] = True
        plt.rcParams['font.size'] = 14

        figure, axes = plt.subplots(2, sharex=True, figsize=(fig_x, fig_y))

        mhe_graphics.add_line(var_type='_p', var_name='k_m0_L', axis=axes[0])
        mhe_graphics.add_line(var_type='_p', var_name='k_m0_T', axis=axes[1])
        sim_graphics.add_line(var_type='_p', var_name='k_m0_L', axis=axes[0])
        sim_graphics.add_line(var_type='_p', var_name='k_m0_T', axis=axes[1])
        axes[0].set_title('k_m0_L')
        axes[1].set_title('k_m0_T')
        axes[0].legend(sim_graphics.result_lines['_p', 'k_m0_L'] +
                       mhe_graphics.result_lines['_p', 'k_m0_L'], ['True', 'Estimation'], loc='upper right')
        axes[1].legend(sim_graphics.result_lines['_p', 'k_m0_T'] +
                       mhe_graphics.result_lines['_p', 'k_m0_T'], ['True', 'Estimation'], loc='upper right')

        figure, axes = plt.subplots(2, sharex=True, figsize=(fig_x, fig_y))
        mhe_graphics.add_line(var_type='_p', var_name='k_m_L', axis=axes[0])
        mhe_graphics.add_line(var_type='_p', var_name='k_m_T', axis=axes[1])
        sim_graphics.add_line(var_type='_p', var_name='k_m_L', axis=axes[0])
        sim_graphics.add_line(var_type='_p', var_name='k_m_T', axis=axes[1])
        axes[0].set_title('k_m_L')
        axes[1].set_title('k_m_T')
        axes[0].legend(sim_graphics.result_lines['_p', 'k_m_L'] + mhe_graphics.result_lines['_p', 'k_m_L'], ['True', 'Estimation'], loc='upper right')
        axes[1].legend(sim_graphics.result_lines['_p', 'k_m_T'] + mhe_graphics.result_lines['_p', 'k_m_T'], ['True', 'Estimation'], loc='upper right')

        figure, axes = plt.subplots(2, sharex=True, figsize=(fig_x, fig_y))
        mhe_graphics.add_line(var_type='_p', var_name='k_p_L', axis=axes[0])
        mhe_graphics.add_line(var_type='_p', var_name='k_p_T', axis=axes[1])
        sim_graphics.add_line(var_type='_p', var_name='k_p_L', axis=axes[0])
        sim_graphics.add_line(var_type='_p', var_name='k_p_T', axis=axes[1])
        axes[0].set_title('k_p_L')
        axes[1].set_title('k_p_T')
        axes[0].legend(sim_graphics.result_lines['_p', 'k_p_L'] + mhe_graphics.result_lines['_p', 'k_p_L'], ['True', 'Estimation'], loc='upper right')
        axes[1].legend(sim_graphics.result_lines['_p', 'k_p_T'] + mhe_graphics.result_lines['_p', 'k_p_T'], ['True', 'Estimation'], loc='upper right')

        figure, axes = plt.subplots(2, sharex=True, figsize=(fig_x, fig_y))
        mhe_graphics.add_line(var_type='_p', var_name='g_m_L', axis=axes[0])
        mhe_graphics.add_line(var_type='_p', var_name='g_m_T', axis=axes[1])
        sim_graphics.add_line(var_type='_p', var_name='g_m_L', axis=axes[0])
        sim_graphics.add_line(var_type='_p', var_name='g_m_T', axis=axes[1])
        axes[0].set_title('g_m_L')
        axes[1].set_title('g_m_T')
        axes[0].legend(sim_graphics.result_lines['_p', 'g_m_L'] + mhe_graphics.result_lines['_p', 'g_m_L'], ['True', 'Estimation'], loc='upper right')
        axes[1].legend(sim_graphics.result_lines['_p', 'g_m_T'] + mhe_graphics.result_lines['_p', 'g_m_T'], ['True', 'Estimation'], loc='upper right')

        figure, axes = plt.subplots(2, sharex=True, figsize=(fig_x, fig_y))
        mhe_graphics.add_line(var_type='_p', var_name='g_p_L', axis=axes[0])
        mhe_graphics.add_line(var_type='_p', var_name='g_p_T', axis=axes[1])
        sim_graphics.add_line(var_type='_p', var_name='g_p_L', axis=axes[0])
        sim_graphics.add_line(var_type='_p', var_name='g_p_T', axis=axes[1])
        axes[0].set_title('g_p_L')
        axes[1].set_title('g_p_T')
        axes[0].legend(sim_graphics.result_lines['_p', 'g_p_L'] + mhe_graphics.result_lines['_p', 'g_p_L'], ['True', 'Estimation'], loc='upper right')
        axes[1].legend(sim_graphics.result_lines['_p', 'g_p_T'] + mhe_graphics.result_lines['_p', 'g_p_T'], ['True', 'Estimation'], loc='upper right')

        figure, axes = plt.subplots(2, sharex=True, figsize=(fig_x, fig_y))
        mhe_graphics.add_line(var_type='_p', var_name='theta_LacI', axis=axes[0])
        mhe_graphics.add_line(var_type='_p', var_name='theta_TetR', axis=axes[1])
        sim_graphics.add_line(var_type='_p', var_name='theta_LacI', axis=axes[0])
        sim_graphics.add_line(var_type='_p', var_name='theta_TetR', axis=axes[1])
        axes[0].set_title('theta_LacI')
        axes[1].set_title('theta_TetR')
        axes[0].legend(sim_graphics.result_lines['_p', 'theta_LacI'] +
                       mhe_graphics.result_lines['_p', 'theta_LacI'], ['True', 'Estimation'], loc='upper right')
        axes[1].legend(sim_graphics.result_lines['_p', 'theta_TetR'] +
                       mhe_graphics.result_lines['_p', 'theta_TetR'], ['True', 'Estimation'], loc='upper right')

        figure, axes = plt.subplots(2, sharex=True, figsize=(fig_x, fig_y))
        mhe_graphics.add_line(var_type='_p', var_name='theta_IPTG', axis=axes[0])
        mhe_graphics.add_line(var_type='_p', var_name='theta_aTc', axis=axes[1])
        sim_graphics.add_line(var_type='_p', var_name='theta_IPTG', axis=axes[0])
        sim_graphics.add_line(var_type='_p', var_name='theta_aTc', axis=axes[1])
        axes[0].set_title('theta_IPTG')
        axes[1].set_title('theta_aTc')
        axes[0].legend(sim_graphics.result_lines['_p', 'theta_IPTG'] +
                       mhe_graphics.result_lines['_p', 'theta_IPTG'], ['True', 'Estimation'], loc='upper right')
        axes[1].legend(sim_graphics.result_lines['_p', 'theta_aTc'] +
                       mhe_graphics.result_lines['_p', 'theta_aTc'], ['True', 'Estimation'], loc='upper right')

        mhe_graphics.plot_results()
        mhe_graphics.reset_axes()
