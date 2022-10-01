from .ToggleSwitchModel import *


class ToggleSwitchSimplifiedModel(ToggleSwitchModel):

    def __init__(self, stochasticity, LacI_ref, TetR_ref, t_step, total_time, avg_period, *args) -> None:
        super().__init__(stochasticity, LacI_ref, TetR_ref, t_step, total_time, avg_period, *args)

    def set_model(self, stochasticity=False, *args):
        """Defines the mathematical model for the toggle switch in the form of differential equations (ODE).

        Args:
            stochasticity (bool): Determines the presence or absence of stochasticity in the model.

        Returns:
            do_mpc.model.Model: The do-mpc model instance.
        """

        model = do_mpc.model.Model(model_type='continuous')

        # Model states
        x1 = model.set_variable(var_type='_x', var_name='x1')
        x2 = model.set_variable(var_type='_x', var_name='x2')

        # Model input
        atc = model.set_variable(var_type='_u', var_name='aTc')
        iptg = model.set_variable(var_type='_u', var_name='IPTG')

        # Model parameters
        k_m0_L = 3.20e-2
        k_m0_T = 1.19e-1
        k_m_L = 8.30
        k_m_T = 2.06
        k_p_L = 9.726e-1
        k_p_T = 9.726e-1
        g_m_L = 1.386e-1
        g_m_T = 1.386e-1
        g_p = 1.65e-2
        theta_LacI = 31.94
        theta_TetR = 30.00
        theta_IPTG = 9.06e-2
        theta_aTc = 11.65
        eta_LacI = 2.00
        eta_TetR = 2.00
        eta_IPTG = 2.00
        eta_aTc = 2.00

        k_1_0 = (k_m0_L*k_p_L) / (g_m_L*theta_LacI*g_p)
        k_1 = (k_m_L*k_p_L) / (g_m_L*theta_LacI*g_p)
        k_2_0 = (k_m0_T*k_p_T) / (g_m_T*theta_TetR*g_p)
        k_2 = (k_m_T*k_p_T) / (g_m_T*theta_TetR*g_p)

        # Defining model's equations
        dx1 = k_1_0 + (k_1/(1 + (x2**2) * (1/((1 + (atc/theta_aTc)**eta_aTc)**eta_TetR)))) - x1
        dx2 = k_2_0 + (k_2/(1 + (x1**2) * (1/((1 + (iptg/theta_IPTG)**eta_IPTG)**eta_LacI)))) - x2

        model.set_rhs('x1', dx1, process_noise=stochasticity)
        model.set_rhs('x2', dx2, process_noise=stochasticity)

        # Cost function
        model.set_expression(expr_name='cost', expr=(((x1 - self.LacI_ref)/self.LacI_ref)**2 + ((x2 - self.TetR_ref)/self.TetR_ref)**2))

        if stochasticity:
            model.n_v = np.random.randn(2, 1)

        model.setup()

        return model

    def set_constraints(self, mpc):

        mpc.bounds['lower', '_x', 'x1'] = 0.426
        mpc.bounds['lower', '_x', 'x2'] = 1.686

        mpc.bounds['lower', '_u', 'aTc'] = 0
        mpc.bounds['upper', '_u', 'aTc'] = 35

        mpc.bounds['lower', '_u', 'IPTG'] = 0
        mpc.bounds['upper', '_u', 'IPTG'] = 0.35

        return mpc

    def set_trajectories(self, x, u):
        """It updates the dictionary of state trajectories based on closed-loop results obtained by the controller.

        Args:
            data (dict): Closed-loop simulation results.
        """

        states = {
            'LacI': x[:, 0],
            'TetR': x[:, 1],
        }

        inputs = {
            'aTc': u[:, 0],
            'IPTG': u[:, 1]
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

        self.trajectories = {
            'states': states,
            'inputs': inputs,
            'avg_traj': avg_trajectories,
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

        f = open('./results/' + type + '/' + name + '.json', mode)
        data_json = json.dumps(self.trajectories)
        f.write(data_json)
        f.close()

    def update_protein(self, i, time, avg_time, LacI, TetR, avg_LacI, avg_TetR, LacI_ref, TetR_ref, lines):
        lines[0].set_data(time[:i], LacI[:i])
        lines[1].set_data(avg_time[:i], avg_LacI[:i])
        lines[2].set_data(time[:i], LacI_ref*np.ones(len(time))[:i])
        lines[3].set_data(time[:i], TetR[:i])
        lines[4].set_data(avg_time[:i], avg_TetR[:i])
        lines[5].set_data(time[:i], TetR_ref*np.ones(len(time))[:i])
        return lines
