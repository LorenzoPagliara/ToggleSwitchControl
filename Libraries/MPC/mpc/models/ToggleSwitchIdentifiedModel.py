from .ToggleSwitchModel import *


class ToggleSwitchIdentifiedModel(ToggleSwitchModel):
    """
    Class extending the model used by the MPC and representing a model with randomic parameters.
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

    def set_model(self, stochasticity=False, *args):
        """Defines the model equations and the cost function. Set parameters to random values.

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

        data = args[0]

        # Model parameters
        k_m0_L = np.mean(data['_p', 'k_m0_L'])
        k_m0_T = np.mean(data['_p', 'k_m0_T'])
        k_m_L = np.mean(data['_p', 'k_m_L'])
        k_m_T = np.mean(data['_p', 'k_m_T'])
        k_p_L = np.mean(data['_p', 'k_p_L'])
        k_p_T = np.mean(data['_p', 'k_p_T'])
        g_m_L = np.mean(data['_p', 'g_m_L'])
        g_m_T = np.mean(data['_p', 'g_m_T'])
        g_p_L = np.mean(data['_p', 'g_p_L'])
        g_p_T = np.mean(data['_p', 'g_p_T'])
        theta_LacI = np.mean(data['_p', 'theta_LacI'])
        theta_TetR = np.mean(data['_p', 'theta_TetR'])
        theta_IPTG = np.mean(data['_p', 'theta_IPTG'])
        theta_aTc = np.mean(data['_p', 'theta_aTc'])
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

        # The process noise w is used to simulate a disturbed system in the Simulator

        # Measurement noise
        if stochasticity:
            model.n_v = np.random.randn(6, 1)

        # Cost function
        model.set_expression(expr_name='cost', expr=((lacI - self.LacI_ref)**2 + (tetR - self.TetR_ref)**2))

        model.setup()

        return model
