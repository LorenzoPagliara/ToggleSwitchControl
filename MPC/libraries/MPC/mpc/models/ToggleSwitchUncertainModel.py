from .ToggleSwitchModel import *


class ToggleSwitchUncertainModel(ToggleSwitchModel):
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

    def set_model(stochasticity=False, LacI_ref=750, TetR_ref=300, *args):
        """Defines the model equations and the cost function. It sets some parameters as uncertain.

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

        theta_LacI = model.set_variable(var_type='parameter', var_name='theta_LacI')  # 31.94
        theta_TetR = model.set_variable(var_type='parameter', var_name='theta_TetR')  # 30.00

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
        model.set_expression(expr_name='cost', expr=((lacI - LacI_ref)**2 + (tetR - TetR_ref)**2))

        model.setup()

        return model

    def set_uncertain_parameters(self, controller):
        """Sets the range of variability of uncertain parameters.

        Args:
            controller (do_mpc.controller.MPC): Controller instance.

        Returns:
            do_mpc.controller.MPC: Instance of the controller.
        """
        theta_LacI_values = np.array([31.94, 32, 30])
        theta_TetR_values = np.array([30, 31, 29])

        controller.set_uncertainty_values(
            theta_LacI=theta_LacI_values,
            theta_TetR=theta_TetR_values
        )

        return controller

    def uncertain_parameters_function(self, simulator):
        """Set the runtime update function for uncertain parameters.

        Args:
            simulator (do_mpc.simulator.Simulator): Instance of the simulator.

        Returns:
            do_mpc.simulator.Simulator: Instance of the simulator.
        """
        p_template = simulator.get_p_template()

        def p_fun(t_now):

            p_template['theta_LacI'] = np.random.uniform(low=30, high=32)
            p_template['theta_TetR'] = np.random.uniform(low=29, high=31)

            return p_template

        simulator.set_p_fun(p_fun)

        return simulator
