import do_mpc
from casadi import *


def template_model(stochasticity):
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

    model.set_rhs('x1', k_1_0 + (k_1/(1 + (x2**2) * (1/((1 + (atc/theta_aTc)**eta_aTc)**eta_TetR)))) - x1, process_noise=stochasticity)
    model.set_rhs('x2', k_2_0 + (k_2/(1 + (x1**2) * (1/((1 + (iptg/theta_IPTG)**eta_IPTG)**eta_LacI)))) - x2, process_noise=stochasticity)

    # Model references
    LacI_ref = 23.48
    TetR_ref = 10.00

    # Cost function
    model.set_expression(expr_name='cost', expr=((x1 - LacI_ref)**2 + (x2 - TetR_ref)**2))

    if stochasticity:
        model.n_v = np.random.randn(2, 1)

    model.setup()

    return model


def template_mpc(model, setup_mpc):
    """Configure and setup the MPC controller, given the model previously defined. It defines the cost function and the constraints on state variables and inputs.

    Args:
        model (do_mpc.model.Model): The do-mpc model instance.
        setup_mpc (dict): Controller parameters.

    Returns:
        _type_: _description_
    """

    mpc = do_mpc.controller.MPC(model)

    mpc.set_param(**setup_mpc)

    # Cost function
    mterm = model.aux['cost']
    lterm = model.aux['cost']

    mpc.set_objective(mterm=mterm, lterm=lterm)
    mpc.set_rterm(aTc=1, IPTG=1)

    # Constraints
    mpc.bounds['lower', '_x', 'x1'] = 0.426
    mpc.bounds['lower', '_x', 'x2'] = 1.686

    mpc.bounds['lower', '_u', 'aTc'] = 0
    mpc.bounds['upper', '_u', 'aTc'] = 35

    mpc.bounds['lower', '_u', 'IPTG'] = 0
    mpc.bounds['upper', '_u', 'IPTG'] = 0.35

    mpc.setup()

    return mpc


def template_simulator(model, t_step):
    """Configure and setup the simulator.

    Args:
        model (do_mpc.model.Model): The do-mpc model instance.
        t_step (float64): Sampling time.

    Returns:
        do_mpc.simulator.Simulator: The do-mpc simulator instance.
    """

    simulator = do_mpc.simulator.Simulator(model)
    simulator.set_param(t_step=t_step)

    simulator.setup()

    return simulator
