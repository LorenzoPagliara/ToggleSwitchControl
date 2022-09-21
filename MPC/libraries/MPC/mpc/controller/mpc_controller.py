import do_mpc
from casadi import *


def template_model(stochasticity=False, LacI_ref=750, TetR_ref=300):

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
    model.set_rhs('mRNA_LacI', k_m0_L + k_m_L*(1 / (1 + ((tetR/theta_TetR) * (1 / (1 + (v1/theta_aTc)**eta_aTc)))**eta_TetR)) -
                  g_m_L * mRNA_LacI, process_noise=stochasticity)
    model.set_rhs('mRNA_TetR', k_m0_T + k_m_T*(1 / (1 + ((lacI/theta_LacI) * (1 / (1 + (v2/theta_IPTG)**eta_IPTG)))**eta_LacI)) -
                  g_m_T * mRNA_TetR, process_noise=stochasticity)
    model.set_rhs('LacI', k_p_L * mRNA_LacI - g_p_L * lacI, process_noise=stochasticity)
    model.set_rhs('TetR', k_p_T * mRNA_TetR - g_p_T * tetR, process_noise=stochasticity)
    model.set_rhs('v1', (k_in_aTc * (aTc - v1)) * (aTc > v1) + (k_out_aTc * (aTc - v1)) * (aTc <= v1), process_noise=stochasticity)
    model.set_rhs('v2', (k_in_IPTG * (iptg - v2)) * (iptg > v2) + (k_out_IPTG * (iptg - v2)) * (iptg <= v2), process_noise=stochasticity)

    # The process noise w is used to simulate a disturbed system in the Simulator

    # Measurement noise
    if stochasticity:
        model.n_v = np.random.randn(6, 1)

    # Cost function
    model.set_expression(expr_name='cost', expr=((lacI - LacI_ref)**2 + (tetR - TetR_ref)**2))

    model.setup()

    return model


def template_mpc(model, setup_mpc):

    mpc = do_mpc.controller.MPC(model)

    mpc.set_param(**setup_mpc)

    # Cost function
    mterm = model.aux['cost']
    lterm = model.aux['cost']

    mpc.set_objective(mterm=mterm, lterm=lterm)
    mpc.set_rterm(aTc=1, IPTG=1)

    # Constraints
    mpc.bounds['lower', '_x', 'mRNA_LacI'] = 3.20e-2
    mpc.bounds['lower', '_x', 'mRNA_TetR'] = 1.19e-1

    mpc.bounds['lower', '_x', 'LacI'] = 0
    mpc.bounds['lower', '_x', 'TetR'] = 0

    mpc.bounds['lower', '_x', 'v1'] = 0
    mpc.bounds['lower', '_x', 'v2'] = 0

    mpc.bounds['lower', '_u', 'aTc'] = 0
    mpc.bounds['upper', '_u', 'aTc'] = 35

    mpc.bounds['lower', '_u', 'IPTG'] = 0
    mpc.bounds['upper', '_u', 'IPTG'] = 0.35

    mpc.setup()

    return mpc


def template_simulator(model, t_step):

    simulator = do_mpc.simulator.Simulator(model)
    simulator.set_param(t_step=t_step)

    simulator.setup()

    return simulator
