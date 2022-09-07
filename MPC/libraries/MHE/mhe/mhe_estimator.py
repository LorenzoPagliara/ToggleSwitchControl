import numpy as np
import do_mpc
from casadi import *


def template_model(stochasticity=False):

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


def template_mhe(model, setup_mhe):

    mhe = do_mpc.estimator.MHE(model, ['k_m0_L', 'k_m0_T', 'k_m_L', 'k_m_T', 'k_p_L', 'k_p_T', 'g_m_L', 'g_m_T', 'g_p_L', 'g_p_T', 'theta_LacI',
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


def template_simulator(model, t_step):

    simulator = do_mpc.simulator.Simulator(model)
    simulator.set_param(t_step=t_step)

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
