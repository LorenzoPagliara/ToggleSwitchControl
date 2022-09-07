from phaseportrait.PhasePortrait2D import *
import matplotlib.pyplot as plt
import numpy as np


def step(x1, x2, *, aTc=0, iptg=0):
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
    eta_LacI = 2
    eta_TetR = 2
    eta_IPTG = 2
    eta_aTc = 2
    k_1_0 = (k_m0_L*k_p_L) / (g_m_L*theta_LacI*g_p_L)
    k_1 = (k_m_L*k_p_L) / (g_m_L*theta_LacI*g_p_L)
    k_2_0 = (k_m0_T*k_p_T) / (g_m_T*theta_TetR*g_p_T)
    k_2 = (k_m_T*k_p_T) / (g_m_T*theta_TetR*g_p_T)

    return k_1_0 + (k_1/(1 + (x2**2) * (1/((1 + (aTc/theta_aTc)**eta_aTc)**eta_TetR)))) - x1, k_2_0 + (k_2/(1 + (x1**2) * (1/((1 + (iptg/theta_IPTG)**eta_IPTG)**eta_LacI)))) - x2


phase_diagram = PhasePortrait2D(step, [0, 150], Density=3, Title='Toggle Switch Phase Diagram', xlabel='X1', ylabel='X2')
phase_diagram.add_slider('aTc', valinit=0, valinterval=35)
phase_diagram.add_slider('iptg', valinit=0, valinterval=0.35)
phase_diagram.add_nullclines(precision=0.005, alpha=1)
phase_diagram.plot()

plt.show()
