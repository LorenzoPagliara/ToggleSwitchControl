import numpy as np
from .ToggleSwitch import *


class ToggleSwitchSimplified(ToggleSwitch):
    """Class representing the simplified model of a Genetic Toggle Switch."""

    def __init__(self) -> None:

        super().__init__()
        self.state = np.zeros(2)

        self.k_1_0 = (self.k_m0_L*self.k_p_L) / (self.g_m_L*self.theta_LacI*self.g_p_L)
        self.k_1 = (self.k_m_L*self.k_p_L) / (self.g_m_L*self.theta_LacI*self.g_p_L)
        self.k_2_0 = (self.k_m0_T*self.k_p_T) / (self.g_m_T*self.theta_TetR*self.g_p_T)
        self.k_2 = (self.k_m_T*self.k_p_T) / (self.g_m_T*self.theta_TetR*self.g_p_T)

    def make_step(self, u,):
        """It calculates the new state from the previous one and the value of the control input, according to the deterministic model.

        Args:
            u (numpy.ndarray[float64]): Current control input.
            noise (bool): If it is True it adds noise to the model.
            step (numpy.ndarray[float64]): Discretization step for each state.

        Returns:
            numpy.ndarray[float64]: Next state.
        """

        x1_km1 = self.state[0]
        x2_km1 = self.state[1]

        aTc = np.clip(u[0], 0, 35)
        iptg = np.clip(u[1], 0, 0.35)

        x1_k = np.clip(self.k_1_0 + (self.k_1/(1 + (x2_km1**2) *
                       (1/((1 + (aTc/self.theta_aTc)**self.eta_aTc)**self.eta_TetR)))) - x1_km1, 0, 150)
        x2_k = np.clip(self.k_2_0 + (self.k_2/(1 + (x1_km1**2) *
                       (1/((1 + (iptg/self.theta_IPTG)**self.eta_IPTG)**self.eta_LacI)))) - x2_km1, 0, 100)

        self.state = np.array([x1_k, x2_k])

        return self.state
