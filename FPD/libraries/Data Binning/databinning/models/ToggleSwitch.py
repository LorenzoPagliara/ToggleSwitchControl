import numpy as np


class ToggleSwitch():

    def __init__(self) -> None:

        self.state = np.zeros(6)

        self.k_m0_L = 3.20e-2
        self.k_m0_T = 1.19e-1
        self.k_m_L = 8.30
        self.k_m_T = 2.06
        self.k_p_L = 9.726e-1
        self.k_p_T = 9.726e-1
        self.g_m_L = 1.386e-1
        self.g_m_T = 1.386e-1
        self.g_p_L = 1.65e-2
        self.g_p_T = 1.65e-2
        self.theta_LacI = 31.94
        self.theta_TetR = 30.00
        self.theta_IPTG = 9.06e-2
        self.theta_aTc = 11.65
        self.eta_LacI = 2
        self.eta_TetR = 2
        self.eta_IPTG = 2
        self.eta_aTc = 2
        self.k_in_aTc = 2.75e-2
        self.k_out_aTc = 2.00e-2
        self.k_in_IPTG = 1.62e-1
        self.k_out_IPTG = 1.11e-1

    def initialState(self, init):
        self.state = init

    def getState(self):
        return self.state

    def make_step(self, u):

        mRNA_LacI = self.state[0]
        mRNA_TetR = self.state[1]
        lacI = self.state[2]
        tetR = self.state[3]
        v1 = self.state[4]
        v2 = self.state[5]

        aTc = np.clip(u[0], 0, 35)
        iptg = np.clip(u[1], 0, 0.35)

        self.state = np.array([
            self.k_m0_L + self.k_m_L*(1 / (1 + ((tetR/self.theta_TetR) * (1 / (1 + (
                v1/self.theta_aTc)**self.eta_aTc)))**self.eta_TetR)) - self.g_m_L * mRNA_LacI,
            self.k_m0_T + self.k_m_T*(1 / (1 + ((lacI/self.theta_LacI) * (1 / (1 + (
                v2/self.theta_IPTG)**self.eta_IPTG)))**self.eta_LacI)) - self.g_m_T * mRNA_TetR,
            self.k_p_L * mRNA_LacI - self.g_p_L * lacI,
            self.k_p_T * mRNA_TetR - self.g_p_T * tetR,
            (self.k_in_aTc * (aTc - v1)) * (aTc > v1) +
            (self.k_out_aTc * (aTc - v1)) * (aTc <= v1),
            (self.k_in_IPTG * (iptg - v2)) * (iptg > v2) +
            (self.k_out_IPTG * (iptg - v2)) * (iptg <= v2)
        ])

        return self.state


class ToggleSwitchSimplified(ToggleSwitch):

    def __init__(self) -> None:

        super().__init__()
        self.state = np.zeros(2)

        self.k_1_0 = (self.k_m0_L*self.k_p_L) / (self.g_m_L*self.theta_LacI*self.g_p_L)
        self.k_1 = (self.k_m_L*self.k_p_L) / (self.g_m_L*self.theta_LacI*self.g_p_L)
        self.k_2_0 = (self.k_m0_T*self.k_p_T) / (self.g_m_T*self.theta_TetR*self.g_p_T)
        self.k_2 = (self.k_m_T*self.k_p_T) / (self.g_m_T*self.theta_TetR*self.g_p_T)

    def make_step(self, u):

        x1 = self.state[0]
        x2 = self.state[1]

        aTc = np.clip(u[0], 0, 35)
        iptg = np.clip(u[1], 0, 0.35)

        dx1 = np.clip(self.k_1_0 + (self.k_1/(1 + (x2**2) * (1/((1 + (aTc/self.theta_aTc)**self.eta_aTc)**self.eta_TetR)))) - x1, 0, 150)
        dx2 = np.clip(self.k_2_0 + (self.k_2/(1 + (x1**2) * (1/((1 + (iptg/self.theta_IPTG)**self.eta_IPTG)**self.eta_LacI)))) - x2, 0, 100)

        self.state = np.array([dx1, dx2])

        return self.state