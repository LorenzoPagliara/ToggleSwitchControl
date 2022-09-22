import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate


def save_results(mpc):
    """Saves all the simulation results in a dictionary. 

    Args:
        mpc (do_mpc.controller.MPC): The do-mpc controller instance.

    Returns:
        dict: Dictionary containing all data.
    """

    states = {
        'x1': [x[0] for x in mpc.data['_x', 'x1'].tolist()],
        'x2': [x[0] for x in mpc.data['_x', 'x2'].tolist()]
    }

    inputs = {
        'aTc': [x[0] for x in mpc.data['_u', 'aTc'].tolist()],
        'IPTG': [x[0] for x in mpc.data['_u', 'IPTG'].tolist()]
    }

    cost = [x[0] for x in mpc.data['_aux', 'cost'].tolist()]
    time = [x[0] for x in mpc.data['_time'].tolist()]

    data = {
        'states': states,
        'inputs': inputs,
        'cost': cost,
        'time': time,
        'ISE': 0,
        'ITAE': 0
    }

    return data


def plot_results(data, LacI_ref, TetR_ref):
    """Plot all the simulation results.

    Args:
        data (dict): Simulation results.
    """

    fig_x = 20
    fig_y = 10

    plt.rcParams['axes.grid'] = True
    plt.rcParams['font.size'] = 14

    figure, axes = plt.subplots(2, sharex=True, figsize=(fig_x, fig_y))

    axes[0].set_ylabel(r'$a.u.$')
    axes[0].set_title(r'$x_1$')

    axes[0].plot(data['time'], data['states']['x1'], color='b')
    axes[0].plot(data['time'], LacI_ref*np.ones(len(data['time'])), color='k', linestyle='--')
    axes[0].legend([r'$x_1$', r'$x_1$ Target'], loc='upper right')

    axes[1].set_ylabel(r'$a.u.$')
    axes[1].set_title(r'$x_1$')
    axes[1].plot(data['time'], data['states']['x2'], color='m')
    axes[1].plot(data['time'], TetR_ref*np.ones(len(data['time'])), color='k', linestyle='--')
    axes[1].legend([r'$x_2$', r'$x_2$ Target'], loc='upper right')
    axes[1].set_xlabel('time [min]')
    figure.set_facecolor("white")

    # -------------------- External inducers concentrations -------------------- #
    # --- aTc --- #
    figure, axes = plt.subplots(2, sharex=True, figsize=(fig_x, fig_y))
    axes[0].set_ylabel('')
    axes[0].set_title('aTc')
    axes[0].plot(data['time'], data['inputs']['aTc'], color='y')
    axes[0].legend(['aTc'], loc='upper right')

    # --- IPTG --- #
    axes[1].set_ylabel('')
    axes[1].set_title('IPTG')
    axes[1].plot(data['time'], data['inputs']['IPTG'], color='r')
    axes[1].legend(['IPTG'], loc='upper right')
    axes[1].set_xlabel('time [min]')
    figure.set_facecolor("white")

    # -------------------- Cost -------------------- #
    figure, axes = plt.subplots(1, figsize=(fig_x, fig_y/2))
    axes.set_title('Cost')
    axes.plot(data['time'], data['cost'], color='g')
    axes.legend(['Cost'], loc='upper right')
    axes.set_xlabel('time [s]')


def compute_performance_metrics(x1_avg, x2_avg, avg_x, total_time, t_step, LacI_ref, TetR_ref):
    """Compute the performance metrics such as: ITAE and ISE.

    Args:
        x1_avg (numpy.ndarray[float64]): First state average trajectory.
        x2_avg (numpy.ndarray[float64]): Second state average trajectory.
        avg_x (numpy.ndarray[float64]): X-axis for average trajectories.
        total_time (float64): Total simulation time.
        t_step (float64): Sample time.

    Returns:
        float64, float64: ISE and ITAE.
    """

    e1_bar = np.array([(x - LacI_ref) / LacI_ref for x in x1_avg])
    e2_bar = np.array([(x - TetR_ref) / TetR_ref for x in x2_avg])

    fe1_bar = interpolate.interp1d(avg_x, e1_bar, fill_value="extrapolate")
    e1_bar = fe1_bar(np.arange(0, total_time, t_step))
    fe2_bar = interpolate.interp1d(avg_x, e2_bar, fill_value="extrapolate")
    e2_bar = fe2_bar(np.arange(0, total_time, t_step))

    e_bar = np.array([np.linalg.norm([e1_bar[i], e2_bar[i]]) for i in range(len(e1_bar))])

    ISE = np.sum((e_bar)**2)
    ITAE = np.sum([np.abs(e_bar[i])*(t_step*(i+1)) for i in range(len(e_bar))])

    return ISE, ITAE
