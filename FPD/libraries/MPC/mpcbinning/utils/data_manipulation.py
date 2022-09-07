import matplotlib.pyplot as plt
import numpy as np


def save_results(mpc, t_step):

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


def plot_results(data, total_time):

    fig_x = 20
    fig_y = 10

    plt.rcParams['axes.grid'] = True
    plt.rcParams['font.size'] = 14

    # -------------------- Proteins -------------------- #
    # --- LacI --- #
    figure_proteins, axes = plt.subplots(
        2, sharex=True, figsize=(fig_x, fig_y))

    axes[0].set_ylabel('')
    axes[0].set_title('LacI')

    line_LacI, = axes[0].plot(data['time'], data['states']['x1'], color='b')
    line_ref_LacI, = axes[0].plot(data['time'], 23.48*np.ones(len(data['time'])), color='k', linestyle='--')
    axes[0].legend(['LacI', 'Avg LacI traj', 'LacI target'], loc='upper right')

    # --- TetR --- #
    axes[1].set_ylabel('')
    axes[1].set_title('TetR')
    line_TetR, = axes[1].plot(data['time'], data['states']['x2'], color='m')
    line_ref_TetR, = axes[1].plot(data['time'], 10.00*np.ones(len(data['time'])), color='k', linestyle='--')
    axes[1].legend(['TetR', 'Avg TetR traj', 'TetR target'], loc='upper right')
    axes[1].set_xlabel('time [s]')

    # -------------------- External inducers concentrations -------------------- #
    # --- aTc --- #
    figure_inducers, axes2 = plt.subplots(
        2, sharex=True, figsize=(fig_x, fig_y))
    axes2[0].set_ylabel('')
    axes2[0].set_title('aTc')
    line_aTc, = axes2[0].plot(data['time'], data['inputs']['aTc'], color='y')
    axes2[0].legend(['aTc'], loc='upper right')

    # --- IPTG --- #
    axes2[1].set_ylabel('')
    axes2[1].set_title('IPTG')
    line_IPTG, = axes2[1].plot(data['time'], data['inputs']['IPTG'], color='r')
    axes2[1].legend(['IPTG'], loc='upper right')
    axes2[1].set_xlabel('time [s]')

    # -------------------- Cost -------------------- #
    figure_cost, axes4 = plt.subplots(1, figsize=(fig_x, fig_y/2))
    axes4.set_ylabel('')
    axes4.set_title('Cost')
    line_cost, = axes4.plot(data['time'], data['cost'], color='g')
    axes4.legend(['Cost'], loc='upper right')
    axes4.set_xlabel('time [s]')
