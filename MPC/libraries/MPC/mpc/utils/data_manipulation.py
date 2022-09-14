import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy import interpolate
import numpy as np
import json


def save_results(mpc, avg_period, t_step):

    states = {
        'mRNA_LacI': [x[0] for x in mpc.data['_x', 'mRNA_LacI'].tolist()],
        'mRNA_TetR': [x[0] for x in mpc.data['_x', 'mRNA_TetR'].tolist()],
        'LacI': [x[0] for x in mpc.data['_x', 'LacI'].tolist()],
        'TetR': [x[0] for x in mpc.data['_x', 'TetR'].tolist()],
        'v1': [x[0] for x in mpc.data['_x', 'v1'].tolist()],
        'v2': [x[0] for x in mpc.data['_x', 'v2'].tolist()],
    }

    u1 = [x[0] for x in mpc.data['_u', 'aTc'].tolist()]
    u2 = [x[0] for x in mpc.data['_u', 'IPTG'].tolist()]

    for i in range(len(u1)):
        if i % 15 != 0:
            u1[i] = u1[i-1]
            u2[i] = u2[i-1]

    inputs = {
        'aTc': u1,
        'IPTG': u2
    }

    # Get average trajectory every 240 minutes (14400 s)
    # Get samples every 5 minutes (300 s)
    # Samples in 240 minutes = 240/5 = 48 samples
    avg_samples_range = int(avg_period/t_step)

    avg_LacI = [np.mean(states['LacI'][x:x + avg_samples_range])
                for x in range(0, len(states['LacI']), avg_samples_range)]

    avg_TetR = [np.mean(states['TetR'][x:x + avg_samples_range])
                for x in range(0, len(states['TetR']), avg_samples_range)]

    avg_trajectory = {
        'LacI': avg_LacI,
        'TetR': avg_TetR
    }

    cost = [x[0] for x in mpc.data['_aux', 'cost'].tolist()]
    time = [x[0] for x in mpc.data['_time'].tolist()]

    data = {
        'states': states,
        'inputs': inputs,
        'avg_traj': avg_trajectory,
        'cost': cost,
        'time': time,
        'ISE': 0,
        'ITAE': 0
    }

    return data


def export_results(data, type, name, mode):

    f = open('./data/' + type + '/' + name + '.json', mode)
    data_json = json.dumps(data)
    f.write(data_json)
    f.close()


def compute_performance_metrics(data, total_time, t_step, avg_period):

    avg_x = np.arange(0, total_time, avg_period)

    e3_bar = np.array([(x - 750) / 750 for x in data['avg_traj']['LacI']])
    e4_bar = np.array([(x - 300) / 300 for x in data['avg_traj']['TetR']])

    fe3_bar = interpolate.interp1d(avg_x, e3_bar, fill_value="extrapolate")
    e3_bar = fe3_bar(np.arange(0, total_time, t_step))
    fe4_bar = interpolate.interp1d(avg_x, e4_bar, fill_value="extrapolate")
    e4_bar = fe4_bar(np.arange(0, total_time, t_step))

    e_bar = np.array([np.linalg.norm([e3_bar[i], e4_bar[i]])
                      for i in range(len(e3_bar))])

    ISE = np.sum((e_bar)**2)
    ITAE = np.sum([np.abs(e_bar[i])*(t_step*(i+1)) for i in range(len(e_bar))])

    data['ISE'] = ISE
    data['ITAE'] = ITAE

    return ISE, ITAE


def plot_results(data, total_time, avg_period):

    fig_x = 20
    fig_y = 10

    x_ticks = [x for x in range(0, total_time + 1, int(total_time/6))]
    x_ticks_label = [int(x/60) for x in x_ticks]
    avg_x = np.arange(0, total_time, avg_period)

    plt.rcParams['axes.grid'] = True
    plt.rcParams['font.size'] = 14

    # -------------------- Proteins -------------------- #
    # --- LacI --- #
    figure_proteins, axes = plt.subplots(
        2, sharex=True, figsize=(fig_x, fig_y))

    axes[0].set_ylabel('')
    axes[0].set_title('LacI')

    line_LacI, = axes[0].plot(data['time'], data['states']['LacI'], color='b')
    line_avg_LacI, = axes[0].plot(
        avg_x, data['avg_traj']['LacI'], color='b', linestyle='--')
    line_ref_LacI, = axes[0].plot(
        data['time'], 750*np.ones(len(data['time'])), color='k', linestyle='--')
    axes[0].legend(['LacI', 'Avg LacI traj', 'LacI target'], loc='upper right')

    # --- TetR --- #
    axes[1].set_ylabel('')
    axes[1].set_title('TetR')
    line_TetR, = axes[1].plot(data['time'], data['states']['TetR'], color='m')
    line_avg_TetR, = axes[1].plot(avg_x, data['avg_traj']
                                  ['TetR'], color='m', linestyle='--')
    line_ref_TetR, = axes[1].plot(
        data['time'], 300*np.ones(len(data['time'])), color='k', linestyle='--')
    axes[1].legend(['TetR', 'Avg TetR traj', 'TetR target'], loc='upper right')
    axes[1].set_xlabel('time [min]')
    axes[1].set_xticks(x_ticks)
    axes[1].set_xticklabels(x_ticks_label, rotation=30)

    figure_proteins.set_facecolor("white")

    # -------------------- mRNAs -------------------- #
    # --- mRNA LacI --- #
    figure_mRNAs, axes1 = plt.subplots(2, sharex=True, figsize=(fig_x, fig_y))
    axes1[0].set_ylabel('')
    axes1[0].set_title('mRNA_LacI')
    line_mRNA_LacI, = axes1[0].plot(data['time'], data['states']
                                    ['mRNA_LacI'], color='b')
    axes1[0].legend(['mRNA LacI'], loc='upper right')

    # --- mRNA TetR --- #
    axes1[1].set_ylabel('')
    axes1[1].set_title('mRNA_TetR')
    line_mRNA_TetR, = axes1[1].plot(data['time'], data['states']
                                    ['mRNA_TetR'], color='m')
    axes1[1].legend(['mRNA TetR'], loc='upper right')
    axes1[1].set_xlabel('time [min]')
    axes1[1].set_xticks(x_ticks)
    axes1[1].set_xticklabels(x_ticks_label, rotation=30)

    figure_mRNAs.set_facecolor("white")

    # -------------------- Internal inducers concentrations -------------------- #
    # --- v1 --- #
    figure_int_inducers, axes2 = plt.subplots(
        2, sharex=True, figsize=(fig_x, fig_y))
    axes2[0].set_ylabel('')
    axes2[0].set_title('$v1$')
    line_int_aTc, = axes2[0].plot(
        data['time'], data['states']['v1'], color='y')
    axes2[0].legend(['$v1$'], loc='upper right')

    # --- v2 --- #
    axes2[1].set_ylabel('')
    axes2[1].set_title('$v2$')
    line_int_IPTG, = axes2[1].plot(
        data['time'], data['states']['v2'], color='r')
    axes2[1].legend(['$v2$'], loc='upper right')
    axes2[1].set_xlabel('time [min]')
    axes2[1].set_xticks(x_ticks)
    axes2[1].set_xticklabels(x_ticks_label, rotation=30)

    figure_int_inducers.set_facecolor("white")

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
    axes2[1].set_xticks(x_ticks)
    axes2[1].set_xticklabels(x_ticks_label, rotation=30)

    figure_inducers.set_facecolor("white")

    # -------------------- Cost -------------------- #
    figure_cost, axes4 = plt.subplots(1, figsize=(fig_x, fig_y/2))
    axes4.set_ylabel('')
    axes4.set_title('Cost')
    line_cost, = axes4.plot(data['time'], data['cost'], color='g')
    axes4.legend(['Cost'], loc='upper right')
    axes4.set_xlabel('time [min]')
    axes4.set_xticks(x_ticks)
    axes4.set_xticklabels(x_ticks_label, rotation=30)

    figure_cost.set_facecolor("white")

    figures = np.array([figure_proteins, figure_mRNAs,
                       figure_int_inducers, figure_inducers, figure_cost])
    lines = np.array([line_LacI, line_avg_LacI, line_ref_LacI, line_TetR, line_avg_TetR, line_ref_TetR,
                     line_mRNA_LacI, line_mRNA_TetR, line_int_aTc, line_int_IPTG, line_aTc, line_IPTG, line_cost])

    return figures, lines


def update_mRNA(i, data, lines):
    lines[0].set_data(data['time'][:i], data['states']['mRNA_LacI'][:i])
    lines[1].set_data(data['time'][:i], data['states']['mRNA_TetR'][:i])
    return lines


def update_protein(i, data, lines, avg_x):
    lines[0].set_data(data['time'][:i], data['states']['LacI'][:i])
    lines[1].set_data(avg_x[:i], data['avg_traj']['LacI'][:i])
    lines[2].set_data(data['time'][:i], 750*np.ones(len(data['time']))[:i])
    lines[3].set_data(data['time'][:i], data['states']['TetR'][:i])
    lines[4].set_data(avg_x[:i], data['avg_traj']['TetR'][:i])
    lines[5].set_data(data['time'][:i], 300*np.ones(len(data['time']))[:i])
    return lines


def update_internal_inducers(i, data, lines):
    lines[0].set_data(data['time'][:i], data['states']['v1'][:i])
    lines[1].set_data(data['time'][:i], data['states']['v2'][:i])
    return lines


def update_external_inducers(i, data, lines):
    lines[0].set_data(data['time'][:i], data['inputs']['aTc'][:i])
    lines[1].set_data(data['time'][:i], data['inputs']['IPTG'][:i])
    return lines


def update_cost(i, data, lines):
    lines[0].set_data(data['time'][:i], data['cost'][:i])
    return lines


def animate_results(type, name, funct, figures, fargs, steps):
    anim = animation.FuncAnimation(figures, funct, fargs=fargs, frames=steps)
    anim.save('./simulations/' + type + '/' + name + '.mp4', fps=60)
