import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.signal import argrelextrema
from scipy.optimize import curve_fit
import itertools
import sys
spinner = itertools.cycle(['-', '/', '|', '\\'])

def tick():
    sys.stdout.write(next(spinner))  # write the next character
    sys.stdout.flush()               # flush stdout buffer (actual character display)
    sys.stdout.write('\b')           # erase the last written char

def loadFrameGrab(path):
    points = []
    with open(path) as f:
        next(f)
        for line in f:
            points.append([float(x) for x in line.replace(',', '').split()])
    return np.array(points)

def lindbladian(t, rho_lis):
    global lindblad_ops
    lindblad_operators = lindblad_ops.copy()
    ### Lab frame
    #if driven == False:
    #    H = np.array([[E0, 0], [0, E1]]) + sigmaZ * Omega_R/2
    #else:
    #    H = np.array([[-omega/2, Omega_R*np.cos(omega*t)], [Omega_R*np.cos(omega*t), omega/2]])
    ### Rotating Wave Approx. (RWA)
    if driven:
        H = np.array([[0, Omega_R/2], [Omega_R/2, 0]], dtype=complex)
    else:
        H = np.array([[0, 0], [0, 0]], dtype = complex)
    rho = np.array([[rho_lis[0], rho_lis[1]], [rho_lis[2], rho_lis[3]]], dtype=complex)
    drho = -1j * (H @ rho - rho @ H)
    for L in lindblad_operators:
        Ld = L.conj().T
        drho += L @ rho @ Ld - 0.5 * (Ld @ L @ rho + rho @ Ld @ L)
    tick()
    return [drho[0][0], drho[0][1], drho[1][0], drho[1][1]]

def exponetial_decay(x, a, b, c):
    return a*np.exp(-2*x/b)+c

def cal_relax_times(solt, soly):
    global gammaP, gammaM, gammaZ
    max_idx = []
    idx_lengths = []
    decay_vals = []
    for i in range(4):
        x_idx = argrelextrema(soly[i], np.greater, order = 80)[0]
        idx_lengths.append(len(x_idx))
        max_idx.append(x_idx)
        plt.scatter(solt[x_idx], soly[i][x_idx])
        popt, _ = curve_fit(exponetial_decay, solt[x_idx], soly[i][x_idx])
        decay_vals.append(popt)
        plt.plot(solt, exponetial_decay(solt, *popt))
    plt.clf()
    for k in decay_vals:
        print(1/(k[1]))

    T1 = 1/(2*(gammaP+gammaM))
    T2 = 1/(2*(gammaZ))
    print('T1', T1)
    print('T2', T2)
    print('T2*', 1/(2*T1) + 1/T2)

def coloring(arr_like):
    import matplotlib.colors
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["red","blue"])
    color_nr = np.linspace(0, 1, len(arr_like))
    colors = []
    for i in color_nr:
        colors.append(cmap(i))
    return np.array(colors)

if False: #__name__ == '__main__':
    # Initial ground state |0><0| ## E.g. np.array([[0.9, 0.2*(1+1j)], [0.2*(1-1j), 0.1]]) + 0j
    rho0 = [1.0, 0*0.2*(1+1j), 0*0.2*(1-1j), 0.0]
    t_span = (0, 5)
    omega = 5#*2*np.pi 
    driven = True
    Omega_R = 2 * omega
    gammaP = 1.0
    gammaM = 0.0 # 0.01
    gammaZ = 0.1 #0.01

    """
    rho0 = [0.9, 0.2*(1+1j), 0.2*(1-1j), 0.1]
    t_span = (0, 30)
    omega = 2*np.pi * 3e3 #GHz
    driven = True
    Omega_R = 2*np.pi / 3.38 #MHz
    gammaP = 0.01*0
    gammaM = 0.01*0
    gammaZ = 0.01*0
    
    rho0 = [0.9, 0.2*(1+1j), 0.2*(1-1j), 0.1]
    t_span = (0, 30)
    omega = 2*np.pi * 3e3 #GHz
    driven = True
    Omega_R = 2*np.pi / 3.38 #MHz
    gammaP = 0.01*0
    gammaM = 0.0045*Omega_R
    gammaZ = 0.01*0
    """
    
    sigmaP = np.array([[0, 0], [1, 0]])
    sigmaM = np.array([[0, 1], [0, 0]])
    sigmaZ = np.array([[1, 0], [0, -1]])
    E1 = omega
    E0 = 0
    L1 = np.sqrt(2*gammaZ) * sigmaZ
    L2 = np.sqrt(2*gammaP) * sigmaP
    L3 = np.sqrt(2*gammaM) * sigmaM
    lindblad_ops = [L1, L2, L3]
    sol = solve_ivp(lindbladian, t_span, rho0, t_eval=np.linspace(t_span[0], t_span[1], 2000), 
                    rtol=1e-5, atol=1e-6, max_step=0.5)
    
    #cal_relax_times(sol.t, sol.y)
    
    plt.plot(sol.t, sol.y[0], label=r'$\rho_{00}$')
    plt.plot(sol.t, sol.y[1].real, label=r'$\rho_{01}$')
    plt.plot(sol.t, sol.y[2].real, label=r'$\rho_{10}$')
    plt.plot(sol.t, sol.y[3], label=r'$\rho_{11}$')
    plt.xlabel(r'$2\tau$ Time [ns]')
    plt.ylabel('Density matrix')
    plt.legend()
    plt.grid()
    plt.show()
    



rho0 = np.array([1.0, 0.0, 0.0, 0.0], dtype= complex)
omega = 2*np.pi * 9.59e3 # GHz
t_span = (0, 8)
Omega_R = 8.21 * 2*np.pi # MHz
#print(omega/Omega_R, Omega_R)

sigmaP = np.array([[0, 0], [1, 0]], dtype=complex)
sigmaM = np.array([[0, 1], [0, 0]], dtype=complex)
sigmaZ = np.array([[1, 0], [0, -1]], dtype=complex)

t_pi2 = np.pi / (2 * Omega_R)
t_pi = np.pi / Omega_R
#print(t_pi2*Omega_R)

def hahn_echo(tau, gamma_z_guess, gamma_p_guess, gamma_m_guess):
        global driven, lindblad_ops
        lindblad_ops = [np.sqrt(2*gamma_z_guess) * sigmaZ, np.sqrt(2*gamma_p_guess) * sigmaP, np.sqrt(2*gamma_m_guess) * sigmaM]
        # pi/2 pulse
        driven = True
        sol1 = solve_ivp(lindbladian, [0, t_pi2], rho0)
        # free precession
        driven = False
        sol2 = solve_ivp(lindbladian, [0, tau], sol1.y[:, -1])
        # pi-pulse
        driven = True
        sol3 = solve_ivp(lindbladian, [0, t_pi], sol2.y[:, -1])
        # free precession
        driven = False
        sol4 = solve_ivp(lindbladian, [0, tau], sol3.y[:, -1])
        return np.abs(sol4.y[1][-1])

if True:
    tau_values = np.linspace(0.005, 4.0, 75)
    tau_fine = np.linspace(0.005, 4.0, 1000)

    gammaZ_fit = 0.343
    gammaP = 0.002604
    gammaM = 0.002604
    print('T2-fit',1/(gammaP+gammaM+4*gammaZ_fit))
    print('T1-test',1/(2*gammaP+2*gammaM))

    T1_paper = 96 #microseconds
    gammaPM = 1 / (4.0 * T1_paper)
    gammaP, gammaM = gammaPM, gammaPM


    # gammaZ_scan = np.array([0.05, 0.1, 0.2, 0.25, 0.3, 0.4, 0.6, 0.8, 1.0, 1.5])
    gammaZ_scan = np.array([0.1, 0.2, 0.25, 0.3, 0.4, 0.6, 0.8])
    colors = coloring(gammaZ_scan)

    for nr_color, gammaZ in enumerate(gammaZ_scan):
        echo_amplitudes = np.array([hahn_echo(t_delay, gammaZ, gammaP, gammaM) for t_delay in tau_values])
        popt, _ = curve_fit(exponetial_decay, tau_values, echo_amplitudes, p0=[0.5, 0.8, 0])
        plt.plot(tau_values*2, echo_amplitudes, 'o', ms=3, color = colors[nr_color])
        plt.plot(tau_fine*2, exponetial_decay(tau_fine, *popt), label=r'$\gamma_z$='+f'{gammaZ:.2f} MHz,' + r'$T_2$=' + f'{popt[1]:.2f}'+r' $\mu$s', color = colors[nr_color])
        print(f'gammaZ = {gammaZ} MHz produces: T2 = {popt[1]}')


    paper_hahn = loadFrameGrab('Hahn_echo_fram_grab.csv')
    popt, _ = curve_fit(exponetial_decay, paper_hahn[:,0], paper_hahn[:,1], p0=[0.5, 0.8, 0])
    plt.plot(paper_hahn[:,0], paper_hahn[:,1], 'o', ms=3, color = 'black')
    plt.plot(tau_fine, exponetial_decay(tau_fine, *popt),  color = 'black', label='Exp. -- Jiao et. al')


    #plt.figure(figsize=(8, 5))
    #plt.plot(tau_values, echo_amplitudes, label="Sim. echo decay", color = 'blue')
    #plt.plot(tau_values, exponetial_decay(tau_values, *popt), color = 'red', label='Sim. decay function')
    plt.xlabel(r'Total delay time $\tau$ ($\mu$s)')
    plt.ylabel('Echo amplitude [a.u.]')
    plt.legend()
    plt.grid()
    plt.xlim(0, 8)
    plt.ylim(0, 0.5)
    plt.savefig('Hahn_echo.pdf')
    plt.show()


    """
    plt.plot(paper_hahn[:,0], paper_hahn[:,1], 'o', ms=3, color = 'black')
    plt.plot(tau_fine, exponetial_decay(tau_fine, *popt),  color = 'black', label='Exp. -- Jiao et. al')

    gammaZ_scan = np.array([0.32, 0.33, 0.34])
    colors = coloring(gammaZ_scan)
    for nr_color, gammaZ in enumerate(gammaZ_scan):
        echo_amplitudes = np.array([hahn_echo(t_delay, gammaZ, gammaP, gammaM) for t_delay in tau_values])
        popt, _ = curve_fit(exponetial_decay, tau_values, echo_amplitudes, p0=[0.5, 0.8, 0])
        plt.plot(tau_values*2, echo_amplitudes, 'o', ms=3, color = colors[nr_color])
        plt.plot(tau_fine*2, exponetial_decay(tau_fine, *popt), label=r'$\gamma_z$='+f'{gammaZ:.2f}, T2={popt[1]:.2f}'+r' $\mu$s', color = colors[nr_color])
        print(f'gammaZ = {gammaZ} MHz produces: T2 = {popt[1]}')
    plt.xlabel(r'Total delay time $\tau$ ($\mu$s)')
    plt.ylabel('Echo amplitude [a.u.]')
    plt.legend()
    plt.grid()
    plt.xlim(0, 8)
    plt.ylim(0, 0.5)
    plt.show()
    """


def inversion_recovery(tau, gamma_z, gamma_p_guess, gamma_m_guess):
    global driven, lindblad_ops
    lindblad_ops = [np.sqrt(2*gamma_z) * sigmaZ, np.sqrt(2*gamma_p_guess) * sigmaP, np.sqrt(2*gamma_m_guess) * sigmaM]
    # pi pulse ---- to invert the population, i.e. switch rho00 and rho11
    driven = True
    sol1 = solve_ivp(lindbladian, [0, t_pi], rho0)
    # free precession
    driven = False
    sol2 = solve_ivp(lindbladian, [0, tau], sol1.y[:, -1])
    # This is the sequence, but to meausre it a Hahn echo with a fixed delay is added:
    
    # pi/2 pulse
    driven = True
    sol3 = solve_ivp(lindbladian, [0, t_pi2], sol2.y[:, -1])
    # free precession
    driven = False
    sol4 = solve_ivp(lindbladian, [0, 0.400], sol3.y[:, -1])
    # pi-pulse
    driven = True
    sol5 = solve_ivp(lindbladian, [0, t_pi], sol4.y[:, -1])
    # free precession
    driven = False
    sol6 = solve_ivp(lindbladian, [0, 0.400], sol5.y[:, -1])

    return np.abs(sol6.y[3][-1]) # rho11

def exponential_recovery(x, a, b, c):
    return a * (1 - 2*np.exp(-x/b)) + c


tau_values_T1 = np.linspace(0.0005, 2000, 75)
tau_fine_T1 = np.linspace(0.0005, 2000, 1000)
gammaZ_fit = 0.325

#gammaPM_scan = np.array([0.001, 0.0015, 0.0020, 0.0025, 0.0030, 0.0035, 0.0040, 0.0060, 0.0080, 0.0100])
gammaPM_scan = np.array([0.0015, 0.0020, 0.00265, 0.0030, 0.0040, 0.0050])
colors = coloring(gammaPM_scan)

for nr_color, gammaPM_guess in enumerate(gammaPM_scan):
    ir_signal = np.array([inversion_recovery(t, gammaZ_fit, gammaPM_guess, gammaPM_guess) for t in tau_values_T1])
    popt, _ = curve_fit(exponential_recovery, tau_values_T1, ir_signal, p0=[0.5, 96, 0])
    T1_fit = popt[1]
    plt.plot(tau_values_T1, ir_signal, 'o', ms=3, color = colors[nr_color])
    plt.plot(tau_fine_T1, exponential_recovery(tau_fine_T1, *popt), label=r'$\gamma_{\pm}$='+f'{gammaPM_guess:.5f} MHz,' + r'$T_1$='+ f'{popt[1]:.2f}'+r' $\mu$s', color = colors[nr_color])
    print(f'gammaPM = {gammaPM_guess} MHz produces: T1 = {T1_fit}')

paper_inversion = loadFrameGrab('Inversion_recovery_fram_grab.csv')
plt.plot(paper_inversion[:,0]/1.3752, paper_inversion[:,1], 'o', ms=3, color = 'black')
popt, _ = curve_fit(exponential_recovery, paper_inversion[:,0], paper_inversion[:,1], p0=[0.005, 96, 1.3])
plt.plot(tau_fine_T1, exponential_recovery(tau_fine_T1, *popt),  color = 'black', label='Exp. -- Jiao et. al')

#def streched(x, a, b, c):
#    return a*np.exp(-(x/b)**c)
#popt, _ = curve_fit(streched, paper_inversion[:,0], paper_inversion[:,1], p0=[0.05, 0.96, 0.5])
#plt.plot(tau_fine_T1, streched(tau_fine_T1, *popt),  color = 'blue', label='Exp. -- Jiao et. al')

plt.xlabel(r'Total delay time $T$ ($\mu$s)')
plt.ylabel('Echo amplitude [a.u.]')
plt.legend()
plt.grid()
plt.xlim(0, 1450)
plt.ylim(0.495, 0.5005)
plt.savefig('IF_seq.pdf')
plt.show()