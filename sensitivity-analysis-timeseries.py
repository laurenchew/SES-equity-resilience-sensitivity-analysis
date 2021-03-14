import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

import sys
sys.path.append('../')
from simulate import simulate_SES

# Plot trajectories for several different initial conditions

# Plot timeseries for a single run
#set resource parameters
r = 10
R_max = 100  # aquifer capacity

# Set industrial payoff parameters
a = 1000  # profit cap
#b1 = 0.05 # profit vs. water parameter
#b2 = 0.05 # profit vs. labor parameter
b1 = 1
b2 = 1
q = -0.5
c = 0.05 # extraction cost parameter
d = 100

# Set domestic user parameters
k = 2 # water access parameter (how steeply benefit of water rises)
p = 3 # out-of-system wellbeing/payoff
h = 0.06 # rate at which wage declines with excess labor
g = 0.01 # rate at which wage increases with marginal benefit of more labor
m = 0.8 # responsiveness of population to wellbeing (h*p > 1 with dt=1)
W_min = 0

# Step size
dt = 0.08

#Define initial conditions
R_0 = 100
U_0 = 10
W_0 = 4

colors = ['#D98FD4', '#649EDB', '#C1D4D9','#92A4A6','#FFDA1A','#F09975', '#E64B1E']
# colors = ['#FCC872', '#DB9F67', '#F0A17D','#FF9698', '#DB7967'] #30B1DB #92A4A6 #BF8069

# plt.rcParams["font.size"] = "12"
size = 14
params = {'legend.fontsize': size,
          'figure.figsize': (11,6),
          'axes.labelsize': size,
          'axes.titlesize': size,
          'xtick.labelsize': size,#*0.75,
          'ytick.labelsize': size,#*0.75,
          'axes.titlepad': 25}
plt.rcParams.update(params)


## Alter parameter
def plot_parameter_varaiations(r=10, R_max=100, a=1000, b1=1, b2=1, q=-0.5, c=0.05, d=100, k=2, p=3, h=0.06, g=0.01, m=0.8,
                                    W_min=0, dt=0.08, R_0=100, W_0=4, U_0=10, parameter='none'):
    
    #Plot unaltered
    ts = simulate_SES(r, R_max, a, b1, b2, q, c, d, k, p, h, g, m,
                        W_min, dt, R_0, W_0, U_0)
    R, E, U, S, W, P, L, convergence = ts.run()
    R = np.array(R)

    n = len(R)
    T = n*dt

    fig, axarr = plt.subplots(2)

    axarr[0].plot(np.arange(n)*dt, R, color = 'k', marker='', linewidth=2 )
    axarr[0].set_ylim([0, R_max])
    axarr[0].set_title('Resource')

    axarr[1].plot(np.arange(n)/(n/T), U, color = 'k', marker='', linewidth=2)
    axarr[1].set_title('Population')
    axarr[1].set_ylim([0, 30])
    axarr[1].set_xlabel('Time')
    plt.tight_layout()
    patches = [mlines.Line2D([], [], color = 'k', linestyle = '-', marker='', linewidth=2, label='Unaltered')]

    #Plot altered parameter

    if parameter == 'r': # resource regeneration rate
        for i, r in enumerate([5,6,7,8,9]): #10 is original

            ts = simulate_SES(r, R_max, a, b1, b2, q, c, d, k, p, h, g, m,
                                    W_min, dt, R_0, W_0, U_0)
            R, E, U, S, W, P, L, convergence = ts.run()

            n = len(R)

            axarr[0].plot(np.arange(n)/(n/T), R[:n], color =colors[i], marker='',linewidth=2 )
            axarr[1].plot(np.arange(n)/(n/T), U[:n], color =colors[i], marker='',linewidth=2 )
            #eff_W = W[1:]*(np.array(L)/np.array(U[1:]))
            #axarr[2].plot(np.arange(n-1)/(n/T), eff_W, color =colors[i], marker='',linewidth=2 )

            patches.append(mlines.Line2D([], [], color =colors[i], linestyle = '-', marker='',linewidth=2,  label=f'{parameter} = {r}'))

        fig.legend(handles=patches, bbox_to_anchor=(0.99, 1), borderaxespad=0. )
        plt.savefig(f'figures/trajectories_{parameter}_altered.png')
    if parameter == 'c': # extraction cost
        for i, c in enumerate([0, 0.01, 0.02, 0.03, 0.04]): #0.05 is original

            ts = simulate_SES(r, R_max, a, b1, b2, q, c, d, k, p, h, g, m,
                                    W_min, dt, R_0, W_0, U_0)
            R, E, U, S, W, P, L, convergence = ts.run()

            n = len(R)

            axarr[0].plot(np.arange(n)/(n/T), R[:n], color =colors[i], marker='',linewidth=2 )
            axarr[1].plot(np.arange(n)/(n/T), U[:n], color =colors[i], marker='',linewidth=2 )
            #eff_W = W[1:]*(np.array(L)/np.array(U[1:]))
            #axarr[2].plot(np.arange(n-1)/(n/T), eff_W, color =colors[i], marker='',linewidth=2 )

            patches.append(mlines.Line2D([], [], color =colors[i], linestyle = '-', marker='',linewidth=2,  label=f'{parameter} = {c}'))

        fig.legend(handles=patches, bbox_to_anchor=(0.99, 1), borderaxespad=0. )
        plt.savefig(f'figures/trajectories_{parameter}_altered.png')

    if parameter == 'd': # fixed cost parameter in profit function
        for i, d in enumerate([95, 105, 120, 140, 160, 175]): #100 is original

            ts = simulate_SES(r, R_max, a, b1, b2, q, c, d, k, p, h, g, m,
                                    W_min, dt, R_0, W_0, U_0)
            R, E, U, S, W, P, L, convergence = ts.run()

            n = len(R)

            axarr[0].plot(np.arange(n)/(n/T), R[:n], color =colors[i], marker='',linewidth=2 )
            axarr[1].plot(np.arange(n)/(n/T), U[:n], color =colors[i], marker='',linewidth=2 )
            #eff_W = W[1:]*(np.array(L)/np.array(U[1:]))
            #axarr[2].plot(np.arange(n-1)/(n/T), eff_W, color =colors[i], marker='',linewidth=2 )

            patches.append(mlines.Line2D([], [], color =colors[i], linestyle = '-', marker='',linewidth=2,  label=f'{parameter} = {d}'))

        fig.legend(handles=patches, bbox_to_anchor=(0.99, 1), borderaxespad=0. )
        plt.savefig(f'figures/trajectories_{parameter}_altered.png')

    if parameter == 'g': # parameter modulating rate at which wage decreases with excess labor
        for i, g in enumerate([0, 0.005, 0.0075, 0.0125, 0.015]): #0.01 is original, [0, 0.02]

            ts = simulate_SES(r, R_max, a, b1, b2, q, c, d, k, p, h, g, m,
                                    W_min, dt, R_0, W_0, U_0)
            R, E, U, S, W, P, L, convergence = ts.run()

            n = len(R)

            axarr[0].plot(np.arange(n)/(n/T), R[:n], color =colors[i], marker='',linewidth=2 )
            axarr[1].plot(np.arange(n)/(n/T), U[:n], color =colors[i], marker='',linewidth=2 )
            #eff_W = W[1:]*(np.array(L)/np.array(U[1:]))
            #axarr[2].plot(np.arange(n-1)/(n/T), eff_W, color =colors[i], marker='',linewidth=2 )

            patches.append(mlines.Line2D([], [], color =colors[i], linestyle = '-', marker='',linewidth=2,  label=f'{parameter} = {g}'))

        fig.legend(handles=patches, bbox_to_anchor=(0.99, 1), borderaxespad=0. )
        plt.savefig(f'figures/trajectories_{parameter}_altered.png')

    if parameter == 'h': # parameter modulating rate at which wage increases with marginal benefit of additional labor
        for i, h in enumerate([0, 0.03, 0.09, 0.12]): #0.06 is original, [0,120]

            ts = simulate_SES(r, R_max, a, b1, b2, q, c, d, k, p, h, g, m,
                                    W_min, dt, R_0, W_0, U_0)
            R, E, U, S, W, P, L, convergence = ts.run()

            n = len(R)

            axarr[0].plot(np.arange(n)/(n/T), R[:n], color =colors[i], marker='',linewidth=2 )
            axarr[1].plot(np.arange(n)/(n/T), U[:n], color =colors[i], marker='',linewidth=2 )
            #eff_W = W[1:]*(np.array(L)/np.array(U[1:]))
            #axarr[2].plot(np.arange(n-1)/(n/T), eff_W, color =colors[i], marker='',linewidth=2 )

            patches.append(mlines.Line2D([], [], color =colors[i], linestyle = '-', marker='',linewidth=2,  label=f'{parameter} = {h}'))

        fig.legend(handles=patches, bbox_to_anchor=(0.99, 1), borderaxespad=0. )
        plt.savefig(f'figures/trajectories_{parameter}_altered.png')

    if parameter == 'm': # responsiveness of domestic users to payoff difference
        for i, m in enumerate([0.3, 0.4, 0.5, 0.6,0.7,  0.9, 1]): #0.8 is original, [0.3, 12.3]

            ts = simulate_SES(r, R_max, a, b1, b2, q, c, d, k, p, h, g, m,
                                    W_min, dt, R_0, W_0, U_0)
            R, E, U, S, W, P, L, convergence = ts.run()

            n = len(R)

            axarr[0].plot(np.arange(n)/(n/T), R[:n], color =colors[i], marker='',linewidth=2 )
            axarr[1].plot(np.arange(n)/(n/T), U[:n], color =colors[i], marker='',linewidth=2 )
            #eff_W = W[1:]*(np.array(L)/np.array(U[1:]))
            #axarr[2].plot(np.arange(n-1)/(n/T), eff_W, color =colors[i], marker='',linewidth=2 )

            patches.append(mlines.Line2D([], [], color =colors[i], linestyle = '-', marker='',linewidth=2,  label=f'{parameter} = {m}'))

        fig.legend(handles=patches, bbox_to_anchor=(0.99, 1), borderaxespad=0. )
        plt.savefig(f'figures/trajectories_{parameter}_altered.png')
    
    if parameter == 'p': #profit
        for i, p in enumerate([2.7, 2.8, 2.9, 3.1, 3.2, 3.3]): #3 is original, [2.7, 18]

            ts = simulate_SES(r, R_max, a, b1, b2, q, c, d, k, p, h, g, m,
                                    W_min, dt, R_0, W_0, U_0)
            R, E, U, S, W, P, L, convergence = ts.run()

            n = len(R)

            axarr[0].plot(np.arange(n)/(n/T), R[:n], color =colors[i], marker='',linewidth=2 )
            axarr[1].plot(np.arange(n)/(n/T), U[:n], color =colors[i], marker='',linewidth=2 )
            #eff_W = W[1:]*(np.array(L)/np.array(U[1:]))
            #axarr[2].plot(np.arange(n-1)/(n/T), eff_W, color =colors[i], marker='',linewidth=2 )

            patches.append(mlines.Line2D([], [], color =colors[i], linestyle = '-', marker='', linewidth=2,  label=f'{parameter} = {p}'))

        fig.legend(handles=patches, bbox_to_anchor=(0.99, 1), borderaxespad=0. )
        plt.savefig(f'figures/trajectories_{parameter}_altered.png')


# Run function for every parameter
for p in  ['r','c','d','g','h','m','p']:
    plot_parameter_varaiations(parameter = p)
