import numpy as np
from simulate import simulate_SES
import pyDOE as doe
import matplotlib.pyplot as plt
# from eq_basins import plot_equilibrium_basins

'''
# Model Parameters
# r = 0.1  # resource regeneration rate, [0.015, 0.105]
r = 10 # resource regeneration rate, normalized
a = 1000 # revenue threshold, [0, 2000]
b1 = 1 # resource share in production function, []
b2 = 1 # labor share in production function, []
c = 0.05 # extraction cost,  [0, 0.5]
d = 100 # fixed cost parameter in profit function, [95, 225]
k = 2 # benefit of resource access in exponential parameter, []
p = 3 #profit, out-of-system payoff for domestic users, [2.7, 18]
h = 0.06  # parameter modulating rate at which wage increases with marginal benefit of additional labor, [0,120]
g = 0.01 # parameter modulating rate at which wage decreases with excess labor, [1, 0.02]
m = 0.8 # responsiveness of domestic users to payoff difference, [0.3, 12.3]

q = -0.5 #substituttion parameter to reflect limited substitutability between resource and labor, do not vary

# State Variables (definite inital state)
self.W_0 = W_0 # zero
self.W_min = W_min # wage
self.U_0 = U_0 # inital population
self.dt = dt
self.s = r/R_max # resource access index?
R_0 inital amount of resource
R_max #total resource amount, defined in production function

# NOTE: Resource state normalized to have max of 100 (not 1, so r = 0.1 in table but 10 in model)
# Quesetions: Do I need to run in the cluster? Am I using with policy?

'''
# CHECK OUT calc_alaignment.py

# Plot timeseries for a single run
# Set resource parameters
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

# Step size
dt = 0.08

R_0 = 100
U_0 = 10
W_0 = 4
W_min = 0

num_points = 20 #originally 80

def alter_parameter(r, R_max, a, b1, b2, q, c, d, k, p, h, g, m,
                          W_min, dt, R_0, W_0, U_0, adjustment=0, param='none'): # alter ['r','c','d','g','h','m','p']

  if 'r' in param :
    r *= adjustment
  if 'c' in param :
    c *= adjustment
  if 'd' in param :
    d *= adjustment
  if 'g' in param :
    g *= adjustment
  if 'h' in param :
    h *= adjustment
  if param == 'none':
    print('no parameter altered')

  # Run model
  ts = simulate_SES(r, R_max, a, b1, b2, q, c, d, k, p, h, g, m,
                          W_min, dt, R_0, W_0, U_0)
  R, E, U, S, W, P, L, convergence = ts.run()
  R = np.array(R)

  # set initial conditions to loop over
  np.random.seed(1)
  initial_points = doe.lhs(3, samples = num_points)
  sust_eq = 0
  # initialize matrix recording whether initial point leads to good or bad eq
  eq_condition = np.zeros(len(initial_points))
  R_array = np.zeros(num_points)
  U_array = np.zeros(num_points)
  resilience = np.zeros(num_points)
  equity = np.zeros(num_points)

  for n, point in enumerate(initial_points): 

    if n % 10 == 0:
      print(n)

    R_0 = point[0]
    U_0 = point[1]
    W_0 = point[2]

    ts = simulate_SES(r, R_max, a, b1, b2, q, c, d, k, p, h, g, m, # No policy
                        W_min, dt, R_0, W_0, U_0)
    R, E, U, S, W, P, L, convergence = ts.run() # this is an array

    if R[-1] > 90 and U[-1] < 1:
      eq_condition[n] = 0
    else:
      eq_condition[n] = 1
      sust_eq += 1

    U_array[n] = U[-1] # population (pseudo for resilience)
    R_array[n] = R[-1]

    resilience[n] = sust_eq / len(initial_points) # proportion of states that lead to non-collapse equilibrium
    equity[n] = np.mean(U_array) #total well being  #(previously resilience)

    # print(resilience, equity)
  return resilience, equity

resilience, equity = alter_parameter(r, R_max, a, b1, b2, q, c, d, k, p, h, g, m,
                          W_min, dt, R_0, W_0, U_0, adjustment=1.1, param = 'none')
resilience_r, equity_r = alter_parameter(r, R_max, a, b1, b2, q, c, d, k, p, h, g, m,
                          W_min, dt, R_0, W_0, U_0, adjustment=1.1, param = 'r')

plt.subplot(2,1,1)
plt.plot(np.arange(num_points)*dt,resilience, label='No alteration', color='#a9aa83', linestyle='', markersize=12)
plt.plot(np.arange(num_points)*dt,resilience_r, label='r altered', color='#7494aa', linestyle='', markersize=12)
# plt.plot(np.arange(num_points)*dt,resilience_r, label='r altered', color='#374b5f', linestyle='', markersize=12) #887e87 #bda27b #a47c6b
plt.legend()

plt.subplot(2,1,2)
plt.plot(np.arange(num_points)*dt,equity,label='No alteration', color='#a9aa83', linestyle='', markersize=12)
plt.plot(np.arange(num_points)*dt,equity_r,label='r altered', color='#7494aa', linestyle='', markersize=12 )
plt.legend()
plt.show()
exit()

def plot_trajectories(R, U):
  n = len(R)
  T = n*dt
  fig, axarr = plt.subplots(2)
  axarr[0].plot(np.arange(n)*dt, R, color = 'k')
  axarr[0].set_ylim([0, R_max])
  axarr[0].set_title('Resource')
  axarr[1].plot(np.arange(n)/(n/T), U, color = 'k')
  axarr[1].set_title('Population')
  axarr[1].set_ylim([0, 30])
  plt.tight_layout()
  patches = [mlines.Line2D([], [], color = 'k', linestyle = '-', linewidth=1, label='No Policy'),
             mlines.Line2D([], [], color = 'c', linestyle = '-', linewidth=1, label='Under a Fine')]
  fig.legend(handles=patches, bbox_to_anchor=(0.99, 1), borderaxespad=0. )

plot_trajectories(R,U)

exit()

r_range = [1.5, 10.5]
c_range = [0, 0.05]
d_range = [95, 225]
g_range = [0, 0.02]
h_range = [0, 120]
m_range = [0.3, 12.3]
p_range = [2.7,18]
#k_range = 
#a_range = 
#b1_range = [0.05, 1.5] 
#b2_range = [0.05, 1.5]

k = 2 # water access parameter (how steeply benefit of water rises)
a = 1000  # profit cap
b1 = 1 # profit vs. water parameter, or = 0.05
b2 = 1 # profit vs. labor parameter, or = 0.05

# Run SES Model for each parameter set

#State Variables
R_0 = 100
U_0 = 10
W_0 = 4
W_min = 0
R_max = 100  # aquifer capacity
q = -0.5 #substituttion parameter to reflect limited substitutability between resource and labor
dt = 0.08

# set policy
fine = 130
fine_cap = 5

def SES_model(x):

  r,c,d,g,h,m,p = list(x)

  # set initial conditions to loop over
  np.random.seed(1)
  num_points = 80 #Was 80
  initial_points = doe.lhs(3, samples = num_points)

  # Scale points ([R, U, W])
  initial_points[:,0] = initial_points[:,0] * 100
  initial_points[:,1] = initial_points[:,1] * 45
  initial_points[:,2] = initial_points[:,2] * 20

  sust_eq = 0
  # initialize matrix recording whether initial point leads to good or bad eq
  eq_condition = np.zeros(len(initial_points))

  for n, point in enumerate(initial_points): 
    R_0 = point[0]
    U_0 = point[1]
    W_0 = point[2]

    # pp = simulate_SES(r, R_max, a, b1, b2, q, c, d, k, p, h, g, m, # With policy
    #                 W_min, dt, R_0, W_0, U_0, fine_cap, fine)
    # R, E, U, S, W, P, L, converged = pp.run()

    ts = simulate_SES(r, R_max, a, b1, b2, q, c, d, k, p, h, g, m, # No policy
                        W_min, dt, R_0, W_0, U_0)
    R, E, U, S, W, P, L, convergence = ts.run() # this is an array


    if R[-1] > 90 and U[-1] < 1:
      eq_condition[n] = 0
    else:
      eq_condition[n] = 1
      sust_eq += 1

    U_array[n] = U[-1] #np.mean(U) # population (pseudo for resilience)

  resilience = sust_eq / len(initial_points) # proportion of states that lead to non-collapse equilibrium
  equity = np.mean(U_array) #total well being  #(previously resilience)

  return resilience, equity

Y_resilience = np.zeros(N)
Y_equity = np.zeros(N)

# Run model
for i in range (N):
  if i % 10 == 0:
    print(i)
  Y_resilience[i], Y_equity[i] = SES_model(param_values[i])
