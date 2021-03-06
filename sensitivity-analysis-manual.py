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
g = 0.01 # parameter modulating rate at which wage decreases with excess labor, [0, 0.02]
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

#State variables
R_0 = 100
U_0 = 10
W_0 = 4
W_min = 0

num_points = 20 #originally 80

size = 14
params = {'legend.fontsize': size,
          'figure.figsize': (11,6),
          'axes.labelsize': size,
          'axes.titlesize': size,
          'xtick.labelsize': size,#*0.75,
          'ytick.labelsize': size,#*0.75,
          'axes.titlepad': 20}
plt.rcParams.update(params)

def alter_parameter(r, R_max, a, b1, b2, q, c, d, k, p, h, g, m,
                          W_min, dt, R_0, W_0, U_0, adjustment=1, param='none'): # alter ['r','c','d','g','h','m','p']

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

    if n % 2 == 0:
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
    equity[n] = np.mean(U_array) #total well being  
    # print(resilience, equity)
  return resilience, equity


# plt.plot(np.arange(num_points)*dt,equity_r, label='r increased by 10%', color='#7494aa', linestyle='', markersize=10)
# colors = ['#3672dd','#7494aa','#374b5f', '#887e87', '#bda27b','#a47c6b' ]
# colors = ['#D98FD4', '#C1D4D9','#92A4A6','#FFDA1A','#BF8069', '#E64B1E']
colors = ['#D98FD4', '#649EDB', '#C1D4D9','#92A4A6','#FFDA1A','#F09975', '#E64B1E']

def add_to_scatter(x,a,param, i):

  if a > 1:
    adjustment = (a - 1) * 100
    string = f'increased by {adjustment:.0f}%'
  elif a < 1 and a >= 0:
    adjustment = (1 - a) * 100
    string = f'decreased by {adjustment:.0f}%'
  elif a == 1:
    param = "No parameters"
    string = 'altered'

  plt.plot(np.arange(num_points)*dt, x, label=f'{param} {string}', color=colors[i], linestyle='', markersize=10, alpha=0.8)



dictionary = {
              'r': [0.4, 0.6, 0.8, 0.9, .95, 1],
              # 'c': [0,0.4, 0.6,0.8, 1],
              'd': [1,1.05, 1.1, 1.2, 1.4, 1.6],
              # 'g': [0.6, 1, 1.4, 1.8, 2],
              # 'h': [1, 1.1, 2, 4],
              # 'm': [0.5, 1, 2, 3],
              # 'p': [.9, .95, 1, 1.05, 1.1],   
              }
for key in dictionary:
  # plt.figure()
  parameter = key
  item = dictionary[key]
  for i, adjustment in enumerate(item):
    resilience, equity = alter_parameter(r, R_max, a, b1, b2, q, c, d, k, p, h, g, m,
                          W_min, dt, R_0, W_0, U_0, adjustment=adjustment, param=parameter)
    plt.figure(1)  
    add_to_scatter(equity, adjustment, parameter, i)

    plt.figure(2)
    add_to_scatter(resilience, adjustment, parameter, i)
 
  plt.figure(1)  
  plt.title('Equity') 
  plt.legend()  
  plt.savefig(f'figures/equity-{key}-parameter-altered.png')

  plt.figure(2)
  plt.title('Resilience')
  plt.legend()
  plt.savefig(f'figures/resilience-{key}-parameter-altered.png')

  plt.figure(1).clear(True)
  plt.figure(2).clear(True)

# plt.show()

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

    U_array[n] = U[-1] # population (pseudo for resilience)

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
