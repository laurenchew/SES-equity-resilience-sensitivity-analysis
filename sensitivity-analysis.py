from SALib.sample import saltelli
from SALib.analyze import sobol
from SALib.test_functions import Ishigami
import numpy as np
from simulate import simulate_SES
import pyDOE as doe

'''
# Model Parameters
# r = 0.1  # resource regeneration rate, [0.015, 0.105]
r = 10 # resource regeneration rate, [1.5, 100] #normalized
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

r_range = [1.5, 100]
c_range = [0, 0.5]
d_range = [95, 225]
g_range = [0, 0.02]
h_range = [0, 120]
m_range = [0.3, 12.3]
p_range = [2.7,18]
#k_range = 
#a_range = 
#b1_range = [0.05, 1.5] ?
#b2_range = [0.05, 1.5]

k = 2 # water access parameter (how steeply benefit of water rises)
a = 1000  # profit cap
b1 = 1 # profit vs. water parameter, or = 0.05
b2 = 1 # profit vs. labor parameter, or = 0.05

problem = {
  'num_vars': 7,
  'names': ['r','c','d','g','h','m','p'], #'k', 'a','b1', 'b2',
  'bounds': [[1.5,100],[0, 0.5],[95, 225],[0, 0.02],[0, 120],[0.3, 12.3],[2.7,18]]
}

# Generate samples
param_values = saltelli.sample(problem, 5)
N = len(param_values) # number of parameter samples #160 with 10 in above line
R_array = np.zeros(N)
U_array = np.zeros(N)
P_array = np.zeros(N)
W_array = np.zeros(N)
L_array = np.zeros(N)
S_array = np.zeros(N)

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
  num_points = 10 #Was 80
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
  equity = np.mean(U_array) # total well being 
  print(resilience, equity) 

  return resilience, equity

Y_resilience = np.zeros(N)
Y_equity = np.zeros(N)

# Run model
for i in range (N):
  if i % 10 == 0:
    print(i)
  Y_resilience[i], Y_equity[i] = SES_model(param_values[i])

# Perform analysis, calculate sensitivity indices
Si = sobol.analyze(problem, Y_resilience, 
                  print_to_console=True, num_resamples = 100) #1000
# Returns a dictionary with keys 'S1', 'S1_conf', 'ST', and 'ST_conf'
# (first and total-order indices with bootstrap confidence intervals)

Sii = sobol.analyze(problem, Y_equity, 
                  print_to_console=True, num_resamples = 100) #1000