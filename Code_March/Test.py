import numpy as np

"""
test_lis = np.array([[1,2,3,4,5,6,7,8,9,10]])
test_lis2 = np.array([[1,2,3,4,5,6,7,8,9,10]])


print((test_lis.T @ test_lis2))

arr = np.zeros((len(test_lis[0])//2, 1))
for j, i in enumerate(range(0, len(test_lis[0]), 2)):
    print((test_lis.T @ test_lis2)[5][i])
    arr[j] = (test_lis.T @ test_lis2)[5][i]
print(arr)
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
 
# Define the ODE as a function: f(t, y)
def lotka_volterra(t, z):
    x, y = z
    alpha, beta, delta, gamma = 1.1, 0.4, 0.1, 0.4
    dxdt = alpha*x - beta*x*y
    dydt = delta*x*y - gamma*y
    return [dxdt, dydt]

t_span = (0, 15)
y0 = [10, 5]
sol = solve_ivp(lotka_volterra, t_span, y0, t_eval=np.linspace(0, 15, 300))
plt.plot(sol.t, sol.y[0], label='Prey')
plt.plot(sol.t, sol.y[1], label='Predator')
plt.legend()
plt.clf()



import matplotlib.colors

def loadFrameGrab(path):
    points = []
    with open(path) as f:
        next(f)
        for line in f:
            points.append([float(x) for x in line.replace(',', '').split()])
    return np.array(points)

Test1 = loadFrameGrab('Hahn_echo_fram_grab.csv')

Test2 = Test1.copy()
Test3 = Test1.copy()
Test4 = Test1.copy()
Test5 = Test1.copy()
Test2[:,1], Test3[:,1], Test4[:,1], Test5[:,1]= Test2[:,1]/2, Test3[:,1]/3, Test4[:,1]/4, Test5[:,1]/5


norm=plt.Normalize(0,111)
cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["red","blue"])

plt.scatter(0,0, color = cmap(0.1))
#plt.scatter(Test1[:,0], Test1[:,1], c=np.arange(0,len(Test1[:,0])), cmap = cmap, norm=norm)



plt.plot(Test1[:,0], Test1[:,1], color = cmap(0.2))
plt.plot(Test2[:,0], Test2[:,1], color = cmap(0.4))
plt.plot(Test3[:,0], Test3[:,1], color = cmap(0.6))
plt.plot(Test4[:,0], Test4[:,1], color = cmap(0.8))
plt.plot(Test5[:,0], Test5[:,1], color = cmap(1.0))

plt.show()







