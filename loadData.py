import numpy as np


dir = '/home/lennartgolks/PycharmProjects/firstModelCheckPhD/'
#dir = '/Users/lennart/PycharmProjects/firstModelCheckPhD/'
B_inv_A_trans_y0 = np.loadtxt(dir + 'B_inv_A_trans_y0.txt')
VMR_O3 = np.loadtxt(dir + 'VMR_O3.txt')
pressure_values = np.loadtxt(dir + 'pressure_values.txt')
height_values = np.loadtxt(dir + 'height_values.txt')
A = np.loadtxt(dir + 'AMat.txt')
APress = np.loadtxt(dir + 'AP.txt')
ATemp = np.loadtxt(dir + 'AT.txt')
gamma0 = np.loadtxt(dir + 'gamma0.txt')
y = np.loadtxt(dir + 'dataY.txt')
L = np.loadtxt(dir + 'GraphLaplacian.txt')
theta_scale_O3 = np.loadtxt(dir + 'theta_scale_O3.txt')
tang_heights_lin = np.loadtxt(dir + 'tan_height_values.txt')
num_mole = np.loadtxt(dir + 'num_mole.txt')
LineIntScal = np.loadtxt(dir +'LineIntScal.txt')
S = np.loadtxt(dir +'S.txt').reshape((909,1))
A_lin = np.loadtxt(dir + 'ALinMat.txt',)
temp_values = np.loadtxt(dir + 'temp_values.txt',)

np.savetxt('temp_values.txt', temp_values, fmt = '%.15f', delimiter= '\t')
np.savetxt('ALinMat.txt',A_lin, fmt = '%.15f', delimiter= '\t')
np.savetxt('S.txt', S, fmt = '%.15f', delimiter= '\t')
np.savetxt('LineIntScal.txt', LineIntScal, fmt = '%.15f', delimiter= '\t')
np.savetxt('num_mole.txt', [num_mole], fmt = '%.15f', delimiter= '\t')
np.savetxt('B_inv_A_trans_y0.txt', B_inv_A_trans_y0, fmt = '%.15f', delimiter= '\t')
np.savetxt('VMR_O3.txt', VMR_O3, fmt = '%.15f', delimiter= '\t')
np.savetxt('dataY.txt', y, fmt = '%.15f', delimiter= '\t')
np.savetxt('AMat.txt',A,fmt = '%.15f', delimiter= '\t')
np.savetxt('AP.txt',APress,fmt = '%.15f', delimiter= '\t')
np.savetxt('AT.txt',ATemp,fmt = '%.15f', delimiter= '\t')
np.savetxt('gamma0.txt',[gamma0])
np.savetxt('height_values.txt', height_values, fmt = '%.15f', delimiter= '\t')
np.savetxt('pressure_values.txt', pressure_values, fmt = '%.15f', delimiter= '\t')
np.savetxt('GraphLaplacian.txt', L, header = 'Graph Lalplacian', fmt = '%.15f', delimiter= '\t')
np.savetxt('theta_scale_O3.txt', [theta_scale_O3], fmt = '%.15f')
np.savetxt('tan_height_values.txt', tang_heights_lin, fmt = '%.15f', delimiter= '\t')

Ax = np.matmul(A, theta_scale_O3 * VMR_O3)

import matplotlib.pyplot as plt
fig3, ax1 = plt.subplots()
ax1.plot(Ax, tang_heights_lin)
ax1.scatter(y, tang_heights_lin)
ax1.plot(y, tang_heights_lin)
plt.show()