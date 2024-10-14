import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from functions import *
from scipy import constants, optimize
import numpy as np
import pytwalk
''' load data and pick wavenumber/frequency'''

files = '634f1dc4.par' #/home/lennartgolks/Python /Users/lennart/PycharmProjects

my_data = pd.read_csv(files, header=None)
data_set = my_data.values

size = data_set.shape
wvnmbr = np.zeros((size[0],1))
S = np.zeros((size[0],1))
F = np.zeros((size[0],1))
g_air = np.zeros((size[0],1))
g_self = np.zeros((size[0],1))
E = np.zeros((size[0],1))
n_air = np.zeros((size[0],1))
g_doub_prime= np.zeros((size[0],1))


for i, lines in enumerate(data_set):
    wvnmbr[i] = float(lines[0][5:15]) # in 1/cm
    S[i] = float(lines[0][16:25]) # in cm/mol
    F[i] = float(lines[0][26:35])
    g_air[i] = float(lines[0][35:40])
    g_self[i] = float(lines[0][40:45])
    E[i] = float(lines[0][46:55])
    n_air[i] = float(lines[0][55:59])
    g_doub_prime[i] = float(lines[0][155:160])


np.savetxt('wvnmbr.txt', wvnmbr, fmt = '%.15f', delimiter= '\t')
np.savetxt('S.txt', S, fmt = '%.30f', delimiter= '\t')
np.savetxt('F.txt', F, fmt = '%.15f', delimiter= '\t')
np.savetxt('g_air.txt', g_air,fmt = '%.15f', delimiter= '\t')
np.savetxt('g_self.txt', g_self,fmt = '%.15f', delimiter= '\t')
np.savetxt('E.txt', E,fmt = '%.15f', delimiter= '\t')
np.savetxt('n_air.txt', n_air,fmt = '%.15f', delimiter= '\t')
np.savetxt('g_doub_prime.txt', g_doub_prime,fmt = '%.15f', delimiter= '\t')


df = pd.read_excel('ExampleOzoneProfiles.xlsx')
#get the values for a given column
press = df['Pressure (hPa)'].values #in hectpascal or millibars
O3 = df['Ozone (VMR)'].values
np.savetxt('press.txt', press, fmt = '%.15f', delimiter= '\t')
np.savetxt('O3.txt', O3, fmt = '%.15f', delimiter= '\t')





press = np.loadtxt('press.txt', delimiter= '\t')
O3 = np.loadtxt('O3.txt', delimiter= '\t')


#O3[42:51] = np.mean(O3[-4::])
minInd = 2
maxInd = 45#54
pressure_values = press[minInd:maxInd]
SpecNumLayers = len(pressure_values)

VMR_O3 = O3[minInd:maxInd].reshape((SpecNumLayers,1))
scalingConstkm = 1e-3

def height_to_pressure(p0, x, dx):
    R = constants.gas_constant
    R_Earth = 6371  # earth radiusin km
    grav = 9.81 * ((R_Earth)/(R_Earth + x))**2
    temp = get_temp_values([x])
    return p0 * np.exp(-28.97 * grav / R * dx/temp )

def pressure_to_height(p0, pplus, x):
    R = constants.gas_constant
    R_Earth = 6371  # earth radiusin km
    grav = 9.81 * ((R_Earth)/(R_Earth + x))**2
    temp = get_temp_values([x])
    return np.log(pplus/p0) /(-28.97 * grav / R /temp )

calc_press = np.zeros((len(press)+1,1))
calc_press[0] = 1013.25
calc_press[1:] = press.reshape((len(press),1)) #hPa
actual_heights = np.zeros((len(press)+1,1))

for i in range(1,len(calc_press)):
    dx = pressure_to_height(calc_press[i-1], calc_press[i], actual_heights[i-1])
    actual_heights[i] = actual_heights[i - 1] + dx


##

heights = actual_heights[1:]

height_values = heights[minInd:maxInd].reshape((SpecNumLayers,1))
temp_values = get_temp_values(height_values)

np.savetxt('height_values.txt',height_values, fmt = '%.15f', delimiter= '\t')
np.savetxt('VMR_O3.txt',VMR_O3, fmt = '%.15f',delimiter= '\t')
np.savetxt('pressure_values.txt',pressure_values, fmt = '%.15f', delimiter= '\t')
np.savetxt('temp_values.txt',temp_values, fmt = '%.15f', delimiter= '\t')




""" analayse forward map without any real data values"""

MinH = height_values[0]
MaxH = height_values[-1]
R_Earth = 6371 # earth radiusin km
ObsHeight = 500 # in km

''' do svd for one specific set up for linear case and then exp case'''

SpecNumMeas = 45


n = SpecNumLayers
m = SpecNumMeas

# find minimum and max angle in radians
# min and max angle are defined by the height values of the retrived profile
MaxAng = np.arcsin((height_values[-1]+ R_Earth) / (R_Earth + ObsHeight))
MinAng = np.arcsin((height_values[0] + R_Earth) / (R_Earth + ObsHeight))


meas_ang = np.linspace(MinAng, MaxAng, SpecNumMeas)
pointAcc = 0.0009
meas_ang = np.array(np.arange(MinAng[0], MaxAng[0], pointAcc))
#meas_ang = np.array(np.arange(MinAng[0], MaxAng[0], 0.00045))
SpecNumMeas = len(meas_ang)
m = SpecNumMeas

A_lin, tang_heights_lin, extraHeight = gen_sing_map(meas_ang,height_values,ObsHeight,R_Earth)
np.savetxt('tang_heights_lin.txt',tang_heights_lin, fmt = '%.15f', delimiter= '\t')


AscalConstKmToCm = 1e3
ind = 623
SNR = 60
scalingConst = 1e11

numberOfDat = 10
DataY = np.zeros((SpecNumMeas,numberOfDat))
newO3 = np.zeros((SpecNumLayers,numberOfDat))
nonLinY = np.zeros((SpecNumMeas,numberOfDat))
J = np.zeros((numberOfDat,SpecNumMeas,SpecNumLayers))
const = np.linspace(0,5e-6,numberOfDat)
fig3, ax3 = plt.subplots()
fig2, ax2 = plt.subplots()
fig1, ax1 = plt.subplots()
fig0, ax0 = plt.subplots()
for i in range(0,numberOfDat):

    tryO3 = VMR_O3 + const[i]
    #tryO3[j] = VMR_O3[j] + const[i]
    newO3[:,i] = tryO3.reshape(SpecNumLayers)
    #newO3[:,i]= np.random.multivariate_normal(VMR_O3.reshape(SpecNumLayers), np.eye(SpecNumLayers))


    A_O3, theta_scale_O3 = composeAforO3(A_lin, temp_values, pressure_values, ind, scalingConst)
    nonLinA = calcNonLin(A_lin, pressure_values, ind, temp_values, newO3[:,i].reshape((SpecNumLayers,1)), AscalConstKmToCm,
                     SpecNumLayers, SpecNumMeas)


    #const is *0.991


    DataY[:, i] = np.matmul(A_O3 *2 ,newO3[:,i].reshape((SpecNumLayers,1))  * theta_scale_O3).reshape(SpecNumMeas)
    nonLinY[:, i] = np.matmul(A_O3 * nonLinA,newO3[:,i].reshape((SpecNumLayers,1)) * theta_scale_O3).reshape(SpecNumMeas)

#DataY[:, i] = DataY[:, i] * (0.1*np.sinc(x) * np.exp(-0.3 * x)).reshape(SpecNumMeas)

    ax1.scatter(DataY[:, i],tang_heights_lin, c='red')
    ax3.scatter(DataY[:, i]-nonLinY[:, i],tang_heights_lin, c='red')
    ax2.scatter(nonLinY[:, i],tang_heights_lin)
    ax0.scatter(newO3[:,i],height_values)

    #construct Jacobian
    for j1 in range(0,SpecNumMeas):
        for j2 in range(0,SpecNumLayers):
            J[i][j1,j2] = (DataY[j1, i] - nonLinY[j1, i])/(newO3[:,i])
#np.savetxt('theta_scale_O3.txt', [theta_scale_O3], fmt = '%.15f', delimiter= '\t')
#np.savetxt('AMat.txt', A, fmt = '%.15f', delimiter= '\t')
#np.savetxt('ALinMat.txt', A_lin, fmt = '%.15f', delimiter= '\t')
#y, gamma0 = add_noise(nonLinY.reshape((m,1)), SNR)


sinc_Values = np.sinc(height_values)
ax3.set_title("difference")
plt.show()

print("done")
##

# ATA= np.matmul((A_O3 *2).T , A_O3 *2)
# ATy = np.matmul((A_O3 *2).T, y)
#
# A= A_O3 *2
#
# tol = 1e-8
# SetDelta = Parabel(height_values, 30, 1e-7, 0.8e-4)
#
# TriU = np.tril(np.triu(np.ones((n, n)), k=1), 1) * SetDelta
# TriL = np.triu(np.tril(np.ones((n, n)), k=-1), -1) * SetDelta.T
# Diag = np.eye(n) * np.sum(TriU + TriL, 0)
#
# L_d = -TriU + Diag - TriL
# L_d[0, 0] = 2 * L_d[0, 0]
# L_d[-1, -1] = 2 * L_d[-1, -1]
# ATA = np.matmul(A.T, A)
# B0 = (ATA + 1 / gamma0 * L_d)
# B_inv_A_trans_y0, exitCode = scy.sparse.linalg.gmres(B0, ATy[:, 0], rtol=tol, restart=25)
#
#
#
# def MargPostSupp(Params):
#     list = []
#     list.append(gamma0 * 1.75 > Params[0] >gamma0 * 0.25)
#     list.append(36> Params[1] > 29)
#     list.append(1e-4 > Params[2] > 1e-9)
#     list.append(1e-4 > Params[3] > 1e-5)
#
#     return all(list)
#
# tWalkSampNumDel = 30000
# burnInDel = 100
#
# log_post = lambda Params: -log_postO3(Params, ATA, ATy, height_values, B_inv_A_trans_y0, VMR_O3, y, A, gamma0)
# d0Mean = 0.8e-4
# hMean = height_values[VMR_O3[:] == np.max(VMR_O3[:])][0]
# a0Mean = 2e-9
#
# MargPost = pytwalk.pytwalk(n=4, U=log_post, Supp=MargPostSupp)
# x0 = np.array([gamma0, hMean, a0Mean, d0Mean])
# xp0 = (1 + 1e-10) * x0
# MargPost.Run(T=tWalkSampNumDel + burnInDel, x0=x0, xp0=xp0)
#
# Samps = MargPost.Output
#
#
# fig4, ax4 = plt.subplots(4, 1, figsize=(8,15))
# # ax4 = BigFig.add_subplot(4, 3, 4)
# ax4[0].hist(Samps[burnInDel:,0], bins=50)
# ax4[0].axvline(x=gamma0, color='r')
# ax4[0].set_xlabel('$\gamma$')
# # ax5 = BigFig.add_subplot(4, 3, 7)
# ax4[1].hist(Samps[burnInDel:,1], bins=50)
# ax4[1].set_xlabel('$h_0$')
# # ax6 = BigFig.add_subplot(4, 3,10)
# ax4[2].hist(Samps[burnInDel:,2], bins=50)
# ax4[2].set_xlabel('$a$')
# # ax7 = BigFig.add_subplot(4, 3,13)
# ax4[3].hist(Samps[burnInDel:,3], bins=50)
# ax4[3].set_xlabel('$\delta_0$')
# #ax4[4].hist(Samps[burnInDel:,4], bins=50)
#
# # ax4[3].axvline(x=deltRes[0,2]/0.4, color='r')
# # plt.savefig('TWalkHistO3.png')
# #plt.show()
#
# print(np.mean(Samps,0))
# fig4, ax4 = plt.subplots()
#
# ax4.hist(Samps[burnInDel:,4], bins=50)
#
# plt.show()
# numOfO3 = 100
# n = SpecNumLayers
# m = len(y)
# O3_Prof = np.zeros((numOfO3,n))
# for i in range(0, numOfO3):
#     MWGRand =  np.random.randint(low=0, high=tWalkSampNumDel)
#     SetGamma = Samps[burnInDel+MWGRand,0]
#     SetDelta = Parabel(height_values, *Samps[MWGRand,1:-1])
#     #SetDelta = np.ones((1,n)) * Samps[MWGRand,3]
#
#     TriU = np.tril(np.triu(np.ones((n, )), k=1), 1) * SetDelta
#     TriL = np.triu(np.tril(np.ones((n, n)), k=-1), -1) * SetDelta.T
#     Diag = np.eye(n) * np.sum(TriU + TriL, 0)
#
#     L_d = -TriU + Diag - TriL
#     L_d[0, 0] = 2 * L_d[0, 0]
#     L_d[-1, -1] = 2 * L_d[-1, -1]
#     SetB = SetGamma * ATA + L_d
#
#     W = np.random.multivariate_normal(np.zeros(len(A)), np.eye(len(A)))
#     v_1 = np.sqrt(SetGamma) * A.T @ W.reshape((m, 1))
#     W2 = np.random.multivariate_normal(np.zeros(len(L_d)), L_d)
#     v_2 = W2.reshape((n, 1))
#
#     RandX = (SetGamma * ATy + v_1 + v_2)
#
#     Results, exitCode = scy.sparse.linalg.gmres(SetB, RandX[0::, 0], rtol=1e-5)
#     O3_Prof[i,:] = Results / theta_scale_O3
#
#
#
# ResCol = "#1E88E5"
# TrueCol = [50/255,220/255, 0/255]
#
# fig3, ax1 = plt.subplots()
#
#
#
# ax1.plot(VMR_O3,height_values,marker = 'o',markerfacecolor = TrueCol, color = TrueCol , label = 'true profile', zorder=1 ,linewidth = 1.5, markersize =7)
#
#
# for r in range(1,numOfO3):
#     Sol = O3_Prof[r, :]
#
#     ax1.plot(Sol,height_values,marker= '+',color = ResCol, zorder = 0, linewidth = 0.5, markersize = 5, alpha = 0.3)
#
# ax1.plot(np.mean(O3_Prof[1:],0), height_values, marker='>', color="k", label='sample mean', zorder=2, linewidth=0.5,
#              markersize=5)
#
# plt.savefig('O3Results.png')
# plt.show()
# print("done")