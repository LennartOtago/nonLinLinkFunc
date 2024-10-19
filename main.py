import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from functions import *
from scipy import constants, optimize
import numpy as np
import pytwalk
import torch
import torch.nn as nn
import torch.optim as optim
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

numberOfDat = 1000# SpecNumMeas
DataY = np.zeros((SpecNumMeas,numberOfDat))
relErr = np.zeros((SpecNumMeas,numberOfDat))
newO3 = np.zeros((SpecNumLayers,numberOfDat))
nonLinY = np.zeros((SpecNumMeas,numberOfDat))
J = np.zeros((SpecNumMeas,SpecNumLayers))
const = np.linspace(0,5e-6,numberOfDat)

# OrgData
A_O3, theta_scale_O3 = composeAforO3(A_lin, temp_values, pressure_values, ind, scalingConst)
nonLinA = calcNonLin(A_lin, pressure_values, ind, temp_values, VMR_O3,
                     AscalConstKmToCm,
                     SpecNumLayers, SpecNumMeas)
OrgData = np.matmul(A_O3 * nonLinA,VMR_O3 * theta_scale_O3).reshape(SpecNumMeas)


SetDelta = Parabel(height_values, 30, 1e-7, 0.8e-4)
SetDelta = np.ones(height_values.size)

TriU = np.tril(np.triu(np.ones((n, n)), k=1), 1) * SetDelta
TriL = np.triu(np.tril(np.ones((n, n)), k=-1), -1) * SetDelta.T
Diag = np.eye(n) * np.sum(TriU + TriL, 0)

L_d = -TriU + Diag - TriL
L_d[0, 0] = 2 * L_d[0, 0]
L_d[-1, -1] = 2 * L_d[-1, -1]

# fig3, ax3 = plt.subplots()
# fig2, ax2 = plt.subplots()
# fig1, ax1 = plt.subplots()
fig0, ax0 = plt.subplots()
for i in range(0,numberOfDat):

    # tryO3 = np.copy(VMR_O3)
    # tryO3[i] = VMR_O3[i] + 1e-6
    #
    # newO3[:,i] = tryO3.reshape(SpecNumLayers)
    newO3[:,i] = np.random.multivariate_normal(VMR_O3.reshape(SpecNumLayers), 1e-12*L_d)
    newO3[:, i][newO3[:,i]<0] = 0

    A_O3, theta_scale_O3 = composeAforO3(A_lin, temp_values, pressure_values, ind, scalingConst)
    nonLinA = calcNonLin(A_lin, pressure_values, ind, temp_values,   newO3[:,i].reshape((SpecNumLayers,1)), AscalConstKmToCm,
                     SpecNumLayers, SpecNumMeas)


    #const is *0.991


    DataY[:, i] = np.matmul(A_O3 *2 ,  newO3[:,i].reshape((SpecNumLayers,1))  * theta_scale_O3).reshape(SpecNumMeas)
    nonLinY[:, i] = np.matmul(A_O3 * nonLinA,  newO3[:,i].reshape((SpecNumLayers,1)) * theta_scale_O3).reshape(SpecNumMeas)

#DataY[:, i] = DataY[:, i] * (0.1*np.sinc(x) * np.exp(-0.3 * x)).reshape(SpecNumMeas)

    #ax1.scatter(DataY[:, i],tang_heights_lin, c='red')
    #relErr[:,i] = (DataY[:, i]-nonLinY[:, i])/nonLinY[:, i]
    #ax3.scatter((DataY[:, i]-nonLinY[:, i])/nonLinY[:, i],tang_heights_lin)


    #ax2.scatter(nonLinY[:, i],tang_heights_lin)
    ax0.plot(newO3[:,i],height_values)
    #ax0.set_title('Train Profiles')

plt.show()
## do machine learing form here
# normallize data
nonLinY_norm = ((nonLinY.T - np.mean(nonLinY,1)) / np.std(nonLinY,1)).T
LinY_norm = ((DataY.T - np.mean(DataY,1)) / np.std(DataY,1)).T

x_tensor = torch.tensor(nonLinY_norm, dtype = torch.float32)
y_tensor = torch.tensor(LinY_norm, dtype = torch.float32)

class LinearModel(nn.Module):
    def __init__(self, in_features , out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        return self.linear(x).squeeze(1)


in_features = SpecNumMeas
out_features =  SpecNumMeas

model = LinearModel(in_features, out_features)

criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=3)

num_epochs = 100



for epoch in range(num_epochs):



    #forward pass
    outputs = model(x_tensor.T)

    #calculate Los
    loss = criterion(outputs, y_tensor.T)

    #backwardpass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f'Epoch[ {epoch+1} / {num_epochs}], Loss: {loss.item():.4f}')

##

print('Machine Learin done')


    #construct Jacobian
#    for j1 in range(0,SpecNumMeas):
#    J[j1,i] = (OrgData[j1] - nonLinY[j1, i])/(VMR_O3[i] - tryO3[i])
#np.savetxt('theta_scale_O3.txt', [theta_scale_O3], fmt = '%.15f', delimiter= '\t')
#np.savetxt('AMat.txt', A, fmt = '%.15f', delimiter= '\t')
#np.savetxt('ALinMat.txt', A_lin, fmt = '%.15f', delimiter= '\t')
#y, gamma0 = add_noise(nonLinY.reshape((m,1)), SNR)

#check linearly independent



#xNonLin, exitCode = scy.sparse.linalg.gmres(nonLinY, np.zeros(SpecNumMeas))

#xLin, exitCode = scy.sparse.linalg.gmres(DataY, np.zeros(SpecNumMeas))

QnonLin, RnonLin = np.linalg.qr(nonLinY, mode = 'complete')
QLin, RLin = np.linalg.qr(DataY, mode = 'complete')

print(np.allclose(nonLinY, np.dot(QnonLin, RnonLin)))
print(np.allclose(DataY, np.dot(QLin, RLin)))
nonLinMap = QnonLin#/sum(QnonLin,0)
LinMap = QLin#/sum(QLin,0)

#ax3.set_title("difference")
#plt.show()
xtry = np.zeros((SpecNumMeas,1))
xtry[1] = -1
# fig0, ax0 = plt.subplots()
# ax0.plot(np.matmul(LinMap,xtry),tang_heights_lin)
plt.show()

Map = np.zeros((SpecNumMeas,SpecNumMeas))
MaschMap = np.zeros((SpecNumMeas,SpecNumMeas))
from sklearn import linear_model
for i in range(0,SpecNumMeas):


    reg = linear_model.LinearRegression()
    #reg = linear_model.BayesianRidge()
    reg.fit(nonLinY.T, DataY[i])

    MaschMap[i] = reg.coef_
    #Map[i] = np.linalg.solve(nonLinY.T, DataY[i])



plt.close("all")

#np.savetxt('Map.txt', Map,fmt = '%.15f', delimiter= '\t')

Map = np.loadtxt('Map.txt')

fig5, ax5 = plt.subplots()

for k in range(0,10):
    fig4, ax4 = plt.subplots()

    tryO3 = np.copy(VMR_O3)
    tryO3[i] = VMR_O3[k+5] + 1e-6

    testO3 = tryO3.reshape(SpecNumLayers)
    #testO3 = np.random.multivariate_normal(VMR_O3.reshape(SpecNumLayers), 1e-15*L_d)
    #testO3 = VMR_O3
    testO3[testO3<0] = 0
    A_O3, theta_scale_O3 = composeAforO3(A_lin, temp_values, pressure_values, ind, scalingConst)
    nonLinA = calcNonLin(A_lin, pressure_values, ind, temp_values,   testO3.reshape((SpecNumLayers,1)), AscalConstKmToCm,
                     SpecNumLayers, SpecNumMeas)





    testDataY = np.matmul(A_O3 *2 ,  testO3.reshape((SpecNumLayers,1))  * theta_scale_O3).reshape(SpecNumMeas)
    testNonLinY= np.matmul(A_O3 * nonLinA,  testO3.reshape((SpecNumLayers,1)) * theta_scale_O3).reshape(SpecNumMeas)

    normTestNonLin = (testNonLinY - np.mean(nonLinY, 1)) / np.std(nonLinY, 1)

    normTestNonLin_tensor = torch.tensor(normTestNonLin, dtype = torch.float32).view(1,-1)
    model.eval()

    with torch.no_grad():
        normTestOutput = model(normTestNonLin_tensor)

    TorchPredic =  np.array(normTestOutput).reshape(SpecNumMeas) * np.std(DataY, 1) + np.mean(DataY, 1)


    noisyNonLinDat, noiseLevel = add_noise(testNonLinY, 60)

    relMapErr = np.linalg.norm(testDataY -  Map @ testNonLinY)/ np.linalg.norm(testDataY) * 100
    relMaschMapErr = np.linalg.norm(testDataY - MaschMap @ testNonLinY) / np.linalg.norm(testDataY) *100
    relTorchMapErr = np.linalg.norm(testDataY - TorchPredic) / np.linalg.norm(testDataY) *100
    relNoiseErr = np.linalg.norm(noisyNonLinDat - testNonLinY)/ np.linalg.norm(testNonLinY) * 100
    ax5.plot(testO3, height_values)
    ax5.set_title('Test Profiles')
    ax4.plot(Map @ testNonLinY,tang_heights_lin, label = 'mapped Lin Data, rel Err: '+ str(np.round(relMapErr,3)) + '%')
    ax4.plot(MaschMap @ testNonLinY, tang_heights_lin, color = "k", label = 'masch mapped Lin Data, rel Err: '+ str(np.round(relMaschMapErr,3)) + '%')
    ax4.plot(TorchPredic, tang_heights_lin, color="r",label='torch mapped Lin Data, rel Err: ' + str(np.round(relTorchMapErr, 3)) + '%')

    ax4.plot(testDataY,tang_heights_lin, label = 'true Lin Data', marker = 'o',linewidth = 1.5, markersize =7, zorder =0)
    ax4.plot(noisyNonLinDat, tang_heights_lin, label='Noisy non Lin Data, rel Err: ' + str(np.round(relNoiseErr,3)) + '%' )

    ax4.legend()

plt.show()

# fig4, ax4 = plt.subplots(14, 1, figsize=(8,15))
# for i in range(0,14):
#     ax4[i].hist(relErr[i, :], bins=50)
#
#
# fig5, ax5= plt.subplots(14, 1, figsize=(8,15))
# for i in range(19,33):
#     ax5[i-19].hist(relErr[i, :], bins=50)
#
#
#
#
# plt.show()


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