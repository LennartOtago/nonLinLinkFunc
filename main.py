import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from functions import *
from scipy import constants, optimize
import numpy as np
import pytwalk
import torch
import time
""" for plotting figures,
PgWidth in points, either collumn width page with of Latex"""
fraction = 1.5
dpi = 300
PgWidthPt = 245
PgWidthPt =  fraction * 421/2 #phd
defBack = mpl.get_backend()
mpl.use(defBack)
mpl.rcParams.update(mpl.rcParamsDefault)
plt.rcParams.update({'font.size': fraction * 12,
                     'text.usetex': True,
                     'font.family' : 'serif',
                     'font.serif'  : 'cm',
                     'text.latex.preamble': r'\usepackage{bm, amsmath}'})
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
SNR = 10
scalingConst = 1e11

numberOfDat = SpecNumMeas
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
OrgData = np.matmul(A_O3 * nonLinA,VMR_O3 * theta_scale_O3)
# OrgData = np.matmul(A_O3 * 2,VMR_O3 * theta_scale_O3)
##
# samplig  linear
SampleRounds = 100
numDatSet = SpecNumMeas
deltRes = np.zeros((SampleRounds, 3))
gamRes = np.zeros((SampleRounds))
Results = np.zeros((SampleRounds, SpecNumLayers))
DataSet = np.zeros((SpecNumMeas))

tol = 1e-8

tWalkSampNum = 7500
burnIn = 500

y , gamma0 = add_noise(OrgData, SNR)

# y = np.loadtxt('/home/lennartgolks/PycharmProjects/AllnonLinear/dataY.txt').reshape((SpecNumMeas,1))
# gamma0 = np.loadtxt('/home/lennartgolks/PycharmProjects/AllnonLinear/gamma0.txt')
# noiseVec = np.loadtxt('/home/lennartgolks/PycharmProjects/AllnonLinear/noiseVec.txt')
#
# ANonLinMat = np.loadtxt('/home/lennartgolks/PycharmProjects/AllnonLinear/ANonLinMat.txt')
# print(np.allclose(A_O3 * nonLinA,ANonLinMat))
# AMat_lin = np.loadtxt('/home/lennartgolks/PycharmProjects/AllnonLinear/ALinMat.txt')
# print(np.allclose(A_lin,AMat_lin))
# FirstA_O3 = np.loadtxt('/home/lennartgolks/PycharmProjects/AllnonLinear/LinAMatO3.txt')
# FirstnonLinA = np.loadtxt('/home/lennartgolks/PycharmProjects/AllnonLinear/nonLinAMat.txt')
# print(np.allclose(A_O3,FirstA_O3))
# print(np.allclose(nonLinA,FirstnonLinA))
# FirstTemp = np.loadtxt('/home/lennartgolks/PycharmProjects/AllnonLinear/temp_values.txt')
# FirstPress = np.loadtxt('/home/lennartgolks/PycharmProjects/AllnonLinear/pressure_values.txt')
# FirstHeight = np.loadtxt('/home/lennartgolks/PycharmProjects/AllnonLinear/height_values.txt')
# print(np.allclose(pressure_values,FirstPress))
# print(np.allclose(temp_values,FirstTemp.reshape((SpecNumLayers,1))))
# print(np.allclose(height_values,FirstHeight.reshape((SpecNumLayers,1))))



DataSet = y.reshape(SpecNumMeas)
gamRes[0] = gamma0
Results[0] = VMR_O3.reshape(SpecNumLayers)
deltRes[0] = np.array([30, 1e-7, 0.8e-4])
SetDelta = Parabel(height_values, *deltRes[0])


TriU = np.tril(np.triu(np.ones((n, n)), k=1), 1) * SetDelta
TriL = np.triu(np.tril(np.ones((n, n)), k=-1), -1) * SetDelta.T
Diag = np.eye(n) * np.sum(TriU + TriL, 0)

L_d = -TriU + Diag - TriL
L_d[0, 0] = 2 * L_d[0, 0]
L_d[-1, -1] = 2 * L_d[-1, -1]





def MargPostSupp(Params):
    list = []
    list.append(Params[0] > 0)
    list.append(height_values[-1] > Params[1] > height_values[0])
    list.append(Params[2] > 0)
    list.append( Params[3] > 0)
    return all(list)


A = A_O3 * 2

ATy = np.matmul(A.T, y)
ATA = np.matmul(A.T, A)

B0 = (ATA + 1 / gamRes[0] * L_d)
B_inv_A_trans_y0, exitCode = scy.sparse.linalg.gmres(B0, ATy[:, 0], rtol=tol, restart=25)
if exitCode != 0:
    print(exitCode)

log_post = lambda Params : log_postO3(Params, ATA, ATy, height_values, B_inv_A_trans_y0, Results[0, :].reshape((SpecNumLayers,1)), y, A, gamma0)
MargPost = pytwalk.pytwalk(n=4, U=log_post, Supp=MargPostSupp)
x0 = np.array([gamRes[0], *deltRes[0, :]])
xp0 = 1.0001 * x0
startTime = time.time()
MargPost.Run(T=tWalkSampNum + burnIn, x0=x0, xp0=xp0)
print('time elapsed:' + str(time.time() - startTime))
Samps = MargPost.Output

for samp in range(1,SampleRounds):


    MWGRand = burnIn + np.random.randint(low=0, high=tWalkSampNum)
    SetGamma = Samps[MWGRand, 0]
    SetDelta = Parabel(height_values, *Samps[MWGRand, 1:-1])

    TriU = np.tril(np.triu(np.ones((n, n)), k=1), 1) * SetDelta
    TriL = np.triu(np.tril(np.ones((n, n)), k=-1), -1) * SetDelta.T
    Diag = np.eye(n) * np.sum(TriU + TriL, 0)

    L_d = -TriU + Diag - TriL
    L_d[0, 0] = 2 * L_d[0, 0]
    L_d[-1, -1] = 2 * L_d[-1, -1]
    SetB = SetGamma * ATA +  L_d

    W = np.random.multivariate_normal(np.zeros(len(A)), np.eye(len(A)))
    v_1 = np.sqrt(SetGamma) * A.T @ W.reshape((m, 1))
    W2 = np.random.multivariate_normal(np.zeros(len(L_d)), L_d)
    v_2 = W2.reshape((n, 1))

    RandX = (SetGamma * ATy + v_1 + v_2)
    O3_Prof, exitCode = scy.sparse.linalg.gmres(SetB, RandX[0::, 0], rtol=tol)
    Results[samp, :] = O3_Prof/ theta_scale_O3
    deltRes[samp, :] = np.array([Samps[MWGRand, 1:-1]])
    gamRes[samp] = SetGamma

        # print(np.mean(O3_Prof))


    #np.savetxt(f'Res{testSet}.txt', Results[testSet],fmt = '%.15f', delimiter= '\t')
##
firstMean = np.mean(Results, 0)
ResCol = "#1E88E5"
fig4, ax4 = plt.subplots(figsize=set_size(PgWidthPt, fraction=fraction))
for r in range(0,SampleRounds):
    Sol = Results[r, :]

    ax4.plot(Sol,height_values,marker= '+',color = ResCol, zorder = 0, linewidth = 0.5, markersize = 5, alpha = 0.3)

ax4.plot(firstMean, height_values,marker= 'o',color = 'r')
ax4.plot(VMR_O3, height_values,marker= 'o',color = 'g')
plt.show()

## use O3 profiles to make up lin and nonLin data and use for mapping
# test Real Map
DatCol =  'gray'
ResCol = "#1E88E5"
TrueCol = [50/255,220/255, 0/255]
currMap = np.eye(SpecNumMeas)
RealMap, relMapErr, LinDataY, NonLinDataY = genDataFindandtestMap(currMap, L_d, gamma0, VMR_O3, Results, AscalConstKmToCm, A_lin, temp_values, pressure_values, ind, scalingConst,SampleRounds)

print(f'Mean rel Error form Map: {np.mean(relMapErr):.2f}')


##
testNum = 1000
testO3 = np.random.multivariate_normal(VMR_O3.reshape(SpecNumLayers), 1e-12 * L_d, size=testNum)
testO3[testO3 < 0] = 0
testDataY = np.zeros((testNum, SpecNumMeas))
testNonLinY = np.zeros((testNum, SpecNumMeas))

for k in range(0, testNum):
    currO3 = testO3[k]
    noise = 0#np.random.normal(0, np.sqrt(1 / gamma0), SpecNumMeas)
    nonLinA = calcNonLin(A_lin, pressure_values, ind, temp_values, currO3.reshape((SpecNumLayers, 1)),
                         AscalConstKmToCm,
                         SpecNumLayers, SpecNumMeas)

    testDataY[k] = np.matmul(A_O3 * 2, currO3.reshape((SpecNumLayers, 1)) * theta_scale_O3).reshape(SpecNumMeas) + noise
    testNonLinY[k] = np.matmul(A_O3 * nonLinA, currO3.reshape((SpecNumLayers, 1)) * theta_scale_O3).reshape(
        SpecNumMeas) + noise

##

print(np.mean(testDataY,0) - np.mean(testNonLinY,0))

from scipy.stats import wasserstein_distance_nd

print(wasserstein_distance_nd(testDataY, testNonLinY))

##
from scipy.stats import wasserstein_distance
for k in range(0, SpecNumMeas):
    print(wasserstein_distance(testDataY[:,k], testNonLinY[:,k]))

##
import torch
from geomloss import SamplesLoss

tensorX = torch.tensor(testNonLinY, requires_grad=True)
tensorY = torch.tensor(testDataY)
start = time.time()
# Define a Sinkhorn (~Wasserstein) loss between sampled measures
Loss = SamplesLoss(loss="sinkhorn", p=2, blur=.05)

Wass_xy  = Loss(tensorX ,tensorY )  # By default, use constant weights = 1/number of samples
g_x, = torch.autograd.grad(Wass_xy , [tensorX]) # GeomLoss fully supports autograd!

end = time.time()
print(
    "Wasserstein distance: {:.3f}, computed in {:.3f}s.".format(
        Wass_xy.item(), end - start
    )
)
print('done')
##
fig4, ax4 = plt.subplots(figsize=set_size(PgWidthPt, fraction=fraction))
# ranInd = np.random.randint(SpecNumMeas)
# ax4.plot(LinDataY[ranInd],tang_heights_lin, linestyle = 'dotted', marker = 'o', label = 'true linear data', markersize = 5 , zorder = 3 )
# relErr = np.linalg.norm(NonLinDataY[ranInd] - RealMap @ LinDataY[ranInd]) / np.linalg.norm( RealMap @ LinDataY[ranInd]) * 100
# print(relErr)
# ax4.plot(RealMap @ LinDataY[ranInd],tang_heights_lin, linestyle = 'dotted', marker = 'o', label = r'map. $\bm{y}$' + f', rel. Err.: {relErr:.1e} \%', markersize = 8, zorder = 1)
# ax4.plot(NonLinDataY[ranInd],tang_heights_lin, linestyle = 'dotted', marker = 'o', label = 'true nonlinear data', markersize = 15, zorder = 0)
LinDat = np.matmul(A_O3 * 2,VMR_O3 * theta_scale_O3).reshape(SpecNumMeas) + noiseVec
ax4.plot(LinDat,tang_heights_lin, linestyle = 'dotted', marker = 'o', label = 'true linear data', markersize = 5 , zorder = 3, color = 'k')
relErr = np.linalg.norm( RealMap @ y.reshape(SpecNumMeas) -  LinDat) / np.linalg.norm(RealMap @ y.reshape(SpecNumMeas)) * 100

print(relErr)
ax4.plot(RealMap @ y.reshape(SpecNumMeas),tang_heights_lin, linestyle = 'dotted', marker = 'o', label = r'map. $\bm{y}$' + f', rel. Err.: {relErr:.1f} \%', markersize = 7, zorder = 1, color ='r')
ax4.plot(y,tang_heights_lin, linestyle = 'dotted', marker = '*', label = 'true data', markersize = 20, zorder = 0, color = DatCol )

ax4.legend()
ax4.set_ylabel('(tangent) height in km')
ax4.set_xlabel(r'spectral radiance in $\frac{\text{W} \text{cm}}{\text{m}^2 \text{sr}} $',labelpad=10)# color =dataCol,
#ax4.tick_params(colors = DatCol, axis = 'x')
ax4.xaxis.set_ticks_position('top')
ax4.xaxis.set_label_position('top')

plt.savefig('MapAssesment.svg')
plt.show()

## plot lin and non lin data to show that it is the same


fig4, ax4 = plt.subplots(figsize=set_size(PgWidthPt, fraction=fraction))
# ranInd = np.random.randint(SpecNumMeas)
for r in range(0,SpecNumMeas):

    ax4.plot(LinDataY[r],tang_heights_lin, linestyle = 'dotted', marker = 'o', label = 'linear data', markersize = 2 , color = 'k', linewidth = 0.1, zorder = 2)
    ax4.plot(RealMap @ NonLinDataY[r],tang_heights_lin, linestyle = 'dotted', marker = 'o', label = 'mapped data', markersize = 5 ,zorder = 1,  color = 'r', linewidth = 0.1)
    ax4.plot(NonLinDataY[r],tang_heights_lin, linestyle = 'dotted', marker = '*', label = 'data', markersize = 15 , zorder = 0, color = DatCol, linewidth = 1)

handles, labels = ax4.get_legend_handles_labels()
legend = ax4.legend(handles = [handles[0], handles[1], handles[2]])# loc='lower right', framealpha = 0.2,fancybox=True)#, bbox_to_anchor=(1.01, 1.01), frameon =True)

ax4.set_ylabel('(tangent) height in km')
ax4.set_xlabel(r'spectral radiance in $\frac{\text{W} \text{cm}}{\text{m}^2 \text{sr}} $',labelpad=10)# color =dataCol,
ax4.xaxis.set_ticks_position('top')
ax4.xaxis.set_label_position('top')

plt.savefig('DataAssesment.svg')
plt.show()



#plot one eaxmaple of map
## for machine learning
# num_epochs = 100
# model = DoMachLearing(num_epochs, NonLinDataY.T, LinDataY.T)
# # evaluate machine model
# # normTestNonLin = ((testNonLinY+noise) - np.mean(NonLinDataY.T, 1)) / np.std(NonLinDataY.T, 1)
# # normTestNonLin_tensor = torch.tensor(normTestNonLin, dtype = torch.float32).view(1,-1)
# #
# # model.eval()
# # with torch.no_grad():
# #     normTestOutput = model(normTestNonLin_tensor)
# # TorchPredic =  np.array(normTestOutput).reshape(SpecNumMeas) * np.std(LinDataY.T, 1) + np.mean(LinDataY.T, 1)
# # relTorchMapErr = np.linalg.norm((testDataY+noise) - TorchPredic) / np.linalg.norm((testDataY+noise)) *100
#
# print('Machine Learning done')




##
lowBoundErr = 2.5
round = 0
meanErr = np.copy(lowBoundErr)
oldMeanErr = np.copy(lowBoundErr)
oldMean = np.copy(firstMean)
while meanErr >= np.copy(lowBoundErr) and round <= 5:

    numDatSet = SpecNumMeas
    deltRes = np.zeros((SampleRounds, 3))
    gamRes = np.zeros((SampleRounds))
    Results = np.zeros((SampleRounds, SpecNumLayers))
    DataSet = np.zeros((SpecNumMeas))

    y_new = RealMap @ np.copy(y)


    gamRes[0] = gamma0
    Results[0] = VMR_O3.reshape(SpecNumLayers)
    deltRes[0] = np.array([30, 1e-7, 0.8e-4])
    SetDelta = Parabel(height_values, *deltRes[0])

    TriU = np.tril(np.triu(np.ones((n, n)), k=1), 1) * SetDelta
    TriL = np.triu(np.tril(np.ones((n, n)), k=-1), -1) * SetDelta.T
    Diag = np.eye(n) * np.sum(TriU + TriL, 0)

    L_d = -TriU + Diag - TriL
    L_d[0, 0] = 2 * L_d[0, 0]
    L_d[-1, -1] = 2 * L_d[-1, -1]


    def MargPostSupp(Params):
        list = []
        list.append(Params[0] > 0)
        list.append(height_values[-1] > Params[1] > height_values[0])
        list.append(Params[2] > 0)
        list.append(Params[3] > 0)
        return all(list)


    A = A_O3 * 2
    #A = RealMap @ np.copy(A)
    ATy = np.matmul(A.T, y_new)
    ATA = np.matmul(A.T, A)

    B0 = (ATA + 1 / gamRes[0] * L_d)
    B_inv_A_trans_y0, exitCode = scy.sparse.linalg.gmres(B0, ATy[:, 0], rtol=tol, restart=25)
    if exitCode != 0:
        print(exitCode)

    log_post = lambda Params: log_postO3(Params, ATA, ATy, height_values, B_inv_A_trans_y0,
                                         Results[0, :].reshape((SpecNumLayers, 1)), y_new, A, gamma0)
    MargPost = pytwalk.pytwalk(n=4, U=log_post, Supp=MargPostSupp)
    x0 = np.array([gamRes[0], *deltRes[0, :]])
    xp0 = 1.0001 * x0
    startTime = time.time()
    MargPost.Run(T=tWalkSampNum + burnIn, x0=x0, xp0=xp0)
    print('time elapsed:' + str(time.time() - startTime))
    Samps = MargPost.Output

    for samp in range(1, SampleRounds):
        MWGRand = burnIn + np.random.randint(low=0, high=tWalkSampNum)
        SetGamma = Samps[MWGRand, 0]
        SetDelta = Parabel(height_values, *Samps[MWGRand, 1:-1])

        TriU = np.tril(np.triu(np.ones((n, n)), k=1), 1) * SetDelta
        TriL = np.triu(np.tril(np.ones((n, n)), k=-1), -1) * SetDelta.T
        Diag = np.eye(n) * np.sum(TriU + TriL, 0)

        L_d = -TriU + Diag - TriL
        L_d[0, 0] = 2 * L_d[0, 0]
        L_d[-1, -1] = 2 * L_d[-1, -1]
        SetB = SetGamma * ATA + L_d

        W = np.random.multivariate_normal(np.zeros(len(A)), np.eye(len(A)))
        v_1 = np.sqrt(SetGamma) * A.T @ W.reshape((m, 1))
        W2 = np.random.multivariate_normal(np.zeros(len(L_d)), L_d)
        v_2 = W2.reshape((n, 1))

        RandX = (SetGamma * ATy + v_1 + v_2)
        O3_Prof, exitCode = scy.sparse.linalg.gmres(SetB, RandX[0::, 0], rtol=tol)
        Results[samp, :] = O3_Prof / theta_scale_O3
        deltRes[samp, :] = np.array([Samps[MWGRand, 1:-1]])
        gamRes[samp] = SetGamma

    newMean = np.mean(Results, 0)

    # fig4, ax4 = plt.subplots(figsize=set_size(PgWidthPt, fraction=fraction))
    # for r in range(0, SampleRounds):
    #     Sol = Results[r, :]
    #
    #     ax4.plot(Sol, height_values, marker='+', color=ResCol, zorder=0, linewidth=0.5, markersize=5, alpha=0.3)
    #
    # ax4.plot(oldMean, height_values, marker='o', color='r')
    # ax4.plot(newMean, height_values, marker='o', color='k')
    # ax4.plot(VMR_O3, height_values, marker='o', color='g')
    # plt.show()
    meanErr = np.linalg.norm(newMean - oldMean) / np.linalg.norm(oldMean) * 100
    oldMean = np.copy(newMean)
    print(f'relErr in between Means: {meanErr} %')

    #if meanErr > oldMeanErr:
    if meanErr < np.copy(lowBoundErr):
        #need to update to old Results
        #print('relative Error increased')
        break
    oldMeanErr = np.copy(meanErr)
    currMap = np.eye(SpecNumMeas)#np.copy(realmap)
    RealMap, relMapErr, LinDataY, NonLinDataY = genDataFindandtestMap(currMap, L_d, gamma0, VMR_O3, Results, AscalConstKmToCm, A_lin, temp_values,  pressure_values, ind, scalingConst, SampleRounds)
    round += 1
    print(round)
##
DatCol =  'gray'
ResCol = "#1E88E5"
TrueCol = [50/255,220/255, 0/255]
fig4, ax4 = plt.subplots(figsize=set_size(PgWidthPt, fraction=fraction))
line3 = ax4.scatter(y, tang_heights_lin, label  = r'data $\bm{y}$', zorder = 0, marker = '*', color =DatCol )#,linewidth = 5

ax1 = ax4.twiny()
for r in range(0, SampleRounds):
    Sol = Results[r, :]

    ax1.plot(Sol, height_values, marker='+', color=ResCol, zorder=0, linewidth=0.5, markersize=5, label = r'$\bm{x} \sim \pi(\bm{x}|\bm{y}, \bm{\theta})$')# alpha=0.3,

ax1.plot(firstMean, height_values, marker='>', color='r',markersize = 10, label =r'initial sample  mean')
ax1.plot(newMean, height_values, marker='>', color='k', label =r'final sample mean')
ax1.plot(VMR_O3, height_values, marker='o', color= TrueCol,markersize = 10, label =r'true $\bm{x}$', zorder = 1)

ax1.set_xlabel(r'ozone volume mixing ratio ')

ax4.set_ylabel('(tangent) height in km')
handles, labels = ax1.get_legend_handles_labels()
handles2, labels2 = ax4.get_legend_handles_labels()
ax1.set_ylim([heights[minInd], heights[maxInd-1]])
ax4.set_xlabel(r'spectral radiance in $\frac{\text{W} \text{cm}}{\text{m}^2 \text{sr}} $',labelpad=10)# color =dataCol,

ax4.tick_params(colors = DatCol, axis = 'x')
ax4.xaxis.set_ticks_position('top')
ax4.xaxis.set_label_position('top')
ax1.xaxis.set_ticks_position('bottom')
ax1.xaxis.set_label_position('bottom')
ax1.spines[:].set_visible(False)
#ax2.spines['top'].set_color(pyTCol)

legend = ax1.legend(handles = [handles[0], handles[-3], handles[-2],handles[-1],handles2[0]])# loc='lower right', framealpha = 0.2,fancybox=True)#, bbox_to_anchor=(1.01, 1.01), frameon =True)

plt.savefig('LinkFuncO3Res.svg')
plt.show()
print('done')

# 3 prozent is good

