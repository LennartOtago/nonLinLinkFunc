import numpy as np
from functions import *
import matplotlib.pyplot as plt
fraction = 1.5
import time
from scipy import constants, optimize
import matplotlib as mpl
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



dir = '/home/lennartgolks/PycharmProjects/firstModelCheckPhD/'
dir = '/Users/lennart/PycharmProjects/firstModelCheckPhD/'
B_inv_A_trans_y0 = np.loadtxt(dir + 'B_inv_A_trans_y0.txt')
VMR_O3 = np.loadtxt(dir + 'VMR_O3.txt')
VMR_O3 = VMR_O3.reshape((len(VMR_O3), 1))
pressure_values = np.loadtxt(dir + 'pressure_values.txt')
pressure_values = pressure_values.reshape((len(pressure_values), 1))
temp_values = np.loadtxt(dir + 'temp_values.txt')
temp_values = temp_values.reshape((len(temp_values), 1))
height_values = np.loadtxt(dir + 'height_values.txt')
height_values = height_values.reshape((len(height_values),1))
A = np.loadtxt(dir + 'AMat.txt')
A_lin = np.loadtxt(dir + 'ALinMat.txt')
APress = np.loadtxt(dir + 'AP.txt')
ATemp = np.loadtxt(dir + 'AT.txt')
APressTemp = np.loadtxt(dir + 'APT.txt')
gamma0 = np.loadtxt(dir + 'gamma0.txt')
y = np.loadtxt(dir + 'dataY.txt')
L = np.loadtxt(dir + 'GraphLaplacian.txt')
theta_scale_O3 = np.loadtxt(dir + 'theta_scale_O3.txt')
tang_heights_lin = np.loadtxt(dir + 'tan_height_values.txt')
nonLinY = np.loadtxt(dir + 'nonLinDataY.txt')
num_mole = np.loadtxt( dir + 'num_mole.txt')
LineIntScal = np.loadtxt(dir + 'LineIntScal.txt')
S = np.loadtxt(dir + 'S.txt').reshape((909,1))
wvnmbr = np.loadtxt(dir + 'wvnmbr.txt').reshape((909,1))
g_doub_prime = np.loadtxt(dir + 'g_doub_prime.txt').reshape((909,1))
E = np.loadtxt(dir + 'E.txt').reshape((909,1))
nonLinA =  np.loadtxt(dir + 'nonLinA.txt')
A_lin_dx = np.loadtxt(dir + 'A_lin_dx.txt')


AscalConstKmToCm = 1e3
ind = 623
SNR = 60
scalingConst = 1e11
SpecNumMeas = len(nonLinY)
SpecNumLayers = len(VMR_O3)
m = SpecNumMeas
n = SpecNumLayers
numberOfDat = SpecNumMeas
DataY = np.zeros((SpecNumMeas,numberOfDat))
relErr = np.zeros((SpecNumMeas,numberOfDat))
newO3 = np.zeros((SpecNumLayers,numberOfDat))
nonLinY = np.zeros((SpecNumMeas,numberOfDat))
J = np.zeros((SpecNumMeas,SpecNumLayers))
const = np.linspace(0,5e-6,numberOfDat)

# OrgData
A_O3, theta_scale_O3 = composeAforO3(A_lin, temp_values, pressure_values, ind, scalingConst)
nonLinA = calcNonLin(tang_heights_lin, A_lin_dx,  height_values, pressure_values, ind, temp_values, VMR_O3, AscalConstKmToCm, wvnmbr, S, E,g_doub_prime)
OrgData = np.matmul(A_O3 * nonLinA,VMR_O3 * theta_scale_O3)
# OrgData = np.matmul(A_O3 * 2,VMR_O3 * theta_scale_O3)
y , gamma0 = add_noise(OrgData, SNR)

## plot linear and non-linear data

fig4, ax4 = plt.subplots(figsize=set_size(PgWidthPt, fraction=fraction))
ax4.plot(OrgData,  tang_heights_lin,marker= 'o',color = 'r', label = 'non Linear')
ax4.plot(np.matmul(A_O3 * 2,VMR_O3 * theta_scale_O3), tang_heights_lin,label= 'linear' ,marker= 'o',color = 'g')
ax4.legend()
plt.show()





noise = np.random.normal(0, np.sqrt(1 / gamma0), size = OrgData.shape)
relMapErrDat = np.linalg.norm((OrgData + noise) - OrgData) / np.linalg.norm((OrgData + noise)) * 100
fig4, ax4 = plt.subplots(figsize=set_size(PgWidthPt, fraction=fraction))
ax4.plot(OrgData + noise,  tang_heights_lin,marker= 'o',color = 'r', label = 'non Linear')
ax4.plot(np.matmul(A_O3 * 2,VMR_O3 * theta_scale_O3) + noise, tang_heights_lin,label= 'linear' ,marker= 'o',color = 'g')
ax4.legend()
plt.show()

print(relMapErrDat)

#
## initalize affine map with MTC
# genarate
FirstSamp = SpecNumMeas
numDatSet = SpecNumMeas
deltRes = np.zeros(FirstSamp)
gamRes = np.zeros(FirstSamp)
Results = np.zeros((FirstSamp, SpecNumLayers))
DataSet = np.zeros(SpecNumMeas)

tol = 1e-8

tWalkSampNum = 7500
burnIn = 500
# DataSet = y.reshape(SpecNumMeas)
# gamRes[0] = gamma0
Results[0] = VMR_O3.reshape(SpecNumLayers)
# deltRes[0] = np.array([30, 1e-7, 0.8e-4])
# SetDelta = Parabel(height_values, *deltRes[0])
A = A_O3 * 2 #linear
ATy = np.matmul(A.T, y)
ATA = np.matmul(A.T, A)

betaG = 1e-10
betaD = 1e-10

TriU = np.tril(np.triu(np.ones((n, n)), k=1), 1) #* SetDelta
TriL = np.triu(np.tril(np.ones((n, n)), k=-1), -1) #* SetDelta.T
Diag = np.eye(n) * np.sum(TriU + TriL, 0)

L_d = -TriU + Diag - TriL
L_d[0, 0] = 2 * L_d[0, 0]
L_d[-1, -1] = 2 * L_d[-1, -1]
startInd = 40
L_d[startInd::, startInd::] = L_d[startInd::, startInd::] * 5
L_d[startInd, startInd] = -L_d[startInd, startInd-1] - L_d[startInd, startInd+1] #-L[startInd, startInd-2] - L[startInd, startInd+2]
lowC_L = scy.linalg.cholesky(L_d, lower = True)

theta = VMR_O3 * theta_scale_O3
vari = np.zeros((len(theta)-2,1))

for j in range(1,len(theta)-1):
    vari[j-1] = np.var([theta[j-1],theta[j],theta[j+1]])
def MinLogMargPost(params):#, coeff):

    # gamma = params[0]
    # delta = params[1]
    gamma = params[0]
    lamb = params[1]
    if lamb < 0  or gamma < 0:
        return np.nan

    n = SpecNumLayers
    m = SpecNumMeas

    Bp = ATA + lamb * L_d

    LowTri = np.linalg.cholesky(Bp)
    UpTri = LowTri.T
    # check if L L.H = B
    B_inv_A_trans_y = lu_solve(LowTri, UpTri, ATy[0::, 0])

    G = g(A, L_d,  lamb)
    F = f(ATy, y,  B_inv_A_trans_y)

    return -n/2 * np.log(lamb) - (m/2 + 1) * np.log(gamma) + 0.5 * G + 0.5 * gamma * F +  ( betaD *  lamb * gamma + betaG *gamma)

#minimum = optimize.fmin(MargPostU, [5e-5,0.5])
minimum = optimize.fmin(MinLogMargPost, [gamma0,1/gamma0* 1/ np.mean(vari)/15], maxiter = 25)
gamma0 = minimum[0]
lam0 = minimum[1]
print(minimum)
#taylor series arounf lam_0

B = (ATA + lam0 * L_d)

LowTri = np.linalg.cholesky(B)
UpTri = LowTri.T
# check if L L.H = B
B_inv_A_trans_y0 = lu_solve(LowTri, UpTri,  ATy[0::, 0])

B_inv_L = np.zeros(np.shape(B))

for i in range(len(B)):
    LowTri = np.linalg.cholesky(B)
    UpTri = LowTri.T
    B_inv_L[:, i] = lu_solve(LowTri, UpTri,  L_d[:, i])

B_inv_L_2 = np.matmul(B_inv_L, B_inv_L)
B_inv_L_3 = np.matmul(B_inv_L_2, B_inv_L)


f_coeff = np.zeros(3)
g_coeff = np.zeros(3)
f_coeff[0] = np.matmul(np.matmul(ATy[0::, 0].T, B_inv_L), B_inv_A_trans_y0)
f_coeff[1] = -1 * np.matmul(np.matmul(ATy[0::, 0].T, B_inv_L_2), B_inv_A_trans_y0)
f_coeff[2] = 1 * np.matmul(np.matmul(ATy[0::, 0].T,B_inv_L_3) ,B_inv_A_trans_y0)

g_coeff[0] = np.trace(B_inv_L)
g_coeff[1] = -1 / 2 * np.trace(B_inv_L_2)
g_coeff[2] = 1 /6 * np.trace(B_inv_L_3)


number_samples = 10000
burnIn = 100
gamma0 = minimum[0]
lam0 = minimum[1]
f_0 = f(ATy, y, B_inv_A_trans_y0)

startTime = time.time()
lambdas, gammas, k = MHwG(number_samples, burnIn, lam0, gamma0, f_0, f_coeff, g_coeff, betaG, betaD, n, m)
print('time elapsed:' + str(time.time() - startTime))
print('acceptance ratio: ' + str(k/(number_samples+burnIn)))

BinHist = 30
lambHist, lambBinEdges = np.histogram(lambdas, bins= BinHist, density= True)
gamHist, gamBinEdges = np.histogram(gammas, bins= BinHist, density= True)
fig, axs = plt.subplots(2, 1,tight_layout=True,figsize=set_size(PgWidthPt, fraction=fraction) )#, dpi = dpi)
axs[0].bar(gamBinEdges[1::],gamHist*np.diff(gamBinEdges)[0], zorder = 0,width = np.diff(gamBinEdges)[0])#10)
axs[0].set_xlabel(r'the noise precision $\gamma$')
axs[1].bar(lambBinEdges[1::],lambHist*np.diff(lambBinEdges)[0], zorder = 0,width = np.diff(lambBinEdges)[0])#10)
axs[1].set_title(r'$\lambda =\delta / \gamma$, the regularization parameter', fontsize = 12)
#plt.savefig('HistoPlot.png')
plt.show()


deltas = lambdas * gammas
SetGammas = gammas[np.random.randint(low=burnIn, high=len(gammas), size=FirstSamp)]
SetDeltas  = deltas[np.random.randint(low=burnIn, high=len(deltas), size=FirstSamp)]

for p in range(1,FirstSamp):

    SetGamma = SetGammas[p]  # minimum[0]
    SetDelta = SetDeltas[p]  # minimum[1]
    W = np.random.normal(loc=0.0, scale=1.0, size=len(A))  # .reshape((len(A),1))
    v_1 = np.sqrt(SetGamma) * A.T @ W

    W2 = lowC_L @ np.random.normal(loc=0.0, scale=1.0, size=len(L_d))  # .reshape((len(L),1))
    v_2 = np.sqrt(SetDelta) * W2
    SetB = SetGamma * ATA + SetDelta * L_d
    RandX = (SetGamma * ATy[0::, 0] + v_1 + v_2)

    LowTri = np.linalg.cholesky(SetB)
    UpTri = LowTri.T
    O3_Prof = lu_solve(LowTri, UpTri, RandX)


    Results[p, :] = O3_Prof/ theta_scale_O3
    deltRes[p] = SetDelta
    gamRes[p] = SetGamma




IDiag = np.eye(len(B))
PostMeanBinHist = 10
lambHist, lambBinEdges = np.histogram(lambdas[burnIn:], bins=PostMeanBinHist, density=True)
gamHist, gamBinEdges = np.histogram(gammas[burnIn:], bins=PostMeanBinHist, density=True)

print(PostMeanBinHist)
CondResults = np.zeros((PostMeanBinHist, len(B)))

VarB = np.zeros((PostMeanBinHist, len(B), len(B)))
gamInt = np.zeros(PostMeanBinHist)
meanGamInt = np.zeros(PostMeanBinHist)
for i in range(0, PostMeanBinHist):

    SetLambda = lambBinEdges[i] + (lambBinEdges[i + 1] - lambBinEdges[i]) / 2

    currB = ATA + SetLambda * L_d

    LowTri = np.linalg.cholesky(currB)
    UpTri = LowTri.T
    B_inv_A_trans_y = lu_solve(LowTri, UpTri, ATy[0::, 0])
    CondResults[i, :] = B_inv_A_trans_y * lambHist[i] / np.sum(lambHist)
    B_inv = np.zeros(currB.shape)
    # startTime = time.time()
    LowTri = np.linalg.cholesky(currB)
    UpTri = LowTri.T
    for j in range(len(B)):
        B_inv[:, j] = lu_solve(LowTri, UpTri, IDiag[:, j])

    VarB[i] = B_inv *  lambHist[i] / np.sum(lambHist)
    curGam = gamBinEdges[i] + (gamBinEdges[i + 1] - gamBinEdges[i]) / 2

    gamInt[i] = 1 / curGam  * gamHist[i] / np.sum(gamHist)
    meanGamInt[i] = curGam * gamHist[i] / np.sum(gamHist)

newCondMean = scy.integrate.trapezoid(CondResults.T) / theta_scale_O3

CondVar = scy.integrate.trapezoid(gamInt) * scy.integrate.trapezoid(VarB.T) / (theta_scale_O3) ** 2


Results = np.random.multivariate_normal(newCondMean, CondVar,size=FirstSamp)

firstMean = np.mean(Results, 0)
ResCol = "#1E88E5"
fig4, ax4 = plt.subplots(figsize=set_size(PgWidthPt, fraction=fraction))
#for r in range(0,FirstSamp):
    # Sol = Results[r, :]
    #
    # ax4.plot(Sol,height_values,marker= '+',color = ResCol, zorder = 0, linewidth = 0.5, markersize = 5, alpha = 0.3)

ax4.plot(firstMean, height_values,marker= 'o',color = 'r', label = 'sample Mean')
line3 = ax4.errorbar(newCondMean,height_values,  xerr = np.sqrt(np.diag(CondVar)), markeredgecolor ='k', color = 'k' ,zorder=3, marker = '.', markersize =3, linewidth =1, capsize = 3)
ax4.plot(VMR_O3, height_values,marker= 'o',color = 'g', label = 'ground truth')
ax4.legend()
plt.show()

fig4, ax4 = plt.subplots(figsize=set_size(PgWidthPt, fraction=fraction))
for k in range(0,len(Results[:,0])):
    ax4.plot(Results[k],height_values)
plt.show()

## use O3 profiles to make up lin and nonLin data and use for mapping
# test Real Map
DatCol =  'gray'
ResCol = "#1E88E5"
TrueCol = [50/255,220/255, 0/255]
currMap = np.eye(SpecNumMeas)
RealMap, relMapErr, LinDataY, NonLinDataY, testO3 = genDataFindandtestMap(currMap, tang_heights_lin, A_lin_dx,  height_values, gamma0, VMR_O3, Results, AscalConstKmToCm, A_lin, temp_values, pressure_values, ind, scalingConst, FirstSamp, relMapErrDat , wvnmbr, S, E,g_doub_prime)
print(f'Mean rel Error form Map: {np.mean(relMapErr):.2f}')

fig4, ax4 = plt.subplots(figsize=set_size(PgWidthPt, fraction=fraction))
for k in range(0,SpecNumMeas):
    ax4.plot(LinDataY[k], tang_heights_lin, linestyle='dotted', marker='o',
             label=r'linModel', markersize=7, zorder=1, color='r')
    ax4.plot(NonLinDataY[k], tang_heights_lin, linestyle='dotted', marker='*', label='non Linear Model', markersize=20,
             zorder=0, color=DatCol)
h, l = ax4.get_legend_handles_labels()
ax4.legend(h[0:2], l[0:2])
plt.show()


noise = np.random.normal(0, np.sqrt(1 / gamma0), size = OrgData.shape)
linTestDat = np.matmul(A_O3 * 2,VMR_O3 * theta_scale_O3) + noise
MapLinTestDat = np.matmul( RealMap @ (A_O3 * 2),VMR_O3 * theta_scale_O3) + noise
nonLinTestDat = OrgData + noise

fig4, ax4 = plt.subplots(figsize=set_size(PgWidthPt, fraction=fraction))
ax4.plot( linTestDat,tang_heights_lin, linestyle = 'dotted', marker = 'o', label = 'true linear data', markersize = 5 , zorder = 3, color = 'k')
relErr = np.linalg.norm( nonLinTestDat -  MapLinTestDat) / np.linalg.norm(nonLinTestDat) * 100
print(relErr)
ax4.plot(MapLinTestDat,tang_heights_lin, linestyle = 'dotted', marker = 'o', label = r'map. $\bm{y}$' + f', rel. Err.: {relErr:.1f} \%', markersize = 7, zorder = 1, color ='r')
ax4.plot(nonLinTestDat,tang_heights_lin, linestyle = 'dotted', marker = '*', label = 'original data', markersize = 20, zorder = 0, color = DatCol )
ax4.legend()
ax4.set_ylabel('(tangent) height in km')
ax4.set_xlabel(r'spectral radiance in $\frac{\text{W} \text{cm}}{\text{m}^2 \text{sr}} $',labelpad=10)# color =dataCol,
#ax4.tick_params(colors = DatCol, axis = 'x')
ax4.xaxis.set_ticks_position('top')
ax4.xaxis.set_label_position('top')
#plt.savefig('MapAssesment.svg')
plt.show()


## do now with mapped data
y_mapped = RealMap @ y
ATy_mapped = np.matmul(A.T, y_mapped)
B = (ATA + lam0 * L_d)

LowTri = np.linalg.cholesky(B)
UpTri = LowTri.T
# check if L L.H = B
B_inv_A_trans_y0_mapped = lu_solve(LowTri, UpTri,  ATy_mapped[0::, 0])





f_coeff = np.zeros(3)

f_coeff[0] = np.matmul(np.matmul(ATy_mapped[0::, 0].T, B_inv_L), B_inv_A_trans_y0_mapped)
f_coeff[1] = -1 * np.matmul(np.matmul(ATy_mapped[0::, 0].T, B_inv_L_2), B_inv_A_trans_y0_mapped)
f_coeff[2] = 1 * np.matmul(np.matmul(ATy_mapped[0::, 0].T,B_inv_L_3) ,B_inv_A_trans_y0_mapped)




number_samples = 10000
burnIn = 100
gamma0 = minimum[0]
lam0 = minimum[1]
f_0 = f(ATy_mapped, y_mapped, B_inv_A_trans_y0_mapped)

startTime = time.time()
lambdas, gammas, k = MHwG(number_samples, burnIn, lam0, gamma0, f_0, f_coeff, g_coeff, betaG, betaD, n, m)
print('time elapsed:' + str(time.time() - startTime))
print('acceptance ratio: ' + str(k/(number_samples+burnIn)))


BinHist = 30
lambHist, lambBinEdges = np.histogram(lambdas, bins= BinHist, density= True)
gamHist, gamBinEdges = np.histogram(gammas, bins= BinHist, density= True)
fig, axs = plt.subplots(2, 1,tight_layout=True,figsize=set_size(PgWidthPt, fraction=fraction) )#, dpi = dpi)
axs[0].bar(gamBinEdges[1::],gamHist*np.diff(gamBinEdges)[0], zorder = 0,width = np.diff(gamBinEdges)[0])#10)
axs[0].set_xlabel(r'the noise precision $\gamma$')
axs[1].bar(lambBinEdges[1::],lambHist*np.diff(lambBinEdges)[0], zorder = 0,width = np.diff(lambBinEdges)[0])#10)
axs[1].set_title(r'$\lambda =\delta / \gamma$, the regularization parameter', fontsize = 12)
#plt.savefig('HistoPlot.png')
plt.show()


deltas = lambdas * gammas
SetGammas = gammas[np.random.randint(low=burnIn, high=len(gammas), size=FirstSamp)]
SetDeltas  = deltas[np.random.randint(low=burnIn, high=len(deltas), size=FirstSamp)]

for p in range(1,FirstSamp):

    SetGamma = SetGammas[p]  # minimum[0]
    SetDelta = SetDeltas[p]  # minimum[1]
    W = np.random.normal(loc=0.0, scale=1.0, size=len(A))  # .reshape((len(A),1))
    v_1 = np.sqrt(SetGamma) * A.T @ W

    W2 = lowC_L @ np.random.normal(loc=0.0, scale=1.0, size=len(L_d))  # .reshape((len(L),1))
    v_2 = np.sqrt(SetDelta) * W2
    SetB = SetGamma * ATA + SetDelta * L_d
    RandX = (SetGamma * ATy[0::, 0] + v_1 + v_2)

    LowTri = np.linalg.cholesky(SetB)
    UpTri = LowTri.T
    O3_Prof = lu_solve(LowTri, UpTri, RandX)


    Results[p, :] = O3_Prof/ theta_scale_O3
    deltRes[p] = SetDelta
    gamRes[p] = SetGamma


firstMean = np.mean(Results, 0)
ResCol = "#1E88E5"
fig4, ax4 = plt.subplots(figsize=set_size(PgWidthPt, fraction=fraction))
for r in range(0,FirstSamp):
    Sol = Results[r, :]

    ax4.plot(Sol,height_values,marker= '+',color = ResCol, zorder = 0, linewidth = 0.5, markersize = 5, alpha = 0.3)

ax4.plot(firstMean, height_values,marker= 'o',color = 'r', label = 'sample Mean')
ax4.plot(VMR_O3, height_values,marker= 'o',color = 'g', label = 'ground truth')
ax4.legend()
plt.show()

##
import torch
from geomloss import SamplesLoss


testNum = 100
testO3 = np.random.multivariate_normal(VMR_O3.reshape(SpecNumLayers), 1e-12 * L_d, size=testNum)
testO3[testO3 < 0] = 0
testDataY = np.zeros((testNum, SpecNumMeas))
testNonLinY = np.zeros((testNum, SpecNumMeas))

for k in range(0, testNum):
    currO3 = testO3[k]
    noise = 0#np.random.normal(0, np.sqrt(1 / gamma0), SpecNumMeas)
    nonLinA = calcNonLin(tang_heights_lin, A_lin_dx,  height_values, pressure_values, ind, temp_values, VMR_O3, AscalConstKmToCm, wvnmbr, S, E,g_doub_prime)

    testDataY[k] = np.matmul(A_O3 * 2, currO3.reshape((SpecNumLayers, 1)) * theta_scale_O3).reshape(SpecNumMeas) #+ noise
    testNonLinY[k] = np.matmul(A_O3 * nonLinA, currO3.reshape((SpecNumLayers, 1)) * theta_scale_O3).reshape(
        SpecNumMeas) #+ noise

##
from scipy.stats import wasserstein_distance
for k in range(0, SpecNumMeas):
    print(wasserstein_distance(testDataY[:,k], testNonLinY[:,k]))

#
# from scipy.stats import wasserstein_distance
# for k in range(0, SpecNumMeas):
#     print(wasserstein_distance(LinY_norm[:,k], nonLinY_norm[:,k]))
##
nonLinY_norm = ((testNonLinY.T - np.mean(testNonLinY, 1)) / np.std(testNonLinY, 1)).T
LinY_norm = ((testDataY.T - np.mean(testDataY, 1)) / np.std(testDataY, 1)).T
nonLinY_norm = testNonLinY / np.sum(testNonLinY)
LinY_norm = testDataY / np.sum(testDataY)

testNonLinYTens = torch.tensor(np.copy(testNonLinY), requires_grad=True)
testDataYTens = torch.tensor(np.copy(testDataY))

z = torch.cat((testNonLinYTens , testDataYTens))
offset = z.mean(dim=0)
scale = 10 * (z - offset).abs().mean()

tensorX, tensorY = (testNonLinYTens - offset) / scale, (testDataYTens - offset) / scale




#tensorX = torch.tensor(nonLinY_norm, requires_grad=True)
#tensorY = torch.tensor(LinY_norm)
start = time.time()
# Define a Sinkhorn (~Wasserstein) loss between sampled measures
Loss = SamplesLoss(loss="sinkhorn", p=2, blur=.05, scaling = 0.9)

Wass_xy  = Loss(tensorX ,tensorY )  # By default, use constant weights = 1/number of samples
g_x, = torch.autograd.grad(Wass_xy , [tensorX]) # GeomLoss fully supports autograd!

end = time.time()
print(
    "Wasserstein distance: {:.5e}, computed in {:.5f}s.".format(
        Wass_xy.item(), end - start
    )
)
##

print('done')

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

print('done')


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

