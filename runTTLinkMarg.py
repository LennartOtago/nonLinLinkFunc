from DockerFunctions import *
import time

tang_heights_lin = np.loadtxt('tan_height_values.txt')
height_values = np.loadtxt('height_values.txt')
A = np.loadtxt('AMat.txt')
gamma0 = np.loadtxt('gamma0.txt')
y = np.loadtxt('dataY.txt')

y = y.reshape((len(y),1))
ATA = np.matmul(A.T,A)
ATy = np.matmul(A.T, y)
B_inv_A_trans_y0 = np.loadtxt('B_inv_A_trans_y0.txt')
VMR_O3 = np.loadtxt('VMR_O3.txt')
pressure_values = np.loadtxt('pressure_values.txt')
L = np.loadtxt('GraphLaplacian.txt')
theta_scale_O3 = np.loadtxt('theta_scale_O3.txt')
num_mole = np.loadtxt( 'num_mole.txt')
LineIntScal = np.loadtxt('LineIntScal.txt')
S = np.loadtxt('S.txt').reshape((909,1))
A_lin = np.loadtxt('ALinMat.txt',)

temp_values = np.loadtxt('temp_values.txt',)

AscalConstKmToCm = 1e3
ind = 623
f_broad = 1
scalingConst = 1e11

theta = VMR_O3 * theta_scale_O3
vari = np.zeros((len(theta)-2,1))
for j in range(1,len(theta)-1):
    vari[j-1] = np.var([theta[j-1],theta[j],theta[j+1]])

Ax = np.matmul(A, theta_scale_O3 * VMR_O3)

# import matplotlib.pyplot as plt
# fig3, ax1 = plt.subplots()
# ax1.plot(Ax, tang_heights_lin)
# ax1.scatter(y, tang_heights_lin)
# ax1.plot(y, tang_heights_lin)
# plt.show()
betaG = 1e-10  # 2e-9
betaD = 1e-10

MinFunc = lambda params: MinLogMargPost(params, A, L, ATy, ATA, y, betaG , betaD)
minimum = scy.optimize.fmin(MinFunc, [gamma0,1/gamma0* 1/ np.mean(vari)/15], maxiter = 25)
lam0 = minimum[1]
print(minimum)
# prepare for fast calculation with taylor expansion
B = (ATA + lam0 * L)

LowTri = np.linalg.cholesky(B)
UpTri = LowTri.T
B_inv_A_trans_y0 = lu_solve(LowTri, UpTri,  ATy[0::, 0])

B_inv_L = np.zeros(np.shape(B))

for i in range(len(B)):
    LowTri = np.linalg.cholesky(B)
    UpTri = LowTri.T
    B_inv_L[:, i] = lu_solve(LowTri, UpTri,  L[:, i])



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


f_0 = f(ATy, y,  B_inv_A_trans_y0)
g_0 = g(A, L,lam0)


gridSize = 100
factor = 4
univarGridO3 = [np.linspace(0.5e-7, 3.5e-7, gridSize),
             np.linspace(100, 3000, gridSize)]




ttFunc = lambda indices: LogMargPost(indices, univarGridO3, A, lam0, f_coeff, g_coeff, f_0, g_0, betaD, betaG)

##
import tt
import tt.cross
from tt.cross.rectcross import rect_cross
initRank = 2

dim = 2


# Run cross
#random inital cores
f0 = tt.rand(gridSize, dim, r=initRank)
ttTrain = rect_cross.cross(ttFunc, f0, nswp=8, kickrank=1, rf=2, eps=1e-5)
'''
    :param myfun: Vectorized function of d variables (it accepts I x D integer array as an input, and produces I numbers as output)
    :type myfun: function handle
'''
print(ttTrain)



TTCore = [None] * dim
maxRank = 1

# Cores of f must be extracted carefully, since we might have discontinuous ps
core = np.zeros((ttTrain.core).size, dtype=np.float64)
ps_my = 0
for i in range(0, ttTrain.d):
    cri = ttTrain.core[range(ttTrain.ps[i] - 1, ttTrain.ps[i + 1] - 1)]
    #print(cri)
    np.savetxt('ttTraincoreMargO3'+str(i)+'.txt', cri, header = str(ttTrain.r[i])+ ' ,'+ str( ttTrain.n[i])+ ',' +str( ttTrain.r[i + 1]) )
    TTCore[i] = cri.reshape((ttTrain.r[i] , ttTrain.n[i] , ttTrain.r[i + 1]))
    core[range(ps_my, ps_my + ttTrain.r[i] * ttTrain.n[i] * ttTrain.r[i + 1])] = cri
    ps_my = ps_my + ttTrain.r[i] * ttTrain.n[i] * ttTrain.r[i + 1]
    np.savetxt('uniVarGridMargO3' + str(i) + '.txt', univarGridO3[i])
    if int(ttTrain.r[i + 1]) > maxRank:
        maxRank = int(ttTrain.r[i + 1])

## find affine map
dim = len(univarGridO3)
gridSize = len(univarGridO3[0])

numberOfSamp = 1000
seeds = np.random.uniform(size=(dim, numberOfSamp))
seeds = [np.linspace(0,1,numberOfSamp+2 )[1:-1],
         np.linspace(0,1,numberOfSamp+2 )[1:-1]]

startTime = time.time()

FuncSampleMargPDF, FuncSample, SeedIndVal = sampleFromTTwithoutIntGrid(numberOfSamp, seeds, maxRank, univarGridO3, TTCore)
print("elapsedtime:"+ str( time.time()-startTime))

FuncW = np.zeros(numberOfSamp)
for i in range(numberOfSamp):
    FuncW[i] = LogMargPost(FuncSample[:,i])

weights =  np.exp(FuncW - np.sum(np.log(FuncSampleMargPDF),0))

EstMeanX = np.zeros(dim)
Z = np.sum(weights)
for k in range(0, dim):
    EstMeanX[k] = np.sum(FuncSample[k] * weights) / Z

# CovMat = np.zeros((dim,dim))
# for k in range(0, dim):
#     for k1 in range(0, dim):
#         CovMat[k1,k] = np.sum((FuncSample[k] - EstMeanX[k])*(FuncSample[k1]- EstMeanX[k1]) * weights) / Z
#Sampl = np.random.multivariate_normal(EstMeanX, CovMat, size = )

PostMeanBinHist = 10
lambHist, lambBinEdges = np.histogram(FuncSample[1] - EstMeanX[1], bins=PostMeanBinHist, density=True)
gamHist, gamBinEdges = np.histogram(FuncSample[0]- EstMeanX[0], bins=PostMeanBinHist, density=True)

IDiag = np.eye(len(ATA))
MargResults = np.zeros((PostMeanBinHist, len(theta)))
MargVarResults = np.zeros((PostMeanBinHist, len(theta)))
B_inv_Res = np.zeros((PostMeanBinHist,len(theta)))


B_inv = np.zeros((PostMeanBinHist, np.shape(B)[0], np.shape(ATA)[0]))
VarB = np.zeros((PostMeanBinHist, np.shape(ATA)[0], np.shape(ATA)[0]))
gamInt = np.zeros(PostMeanBinHist)
for p in range(PostMeanBinHist):

    SetLambda = lambBinEdges[p] + (lambBinEdges[p + 1] - lambBinEdges[p]) / 2

    SetB = ATA + SetLambda * L

    LowTri = np.linalg.cholesky(SetB)
    UpTri = LowTri.T
    B_inv_A_trans_y = lu_solve(LowTri, UpTri, ATy[0::, 0])

    MargResults[p, :] = B_inv_A_trans_y * lambHist[p] / np.sum(lambHist)
    B_inv_Res[p, :] = B_inv_A_trans_y
    for i in range(len(SetB)):
        LowTri = np.linalg.cholesky(SetB)
        UpTri = LowTri.T
        B_inv[p, :, i] = lu_solve(LowTri, UpTri, IDiag[:, i])

    VarB[p] = B_inv[p] * lambHist[p] / np.sum(lambHist)
    curGam = gamBinEdges[p] + (gamBinEdges[p + 1] - gamBinEdges[p]) / 2

newMargInteg = scy.integrate.trapezoid(MargResults.T)/ (
            num_mole * S[ind, 0] * f_broad * 1e-4 * scalingConst) ** 2

MargVar = scy.integrate.trapezoid(gamInt) * scy.integrate.trapezoid(VarB.T) / (
            num_mole * S[ind, 0] * f_broad * 1e-4 * scalingConst) ** 2

FirstSamp = len(y)
relMapErrDat = 4
Results = np.random.multivariate_normal(newMargInteg, MargVar,size=FirstSamp)
currMap = np.eye(len(y))
RealMap, relMapErr, LinDataY, NonLinDataY, testO3 = genDataFindandtestMap(currMap, gamma0, A, theta_scale_O3, Results, AscalConstKmToCm, A_lin, temp_values,
                      pressure_values, ind, FirstSamp, relMapErrDat, num_mole, LineIntScal,)

print(f'Mean rel Error form Map: {np.mean(relMapErr):.2f}')