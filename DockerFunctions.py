import numpy as np
import scipy as scy

def MinLogMargPost(params, A, L, ATy, ATA, y, betaG, betaD):

    # gamma = params[0]
    # delta = params[1]
    gamma = params[0]
    lamb = params[1]
    if lamb < 0  or gamma < 0:
        return np.nan

    n = len(A)
    m = len(y)

    Bp = ATA + lamb * L

    LowTri = np.linalg.cholesky(Bp)
    UpTri = LowTri.T
    B_inv_A_trans_y = lu_solve(LowTri, UpTri, ATy[0::, 0])

    G = g(A, L,  lamb)
    F = f(ATy, y,  B_inv_A_trans_y)

    return -n/2 * np.log(lamb) - (m/2 + 1) * np.log(gamma) + 0.5 * G + 0.5 * gamma * F +  ( betaD *  lamb * gamma + betaG *gamma)



def LogMargPost(indices, univarGrid, A, lam0, f_coeff, g_coeff, f_0, g_0, betaD, betaG):
    Values = np.zeros(len(indices))

    for j in range(len(indices)):
        gamma = univarGrid[0][indices[j,0].astype(np.int32)]
        lamb = univarGrid[1][indices[j,1].astype(np.int32)]



        if lamb < 0  or gamma < 0:
            return np.nan

        m, n = A.shape

        f_0_1 = f_coeff[0]
        f_0_2 = f_coeff[1]
        f_0_3 = f_coeff[2]
        g_0_1 = g_coeff[0]
        g_0_2 = g_coeff[1]
        g_0_3 = g_coeff[2]
        #Bp = ATA + lamb * L

        delta_lam = lamb - lam0
        F = f_0 + f_0_1 * delta_lam + f_0_2 * delta_lam**2 + f_0_3 * delta_lam**3 #+ f_0_4 * delta_lam**4 + f_0_5 * delta_lam**5
        G = g_0 + g_0_1 * delta_lam + g_0_2 * delta_lam**2 + g_0_3 * delta_lam**3 #+ g_0_4 * delta_lam**4 + g_0_5 * delta_lam**5

        Values[j] =+1000 + n/2 * np.log(lamb) + (m/2 + 1) * np.log(gamma) - 0.5 * G - 0.5 * gamma * F -  ( betaD *  lamb * gamma + betaG *gamma)

    return np.exp(Values)



def g(A, L, l):
    """ calculate g"""
    B = np.matmul(A.T, A) + l * L
    upL = scy.linalg.cholesky(B)

    return 2 * np.sum(np.log(np.diag(upL)))

def f(ATy, y, B_inv_A_trans_y):
    return np.matmul(y[0::, 0].T, y[0::, 0]) - np.matmul(ATy[0::, 0].T, B_inv_A_trans_y)

def forward_substitution(L, b):
    # Get number of rows
    n = L.shape[0]

    # Allocating space for the solution vector
    y = np.zeros_like(b, dtype=np.double);

    # Here we perform the forward-substitution.
    # Initializing  with the first row.
    y[0] = b[0] / L[0, 0]

    # Looping over rows in reverse (from the bottom  up),
    # starting with the second to last row, because  the
    # last row solve was completed in the last step.
    for i in range(1, n):
        y[i] = (b[i] - np.dot(L[i, :i], y[:i])) / L[i, i]

    return y

def back_substitution(U, y):
    # Number of rows
    n = U.shape[0]

    # Allocating space for the solution vector
    x = np.zeros_like(y, dtype=np.double)

    # Here we perform the back-substitution.
    # Initializing with the last row.
    x[-1] = y[-1] / U[-1, -1]

    # Looping over rows in reverse (from the bottom up),
    # starting with the second to last row, because the
    # last row solve was completed in the last step.
    for i in range(n - 2, -1, -1):
        x[i] = (y[i] - np.dot(U[i, i:], x[i:])) / U[i, i]

    return x


def lu_solve(L, U, b):

    y = forward_substitution(L, b)

    return back_substitution(U, y)

def genDataFindandtestMap(currMap, gamma0, A_O3, theta_scale_O3, Results, AscalConstKmToCm, A_lin, temp_values, pressure_values, ind,SampleRounds, relMapErrDat, num_mole, LineIntScal):
    '''Find map from non-linear to linear data'''

    SpecNumMeas, SpecNumLayers = A_lin.shape
    relMapErr = relMapErrDat
    while np.mean(relMapErr) >= relMapErrDat:
        testDat = SpecNumMeas

        # Results = np.zeros((SampleRounds, SpecNumLayers))
        LinDataY = np.zeros((testDat, SpecNumMeas))
        NonLinDataY = np.zeros((testDat, SpecNumMeas))
        for test in range(testDat):
            ProfRand = np.random.randint(low=0, high=SampleRounds)
            # Results = np.loadtxt(f'Res{testSet}.txt')

            O3_Prof = Results[ProfRand]
            O3_Prof[O3_Prof < 0] = 0
            nonLinA = calcNonLin(A_lin, pressure_values, ind, temp_values,
                                 O3_Prof.reshape((SpecNumLayers, 1)), AscalConstKmToCm, num_mole, LineIntScal, S)
            noise = np.random.normal(0, np.sqrt(1 / gamma0), SpecNumMeas)
            # noise = np.zeros(SpecNumMeas)

            LinDataY[test] = np.matmul(A_O3 * 2, O3_Prof.reshape((SpecNumLayers, 1)) * theta_scale_O3).reshape(
                SpecNumMeas) + noise
            NonLinDataY[test] = np.matmul(A_O3 * nonLinA,
                                          O3_Prof.reshape((SpecNumLayers, 1)) * theta_scale_O3).reshape(
                SpecNumMeas) + noise
            #currMap = np.eye(SpecNumMeas)
            NonLinDataY[test] = currMap @ NonLinDataY[test]
        RealMap = LinModelSolve(LinDataY, NonLinDataY, SpecNumMeas)


        #test map do find error
        testNum = len(Results)#100
        testO3 = Results#np.random.multivariate_normal(VMR_O3.reshape(SpecNumLayers), 1e-18 * L_d, size=testNum)
        testO3[testO3 < 0] = 0

        testDataY = np.zeros((testNum, SpecNumMeas))
        testNonLinY = np.zeros((testNum, SpecNumMeas))

        for k in range(0, testNum):
            currO3 = testO3[k]
            noise = np.random.normal(0, np.sqrt(1 / gamma0), SpecNumMeas)
            nonLinA = calcNonLin(A_lin, pressure_values, ind, temp_values, currO3.reshape((SpecNumLayers, 1)),
                                 AscalConstKmToCm,num_mole, LineIntScal)

            testDataY[k] = np.matmul(A_O3 * 2, currO3.reshape((SpecNumLayers, 1)) * theta_scale_O3).reshape(SpecNumMeas) + noise
            testNonLinY[k] = np.matmul(A_O3 * nonLinA, currO3.reshape((SpecNumLayers, 1)) * theta_scale_O3).reshape(
                SpecNumMeas) + noise
            testNonLinY[k] = currMap @ testNonLinY[k]

        relMapErr = testSolvedMap(RealMap, gamma0, testNum, SpecNumMeas, testNonLinY, testDataY)
        print(np.mean(relMapErr))
        currMap = RealMap @ np.copy(currMap)
    return currMap, relMapErr, LinDataY, NonLinDataY, testO3

def calcNonLin(A_lin, pressure_values, ind, temp_values, VMR_O3, AscalConstKmToCm, wvnmbr, S, E,g_doub_prime):
    '''careful that A_lin is just dx values
    maybe do A_lin_copy = np.copy(A_lin/2)
    A_lin_copy[:,-1] = A_lin_copy[:,-1] * 2
    if A_lin has been generated for linear data'''
    A_lin = A_lin / 2
    A_lin[:, -1] = A_lin[:, -1] * 2
    SpecNumMeas, SpecNumLayers = np.shape(A_lin)
    temp = temp_values.reshape((SpecNumLayers, 1))
    # wvnmbr = np.loadtxt('wvnmbr.txt').reshape((909,1))
    # S = np.loadtxt('S.txt').reshape((909,1))
    # E = np.loadtxt('E.txt').reshape((909,1))
    # g_doub_prime = np.loadtxt('g_doub_prime.txt').reshape((909,1))

    # from : https://hitran.org/docs/definitions-and-units/
    HitrConst2 = 1.4387769  # in cm K
    v_0 = wvnmbr[ind][0]

    f_broad = 1
    #scalingConst = 1e11
    Q = g_doub_prime[ind, 0] * np.exp(- HitrConst2 * E[ind, 0] / temp)
    Q_ref = g_doub_prime[ind, 0] * np.exp(- HitrConst2 * E[ind, 0] / 296)
    LineIntScal = Q_ref / Q * np.exp(- HitrConst2 * E[ind, 0] / temp) / np.exp(- HitrConst2 * E[ind, 0] / 296) * (
                1 - np.exp(- HitrConst2 * wvnmbr[ind, 0] / temp)) / (
                              1 - np.exp(- HitrConst2 * wvnmbr[ind, 0] / 296))



    # take linear
    num_mole = 1 / (constants.Boltzmann)

    theta = num_mole * f_broad * 1e-4 * VMR_O3.reshape((SpecNumLayers,1)) * S[ind,0]
    ConcVal = - pressure_values.reshape((SpecNumLayers, 1)) * 1e2 * LineIntScal / temp_values * theta * AscalConstKmToCm


    mask = A_lin * np.ones((SpecNumMeas, SpecNumLayers))
    mask[mask != 0] = 1
    preTrans = np.zeros((SpecNumMeas, SpecNumLayers))


    for i in range(0,SpecNumMeas):
        for j in range(0, SpecNumLayers-1):
            if mask[i,j] !=0 :
                currMask = np.copy(mask[i, :])
                currMask[j] = 0.5
                currMask[-1] = 0.5
                ValPerLayPre = np.sum(ConcVal.T * currMask * A_lin[i,:])
                preTrans[i,j] = np.exp(ValPerLayPre)
        preTrans[i, -1] = 1
    afterTrans = np.zeros((SpecNumMeas, SpecNumLayers))
    for i in range(0,SpecNumMeas):
        for j in range(0, SpecNumLayers):
            if mask[i,j] !=0 :
                currMask1 = np.copy(mask[i, :])
                currMask1[-1] = 0.5
                currMask2 = np.copy(mask[i, :j+1])
                currMask2[-1] = 0.5
                ValPerLayAfter = np.sum(ConcVal.T * currMask1 * A_lin[i,:]) + np.sum(ConcVal[:j+1].T * currMask2 * A_lin[i,:j+1])
                afterTrans[i,j] = np.exp(ValPerLayAfter)


    return preTrans + afterTrans

def LinModelSolve(NonLinDataY, LinDataY, SpecNumMeas):
    RealMap = None
    while RealMap is None:
        try:

            RealMap = np.zeros((SpecNumMeas, SpecNumMeas))

            for i in range(0, SpecNumMeas):
                RealMap[i] = np.linalg.solve(LinDataY, NonLinDataY[:, i])

        except np.linalg.LinAlgError:
            RealMap = None
            print('pass')
            pass

    return RealMap

def testSolvedMap(RealMap, gamma0, testNum, SpecNumMeas, testNonLinY, testDataY):
    relMapErr = np.zeros(testNum)
    for k in range(0, testNum):

        noise = np.random.multivariate_normal(np.zeros(SpecNumMeas), np.sqrt(1 / gamma0) * np.eye(SpecNumMeas))

        mappedDat = RealMap @ (testNonLinY[k] + noise)
        relMapErr[k] = np.linalg.norm((testDataY[k] + noise) - mappedDat) / np.linalg.norm((testDataY[k] + noise)) * 100


    return relMapErr


def sampleFromTTwithoutIntGrid(numberOfSamp, seeds, maxRank, univarGrid, TTCore):
    '''
    do inverse rosenblatt transport IRT sampling on unnormalized function represented through
    see algorithmic box in Approximation and sampling of multivariate probability distributions in the tensor train decomposition
    https://doi.org/10.1007/s11222-019-09910-z

    :param numberOfSamp:
    :param maxRank:
    :param IntGrid:
    :param univarGrid:
    :param TTCore:
    :return:
    '''
    dim = len(univarGrid)
    gridSize = len(univarGrid[0])
    VecP = np.zeros((dim + 1, maxRank))
    Integral = np.zeros((dim, maxRank, maxRank))
    VecP[dim, 0] = 1

    for k in range(dim - 1, 0, -1):
        r_k, d, r_kpls1 = np.shape(TTCore[k])

        Integral = TTCore[k][:,0, :] * 0.5 * (univarGrid[k][1] - univarGrid[k][0])
        for i in range(1, gridSize-1):
            Integral = np.copy(Integral) + (univarGrid[k][i+1] - univarGrid[k][i]) * TTCore[k][:,i,:]

        Integral = np.copy(Integral)  + TTCore[k][:,-1,:] * 0.5 * (univarGrid[k][-1] - univarGrid[k][-2])


        VecP[k, 0:r_k] = (Integral.reshape((r_k, r_kpls1)) @ VecP[k + 1, 0:r_kpls1].reshape(
            (r_kpls1, 1))).reshape((1, r_k))


    indices = np.zeros((dim,numberOfSamp))
    MargPDF = np.zeros((dim, gridSize))
    MargCDF = np.zeros((dim, gridSize))
    Sample = np.zeros((dim, numberOfSamp))
    SampleMargPDF = np.zeros((dim, numberOfSamp))
    Phi = np.zeros((dim + 1, numberOfSamp, maxRank))
    Psi = np.zeros((dim, gridSize, maxRank))
    r_k, d, r_kpls1 = TTCore[0].shape
    Phi[0][0:numberOfSamp, 0:r_k] = np.ones((numberOfSamp, r_k))

    for k in range(0, dim):

        r_k, d, r_kpls1 = TTCore[k].shape

        for i in range(0, gridSize):
            Psi[k][i, 0:r_k] = (TTCore[k][:,i, :] @ VecP[k + 1, 0:r_kpls1].reshape((r_kpls1, 1))).reshape((1, r_k))

        for l in range(0, numberOfSamp):
            #margPDF
            for i in range(0,gridSize):
                MargPDF[k][i] = abs(Phi[k][l, 0:r_k] @ Psi[k][i, 0:r_k])

            wholeMarg = scy.integrate.trapezoid(MargPDF[k], x = univarGrid[k] )
            MargCDF[k] = scy.integrate.cumulative_trapezoid(y=MargPDF[k], x=univarGrid[k], initial=0)/wholeMarg

            minInd = np.argmin((MargCDF[k] - seeds[k][l]) ** 2)
            Sample[k][l] = np.interp(seeds[k][l], MargCDF[k], univarGrid[k])
            SampleMargPDF[k][l] = np.interp(Sample[k][l], univarGrid[k], MargPDF[k])

            if Sample[k][l] <= univarGrid[k][minInd]:

                minInd = minInd - 1
            # if minInd <= 100:
            #     TTCore[k][:, minInd-1, :]

            indices[k, l] = np.copy(minInd)
            PDFSampApprox = ((Sample[k][l] - univarGrid[k][minInd]) / (
                    univarGrid[k][minInd + 1] - univarGrid[k][minInd])) * TTCore[k][:, minInd+ 1, :] + (
                                                            (univarGrid[k][minInd + 1] - Sample[k][l]) / (
                                                            univarGrid[k][minInd + 1] - univarGrid[k][minInd])) * TTCore[k][:, minInd, :]


            Phi[k + 1][l, 0:r_kpls1] = Phi[k][l, 0:r_k].reshape((1, r_k)) @ PDFSampApprox.reshape((r_k, r_kpls1))
    return SampleMargPDF, Sample, indices