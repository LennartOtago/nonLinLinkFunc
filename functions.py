import numpy as np
import scipy as scy
from scipy import constants

def get_temp_values(height_values):
    """ used to be based on the ISA model see omnicalculator.com/physics/altitude-temperature
    now https://www.grc.nasa.gov/www/k-12/airplane/atmosmet.html """
    temp_values = np.zeros(len(height_values))
    #temp_values[0] = 288.15#15 - (height_values[0] - 0) * 6.49 + 273.15
    #calculate temp values
    for i in range(0, len(height_values)):
        if height_values[i] <= 11:
            temp_values[i] = np.around(- height_values[i] * 6.49 + 288.15,2)
        if 11 < height_values[i] <= 25:
            temp_values[i] = np.around(216.76,2)
        if 20 < height_values[i] <= 32:
            temp_values[i] = np.around(216.76 + (height_values[i] - 20) * 1, 2)
        if 32 < height_values[i] <= 47:
            temp_values[i] = np.around(228.76 + (height_values[i] - 32) * 2.8, 2)
        if 47 < height_values[i] <= 51:
            temp_values[i] = np.around(270.76, 2)
        if 51 < height_values[i] <= 71:
            temp_values[i] = np.around(270.76 - (height_values[i] - 51) * 2.8, 2)
        if 71 < height_values[i] <= 85:
            temp_values[i] = np.around(214.76 - (height_values[i] - 71) * 2.0, 2)
        if 85 < height_values[i]:
            temp_values[i] = 186.8

    return temp_values.reshape((len(height_values),1))

def gen_sing_map(meas_ang, height, obs_height, R):
    tang_height = np.around((np.sin(meas_ang) * (obs_height + R)) - R, 2)
    num_meas = len(tang_height)
    # add one extra layer so that the difference in height can be calculated
    layers = np.zeros((len(height)+1,1))
    layers[0:-1] = height
    layers[-1] = height[-1] + (height[-1] - height[-2])/2


    A_height = np.zeros((num_meas, len(layers)-1))
    t = 0
    for m in range(0, num_meas):

        while layers[t] <= tang_height[m]:

            t += 1

        # first dr
        A_height[m, t - 1] = 0.5 * np.sqrt((layers[t] + R) ** 2 - (tang_height[m] + R) ** 2)
        dr = 2 * A_height[m, t - 1]
        for i in range(t, len(layers) - 1):
            A_height[m, i] = np.sqrt((layers[i + 1] + R) ** 2 - (tang_height[m] + R) ** 2) - dr
            dr = dr + A_height[m, i]
        A_height[m, i] = 0.5 * A_height[m, i]
    #return 2 * (A_...) for linear part
    return  A_height, tang_height, layers[-1]


def composeAforO3(A_lin, temp, press, ind, scalingConst):

    SpecNumMeas, SpecNumLayers = np.shape(A_lin)
    temp = temp.reshape((SpecNumLayers, 1))
    wvnmbr = np.loadtxt('wvnmbr.txt').reshape((909,1))
    S = np.loadtxt('S.txt').reshape((909,1))
    F = np.loadtxt('F.txt').reshape((909,1))
    g_air = np.loadtxt('g_air.txt').reshape((909,1))
    g_self = np.loadtxt('g_self.txt').reshape((909,1))
    E = np.loadtxt('E.txt').reshape((909,1))
    n_air = np.loadtxt('n_air.txt').reshape((909,1))
    g_doub_prime = np.loadtxt('g_doub_prime.txt').reshape((909,1))

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

    C1 = 2 * constants.h * constants.c ** 2 * v_0 ** 3 * 1e8
    C2 = constants.h * constants.c * 1e2 * v_0 / (constants.Boltzmann * temp)
    # plancks function
    Source = np.array(C1 / (np.exp(C2) - 1))

    # take linear
    num_mole = 1 / (constants.Boltzmann)  # * temp_values)

    AscalConstKmToCm = 1e3

    # 1e2 for pressure values from hPa to Pa
    A_scal = press.reshape((SpecNumLayers, 1)) * 1e2 * LineIntScal * Source * AscalConstKmToCm / (temp)
    theta_scale = num_mole *  f_broad * 1e-4 * scalingConst * S[ind, 0]
    A = A_lin * A_scal.T
    #np.savetxt('AMat.txt', A, fmt='%.15f', delimiter='\t')
    return A, theta_scale



def add_noise(signal, snr_percent):
    """
    Add noise to a signal based on the specified SNR (in percent).

    Parameters:
        signal: numpy array
            The original signal.
        snr_percent: float
            The desired signal-to-noise ratio in percent.

    Returns:
        numpy array
            The signal with added noise.
    """
    # Calculate signal power
    signal_power = np.mean(np.abs(signal) ** 2)

    # Calculate noise power based on SNR (in percent)
    #noise_power = signal_power / (100 / snr_percent)
    noise_power = signal_power / snr_percent

    # Generate noise
    noise = np.random.normal(0, np.sqrt(noise_power), signal.shape)

    # Add noise to the signal
    noisy_signal = signal + noise

    return noisy_signal, 1/noise_power


def calcNonLin(A_lin, pressure_values, ind, temp_values, VMR_O3, AscalConstKmToCm, SpecNumLayers, SpecNumMeas):


    SpecNumMeas, SpecNumLayers = np.shape(A_lin)
    temp = temp_values.reshape((SpecNumLayers, 1))
    wvnmbr = np.loadtxt('wvnmbr.txt').reshape((909,1))
    S = np.loadtxt('S.txt').reshape((909,1))
    F = np.loadtxt('F.txt').reshape((909,1))
    g_air = np.loadtxt('g_air.txt').reshape((909,1))
    g_self = np.loadtxt('g_self.txt').reshape((909,1))
    E = np.loadtxt('E.txt').reshape((909,1))
    n_air = np.loadtxt('n_air.txt').reshape((909,1))
    g_doub_prime = np.loadtxt('g_doub_prime.txt').reshape((909,1))

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

    C1 = 2 * constants.h * constants.c ** 2 * v_0 ** 3 * 1e8
    C2 = constants.h * constants.c * 1e2 * v_0 / (constants.Boltzmann * temp)
    # plancks function
    Source = np.array(C1 / (np.exp(C2) - 1))

    # take linear
    num_mole = 1 / (constants.Boltzmann)

    theta = num_mole * f_broad * 1e-4 * VMR_O3.reshape((SpecNumLayers,1)) * S[ind,0]
    ConcVal = - pressure_values.reshape(
        (SpecNumLayers, 1)) * 1e2 * LineIntScal / temp_values * theta * AscalConstKmToCm


    mask = A_lin * np.ones((SpecNumMeas, SpecNumLayers))
    mask[mask != 0] = 1

    ConcValMat = np.tril(np.ones(len(ConcVal))) * ConcVal
    ValPerLayPre = mask * (A_lin @ ConcValMat)
    preTrans = np.exp(ValPerLayPre)

    ConcValMatAft = np.triu(np.ones(len(ConcVal))) * ConcVal
    ValPerLayAft = mask * (A_lin @ ConcValMatAft)
    BasPreTrans = (A_lin @ ConcValMat)[0::, 0].reshape((SpecNumMeas, 1)) @ np.ones((1, SpecNumLayers))
    afterTrans = np.exp( (BasPreTrans + ValPerLayAft) * mask )


    return preTrans + afterTrans

def Parabel(x, h0, a0, d0):
    return a0 * np.power((h0 - x), 2) + d0
def g(A, L, l):
    """ calculate g"""
    B = np.matmul(A.T, A) + l * L
    # Bu, Bs, Bvh = np.linalg.svd(B)
    upL = scy.linalg.cholesky(B)
    # return np.sum(np.log(Bs))
    return 2 * np.sum(np.log(np.diag(upL)))

def f(ATy, y, B_inv_A_trans_y):
    return np.matmul(y[0::, 0].T, y[0::, 0]) - np.matmul(ATy[0::, 0].T, B_inv_A_trans_y)


def log_postO3(Params, ATA, ATy, height_values, B_inv_A_trans_y0, VMR_O3, y, A, gamma0):
    tol = 1e-8
    #n = len(height_values)
    #m = len(y)

    m, n = np.shape(A)
    gam = Params[0]
    h1 = Params[1]
    a0 = Params[2]
    d0 = Params[3]

    delta = Parabel(height_values,h1, a0, d0)
    TriU = np.tril(np.triu(np.ones((n, n)), k=1), 1) * delta
    TriL = np.triu(np.tril(np.ones((n, n)), k=-1), -1) * delta.T
    Diag = np.eye(n) * np.sum(TriU + TriL, 0)

    L_d = -TriU + Diag - TriL
    L_d[0, 0] = 2 * L_d[0, 0]
    L_d[-1, -1] = 2 * L_d[-1, -1]
    try:
        upTriL = scy.linalg.cholesky(L_d)
        detL = 2 * np.sum(np.log(np.diag(upTriL)))
    except scy.linalg.LinAlgError:
        try:
            L_du, L_ds, L_dvh = np.linalg.svd(L_d)
            detL = np.sum(np.log(L_ds))
        except np.linalg.LinAlgError:
            print("SVD did not converge, use scipy.linalg.det()")
            detL = np.log(scy.linalg.det(L_d))

    Bp = ATA + 1/gam * L_d

    B_inv_A_trans_y, exitCode = scy.sparse.linalg.gmres(Bp, ATy[:,0], x0 = B_inv_A_trans_y0, atol=tol, restart=25)
    if exitCode != 0:
        print(exitCode)

    try:
        G = g(A, L_d,  1/gam)
    except np.linalg.LinAlgError:
        return - np.inf

    F = f(ATy, y,  B_inv_A_trans_y)

    hMean = height_values[VMR_O3[:] == np.max(VMR_O3[:])]

    d0Mean =0.8e-4


    dataVal = - (m / 2 - n / 2) * np.log(gam) - 0.5 * detL + 0.5 * G + 0.5 * gam * F
    Value =  dataVal + 0.5 * ((d0 - d0Mean) / (0.75e-5)) ** 2 + 0.5 * ((gam - gamma0) / (gamma0 * 0.01)) ** 2 + 0.5 * ((h1 - hMean) / 1) ** 2 - 2* np.log(a0) + 1e6 * a0
    return Value[0]

def DoMachLearing(num_epochs, nonLinY, DataY):
    import torch
    import torch.nn as nn
    import torch.optim as optim
    SpecNumMeas , numberOfDat = nonLinY.shape
    print(nonLinY.shape)
    nonLinY_norm = ((nonLinY.T - np.mean(nonLinY, 1)) / np.std(nonLinY, 1)).T
    LinY_norm = ((DataY.T - np.mean(DataY, 1)) / np.std(DataY, 1)).T

    x_tensor = torch.tensor(nonLinY_norm, dtype=torch.float32)
    y_tensor = torch.tensor(LinY_norm, dtype=torch.float32)

    class LinearModel(nn.Module):
        def __init__(self, in_features, out_features):
            super().__init__()
            self.linear = nn.Linear(in_features, out_features)

        def forward(self, x):
            return self.linear(x).squeeze(1)

    in_features = SpecNumMeas
    out_features = SpecNumMeas

    model = LinearModel(in_features, out_features)

    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=1)

    outputs = model(x_tensor.T)
    # calculate Los
    loss = criterion(outputs, y_tensor.T)

    # backwardpass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    prevLoss = np.round(loss.item(),4)

    outputs = model(x_tensor.T)
    # calculate Los
    loss = criterion(outputs, y_tensor.T)

    # backwardpass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    currLoss = np.round(loss.item(),4)
    epoch = 1
    #for epoch in range(num_epochs):
    while prevLoss > currLoss:
        # forward pass
        outputs = model(x_tensor.T)

        # calculate Los
        loss = criterion(outputs, y_tensor.T)

        # backwardpass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        prevLoss = np.copy(currLoss)
        currLoss = np.round(loss.item(),4)
        epoch += 1

        print(f'Epoch[ {epoch + 1}], Loss: {loss.item():.4f}')

    return model


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

def genDataFindandtestMap(currMap, L_d, gamma0, VMR_O3, Results, AscalConstKmToCm, A_lin, temp_values, pressure_values, ind, scalingConst,SampleRounds):
    SpecNumMeas, SpecNumLayers = A_lin.shape
    relMapErr = 1
    while np.mean(relMapErr) >= 1:
        testDat = SpecNumMeas
        A_O3, theta_scale_O3 = composeAforO3(A_lin, temp_values, pressure_values, ind, scalingConst)

        # Results = np.zeros((SampleRounds, SpecNumLayers))
        LinDataY = np.zeros((testDat, SpecNumMeas))
        NonLinDataY = np.zeros((testDat, SpecNumMeas))
        for test in range(testDat):
            ProfRand = np.random.randint(low=0, high=SampleRounds)
            # Results = np.loadtxt(f'Res{testSet}.txt')

            O3_Prof = Results[ProfRand]
            O3_Prof[O3_Prof < 0] = 0
            nonLinA = calcNonLin(A_lin, pressure_values, ind, temp_values,
                                 O3_Prof.reshape((SpecNumLayers, 1)), AscalConstKmToCm,
                                 SpecNumLayers, SpecNumMeas)
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

        testNum = 100
        testO3 = np.random.multivariate_normal(VMR_O3.reshape(SpecNumLayers), 1e-12 * L_d, size=testNum)
        testO3[testO3 < 0] = 0
        testDataY = np.zeros((testNum, SpecNumMeas))
        testNonLinY = np.zeros((testNum, SpecNumMeas))

        for k in range(0, testNum):
            currO3 = testO3[k]
            nonLinA = calcNonLin(A_lin, pressure_values, ind, temp_values, currO3.reshape((SpecNumLayers, 1)),
                                 AscalConstKmToCm,
                                 SpecNumLayers, SpecNumMeas)

            testDataY[k] = np.matmul(A_O3 * 2, currO3.reshape((SpecNumLayers, 1)) * theta_scale_O3).reshape(SpecNumMeas)
            testNonLinY[k] = np.matmul(A_O3 * nonLinA, currO3.reshape((SpecNumLayers, 1)) * theta_scale_O3).reshape(
                SpecNumMeas)
            testNonLinY[k] = currMap @ testNonLinY[k]

        relMapErr = testSolvedMap(RealMap, gamma0, testNum, SpecNumMeas, testNonLinY, testDataY)

    return RealMap @ currMap, relMapErr, LinDataY, NonLinDataY
# def testMachMap(Map):


def set_size(width, fraction=1):
    """Set figure dimensions to avoid scaling in LaTeX.

    Parameters
    ----------
    width: float
            Document textwidth or columnwidth in pts
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy

    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    # Width of figure (in pts)
    fig_width_pt = width * fraction

    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    # https://disq.us/p/2940ij3
    golden_ratio = 1#(5**.5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * golden_ratio

    fig_dim = (fig_width_in, fig_height_in)

    return fig_dim