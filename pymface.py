import os

from datetime import datetime

import numpy as np
# import numpy.ma as ma

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

import hdf5storage
import numpy.matlib
from scipy.interpolate import InterpolatedUnivariateSpline
import  math

def test():
    print("test")

def get_data_path(filename):
    res = os.path.dirname(os.path.abspath(__file__))
    res = os.path.join(res, filename)
    return res

def get_example_input_data_path():
    return get_data_path('exampleData.mat')

def get_N_path():
    return get_data_path('MFACE_30min_Lag_N_v1.0.mat')

def get_S_path():
    return get_data_path('MFACE_30min_Lag_S_v1.0.mat')

def get_AE_path():
    res = os.path.dirname(os.path.abspath(__file__))
    res = os.path.join(res, 'AEmodel.mat')
    return res


def datenum(d, days_of_year):
    return days_of_year + d.toordinal() + (d - datetime.fromordinal(d.toordinal())).total_seconds()/(24*60*60)




def interp1d(delta_MLat_grid, EOFs, rMLAT_OKlat_):
    N = EOFs.shape[1]
    EOFs_interp = np.zeros((rMLAT_OKlat_.shape[0], N))
    for i in range(0, N):
        ft = InterpolatedUnivariateSpline(np.array(delta_MLat_grid), np.array(EOFs[:, [i]]), k=1)
        EOFs_interp[:, [i]] = ft(rMLAT_OKlat_)

    return EOFs_interp


def x2fx(x, model):
    ###Функция для построения матрицы по модели

    result = np.zeros([x.shape[0]])
    colomn = np.ones(x.shape[0])
    for i in range(0, model.shape[0]):
        for j in range(0, x.shape[1]):
            if (model[i][j] != 0):
                colomn = colomn * x[:, j] ** model[i][j]

        result = np.c_[result, colomn]
        colomn = np.ones(x.shape[0])
    result = result[:, 1:]  

    return result


def MFACE_NS_JUB(d): 
    nargin = len(d)
    MLAT = d['MLAT']
    X_0 = d['X_0']

    if (nargin > 2):
        EOFid = d['EOFid']

    if all(MLAT >= 0):
        tmp = get_N_path()
        #print('1--',tmp)
        FAC_ = hdf5storage.loadmat(tmp);
    elif all(MLAT <= 0):
        tmp = get_S_path()
        #print('2--',tmp)
        FAC_ = hdf5storage.loadmat(tmp);

    J0_taken_into_account = 1

    if (nargin > 2):
        J0_taken_into_account = 0

        if (EOFid == 0):
            J0_taken_into_account = 1

        if (EOFid > 0):
            EOF_single = FAC_['coef_J_EOFs'][:, EOFid - 1]
            FAC_['coef_J_EOFs'] = FAC_['coef_J_EOFs'] * 0
            FAC_['coef_J_EOFs'][:, EOFid - 1] = EOF_single
            J0 = 0
        else:
            FAC_['coef_J_EOFs'] = FAC_['coef_J_EOFs'] * 0
            J0 = 0

    # gernerate interaction and squared terms
    X_2 = x2fx(X_0, FAC_['logical_index_organizing_regressors'])

    X2z = (np.array(X_2) - np.array(np.matlib.repmat(FAC_['average_X'], X_0.shape[0], 1))) / np.array(
        np.matlib.repmat(FAC_['std_X'], X_0.shape[0], 1))

    X2z[:, 0] = 1 

    ACC_MLat = np.dot(X2z, FAC_['coef_ACC_MLat'])
    rMLAT = MLAT - ACC_MLat

    OKlat_ = ((rMLAT > FAC_['delta_MLat_grid'].min()) * (rMLAT < FAC_['delta_MLat_grid'].max()))

    J = np.zeros(OKlat_.shape)
    YR = np.zeros([X2z.shape[0], 12])

    ####################################################################
    if sum(OKlat_) > 0:
        EOFs_interp = interp1d(FAC_['delta_MLat_grid'], FAC_['EOFs'],
                               rMLAT[OKlat_].reshape(rMLAT[OKlat_].shape[0], 1, order='F'))

        #######################################################################################################################
        YR[OKlat_[:, 0], :] = np.dot(X2z[OKlat_[:, 0], :], FAC_['coef_J_EOFs'])
        YR = YR[OKlat_[:, 0] != 0., :]

        J[OKlat_[:, 0] != 0] = np.sum(EOFs_interp * YR, axis=1).reshape(sum(OKlat_)[0], 1)

        if J0_taken_into_account != 0:
            J0 = interp1d(FAC_['delta_MLat_grid'], FAC_['avarage_current'],
                          rMLAT[OKlat_].reshape(rMLAT[OKlat_].shape[0], 1, order='F'))

            J[OKlat_[:, 0] != 0] = J[OKlat_[:, 0] != 0] + J0

    return J.T, ACC_MLat.T


def MFACE_v1(d):
    # Задаём значения по умолчанию

    MLT = d.get('MLT', None)
    MLAT = d.get('MLAT', None)
    doy = d.get('doy', None)

    imfBy = d.get('imfBy', None)
    imfBz = d.get('imfBz', None)
    imfBx = d.get('imfBx', None)
    Vsw = d.get('Vsw', None)
    AE = d.get('AE', None)

    varargin = d;
    nin = len(d)

    if MLT is None:
        raise ValueError("Отсутствует обязательный параметр MLT")
    if MLAT is None:
        raise ValueError("Отсутствует обязательный параметр MLAT")

    if doy is None:
        doy = 81 + MLT * 0;

    if imfBx is None:
        imfBx = MLT * 0 + 3

    if imfBy is None:
        imfBy = MLT * 0 + 3
    if imfBz is None:
        imfBz = MLT * 0 + 3
    if Vsw is None:
        Vsw = 450 + MLT * 0

    doy[~np.isinf(doy)] = 81
    imfBx[np.isinf(imfBx)] = 3
    imfBy[np.isinf(imfBy)] = 0
    imfBz[np.isinf(imfBz)] = 0
    Vsw[np.isinf(Vsw)] = 450

    # 2. подготовка регрессоров
    imfB_scale = np.sqrt(np.power(imfBx.tolist(), 2) + np.power(imfBy.tolist(), 2) + np.power(imfBz.tolist(), 2))
    SIZE = np.array(MLT.shape)
##
    if MLT.shape[1] > 1:
        MLT = MLT.reshape(MLT.shape[0] * MLT.shape[1], 1, order='F')  # по столбцам
    if MLAT.shape[1] > 1:
        MLAT = MLAT.reshape(MLAT.shape[0] * MLAT.shape[1], 1, order='F')
    if doy.shape[1] > 1:
        doy = doy.reshape(doy[0].shape[0] **2, 1, order='F')
    if imfBy.shape[1] > 1:
        imfBy = imfBy.reshape(imfBy[0].shape[0] **2, 1, order='F')
    if imfBz.shape[1] > 1:
        imfBz = imfBz.reshape(imfBz[0].shape[0] **2, 1, order='F')
    if imfB_scale.shape[1] > 1:
        imfB_scale = imfB_scale.reshape(imfB_scale[0].shape[0]**2, 1, order='F')
    if Vsw.shape[1] > 1:
        Vsw = Vsw.reshape(Vsw[0].shape[0]**2, 1, order='F')
###

    MLT_in_radian = MLT / 24 * 2 * np.pi
    IMFB = imfBy + imfBz * 1j

    B_t = abs(IMFB);  # Находим Тангенциальную составляющую IMF в плоскости y-z GSM
    clock_IMF = np.angle(IMFB)
    elev_IMF = (imfBz / imfB_scale);
    for i in range(0, elev_IMF.shape[0]):
        for j in range(0, elev_IMF.shape[1]):
            elev_IMF[i][j] = math.acos(elev_IMF[i][j])

    PotentialI = Vsw ** 2 * 1e-4 + 11.7 * imfB_scale * np.sin(elev_IMF / 2) ** 3
    doy_in_radian = (doy + 10.) / 365.25 * 2 * math.pi
    doy_in_radian = np.mod(doy_in_radian, 2 * math.pi)

    if AE is None:
        AE = np.zeros(MLT.shape)
        try:
            if AE is None:
                raise UnboundLocalError('')
            ae_ = AE[AE == np.nan]
            aegood = AE[AE != np.nan]


        except UnboundLocalError:
            print("Error")

        if (not 'bsint1h' in locals()):
            bsint1h = imfBz * 0
            bsint1h = imfBz * (imfBz < 0)

        else:
            try:
                if (any(bsint1h[bsint1h == np.nan])):
                    bs_ = bsint1h[bsint1h == np.nan]
                    bsgood = bsint1h[bsint1h != np.nan]
                    bsint1h = imfBz * (imfBz < 0)
                    if ('bs_' in locals()):
                        bsint1h[~bs_ == 1] = bsgood

            except UnboundLocalError:
                print("Error")

        if (AE==0).all():
            AE = AE.reshape(MLAT.shape, order='F')  # меняем размерность массива

            bsint1h = bsint1h.reshape(bsint1h.shape[0], 1, order='F')
            X0AE = np.c_[np.sin(doy_in_radian), np.cos(doy_in_radian), np.sin(doy_in_radian * 2), np.cos(doy_in_radian * 2),
                     np.sin(clock_IMF), np.cos(clock_IMF), np.sin(clock_IMF * 2), np.cos(clock_IMF * 2),
                     np.sin(clock_IMF * 3), np.cos(clock_IMF * 3), B_t, PotentialI, bsint1h];
            tmp = get_AE_path()
        #print('ae is ', tmp)
            AEmodel = hdf5storage.loadmat(tmp)
            X0AE = np.array(X0AE)
            X0AE = X0AE.reshape(X0AE.shape[0], X0AE.shape[1], order='F').T
            X0AE = X0AE.T
            X2AE = x2fx(X0AE, AEmodel['modelab'])

            divide = np.matlib.repmat(AEmodel['sigmaX'], X0AE.shape[0], 1)
            divide[divide == 0] = 1
            X2AEz = (np.array(X2AE) - np.matlib.repmat(AEmodel['mX'], X0AE.shape[0], 1)) / divide

            X2AEz[:, 0] = 1

            AE = X2AEz @ AEmodel['b21']

            if (('ae' in locals()) & ('ae_' in locals()) & sum(ae_ == 1) > 0):
                ae[ae_ == 1] = aegood

     
    AE = AE.reshape(MLAT.shape, order='F')  

    X_0 = np.c_[np.sin(MLT_in_radian), np.cos(MLT_in_radian),
                np.sin(MLT_in_radian * 2), np.cos(MLT_in_radian * 2), np.sin(MLT_in_radian * 3),
                np.cos(MLT_in_radian * 3), np.sin(MLT_in_radian * 4), np.cos(MLT_in_radian * 4),
                np.sin(doy_in_radian), np.cos(doy_in_radian), np.sin(doy_in_radian * 2),
                np.cos(doy_in_radian * 2), np.sin(clock_IMF), np.cos(clock_IMF), np.sin(clock_IMF * 2),
                np.cos(clock_IMF * 2), np.sin(clock_IMF * 3), np.cos(clock_IMF * 3), B_t, AE, PotentialI]

    NS_ = MLAT >= 0
    NS_ = np.array(NS_)
    J = np.zeros(NS_.shape)

    nargout = 6

    ACC_MLat = np.zeros(J.shape)

    # northen Hemisphere

    if sum(NS_) > 0:
        temp = np.zeros(X_0.shape[0])
        for i in range(0, X_0.shape[1]):
            temp = np.c_[temp, X_0[:, [i]][NS_]]
        X_0_NS = temp[:, 1:]
        MLAT_NS = MLAT[NS_]
        MLAT_NS = MLAT_NS.reshape(MLAT_NS.shape[0], 1, order='F')

        J, ACC_MLat = MFACE_NS_JUB({'MLAT': MLAT_NS, 'X_0': X_0_NS})

        # J=-J
        if nargout > 2:
            EOF1_NS_, temp = MFACE_NS_JUB({'MLAT': MLAT_NS, 'X_0': X_0_NS, 'EOFid': 1})
            EOF2_NS_, temp = MFACE_NS_JUB({'MLAT': MLAT_NS, 'X_0': X_0_NS, 'EOFid': 2})
            EOF0_NS_, temp = MFACE_NS_JUB({'MLAT': MLAT_NS, 'X_0': X_0_NS, 'EOFid': 0})

    # southern Hemisphere
    NS_ = np.invert(NS_)

    if sum(NS_) > 0:

        temp = np.zeros(X_0.shape[0])
        for i in range(0, X_0.shape[1]):
            temp = np.c_[temp, X_0[:, [i]][NS_]]
        X_0_NS = temp[:, 1:]
        MLAT_NS = MLAT[NS_]
        MLAT_NS = MLAT_NS.reshape(MLAT_NS.shape[0], 1, order='F')

        J, ACC_MLat = MFACE_NS_JUB({'MLAT': MLAT_NS, 'X_0': X_0_NS})

        J = -J
        if nargout > 2:
            EOF1_NS_, temp = MFACE_NS_JUB({'MLAT': MLAT_NS, 'X_0': X_0_NS, 'EOFid': 1})
            EOF2_NS_, temp = MFACE_NS_JUB({'MLAT': MLAT_NS, 'X_0': X_0_NS, 'EOFid': 2})
            EOF0_NS_, temp = MFACE_NS_JUB({'MLAT': MLAT_NS, 'X_0': X_0_NS, 'EOFid': 0})
            EOF1_NS_ = - EOF1_NS_
            EOF2_NS_ = - EOF2_NS_
            EOF0_NS_ = -EOF0_NS_

           

    J = J.reshape(SIZE)

    ACC_MLat = ACC_MLat.reshape(SIZE)

    if nargout > 2:
        EOF1 = EOF1_NS_.reshape(SIZE)
        EOF2 = EOF2_NS_.reshape(SIZE)
        EOF0 = EOF0_NS_.reshape(SIZE)

    return J.T, ACC_MLat.T, EOF1.T, EOF2.T, EOF0.T
