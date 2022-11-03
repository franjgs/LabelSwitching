#!/usr/bin/python
                                                                                
import time
import random
import numpy as np
import math
import sys

import copy

import scipy.io as sio
from scipy.interpolate import interp1d
from scipy import interp

from glob import glob

from sklearn import metrics
from sklearn.metrics import f1_score, roc_curve, auc, roc_auc_score
from sklearn.metrics import matthews_corrcoef, balanced_accuracy_score

from sklearn.preprocessing import StandardScaler, MinMaxScaler

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression, Lasso, ElasticNet

from sklearn.manifold import MDS
from sklearn.decomposition import PCA

import torch
from bigmoe import BigMoE, compute_weights, f1_loss
from bigmoe import weighted_mse_fit, weighted_bce_fit, f1_fit, TPR_fit
from bigmoe import train_LBFGS

from imblearn.over_sampling import RandomOverSampler, SMOTE, BorderlineSMOTE, KMeansSMOTE, ADASYN
from imblearn.under_sampling import ClusterCentroids, RandomUnderSampler, NearMiss

from scipy.stats import norm, multivariate_normal

import statsmodels.api as sm

eps=np.finfo(float).eps


###################### COMMON PARAMETERS ######################
# 2, 4, 8, 12, 16
n_folds = 1
Nruns = 20
nh = 4
Test_size = 25
K_ens= [31]


verbose = False

Mode = 'Asymmetric'

Alpha = np.linspace(0.0,0.45,10) # [::-1]      # Label Switching Rate from Majority to Minority class
Alpha = [0.3]                         # Label Switching Rate from Majority to Minority class
Beta = np.linspace(0,0.2,5)              # Label Switching Rate from Minority to Majority class
Beta = [0.0]                                 # Label Switching Rate from Minority to Majority class

# Parámetro que te permite decidir si puerta y expertos se entrenan con los mismos
# datos (False) o si se hace un resample distinto de x_train para cada uno (True)
resample = False
Clusters = False

Niter_EM = 5   # Expectation-Maximization Loops
v_ini = True     # True: 1-step EM Maximization + IRLS
lamda_ini = 5e-2
Niter_IRLS = 0
if Niter_IRLS > 0:
    IRLS = True
Bias=1  
Convex_comb = True # False: input-independent aggregation
if Convex_comb == False:
    cX = 0
    file_suffix = '_wp_p1_cc'
else:
    cX = 1
    file_suffix = '_wp_p1'
    
mode_IRLS = 'lasso' # 'en' # 'mlr' #  'rr'     # 'lasso' and lamabda_en 1e-3 bien en "balance"
lamda_IRLS = 1e-2

pyX = 'Norm' # 'BCE' # 'Norm'
pyX_sig='Auto'
pyX_sig=0.5    # Solarflare: pyX_sig = 1
    
v_sig=0.01
f_sel = f1_score    # TPR_fit #  weighted_mse_fit # balanced_accuracy_score #  f1_fit # matthews_corrcoef # weighted_bce_fit #   matthews_corrcoef # 
validate = True # True    # True: 1/3 of Training Dataset used to select the best Gate Coefficients
full_Train = True  # True: Experts are trained with the full Training Dataset  

print('Ensemble: K_ens: %d. nh: %d. Niter_EM = %d' % (K_ens[0] , nh, Niter_EM))

Q_RB_S = 0 # SIEMPRE uno de estos dos tiene que tener corchetes
Q_RB_C = [0]

Weight_MoE = 2 # Este es el peso extra que tiene la clase minoritaria al entrenar la MoE
                    # Solarflare : Weight_MoE = 4
                    # Abalone 9-18: Wieght_MoE = 9

RB = 0
if Q_RB_S == 0:
    RB = Q_RB_C
else:
    RB = Q_RB_S


############################## FUNCTIONS ##############################

def label_switching(y, alphasw=0.0, betasw=0.0):

    # alphasw: Label Switching Rate from Majority to Minority class
    # betasw: Label Switching Rate from Minority to Majority class

    ysw=1*y;

    idx1=np.where(y==+1)[0]
    l1=len(idx1)
    bet_1=int(round(l1*betasw))
    idx1_sw=np.random.choice(idx1,bet_1, replace=False)
    ysw[idx1_sw]=-y[idx1_sw]

    idx0=np.where(y==-1)[0]
    l0=len(idx0)
    alph_0=int(round(l0*alphasw))
    idx0_sw=np.random.choice(idx0,alph_0, replace=False)
    ysw[idx0_sw]=-y[idx0_sw]

    return ysw


def breiman_switching(y, frsw=0.0):

    # frsw: Label Switching Rate from Minority to Majority class

    ysw=1*y;

    idx1=np.where(y==+1)[0]
    idx0=np.where(y==-1)[0]

    n_flipped=int(round(len(idx1)*frsw))

    idx1_sw=np.random.choice(idx1, n_flipped, replace=False)
    ysw[idx1_sw]=-y[idx1_sw]

    idx0_sw=np.random.choice(idx0, n_flipped, replace=False)
    ysw[idx0_sw]=-y[idx0_sw]

    return ysw


def perf_measure(y_actual, y_hat):

    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(y_hat)):
        if y_actual[i]==y_hat[i]:
            if y_actual[i]==1:
                TP += 1
            else:
                TN += 1
        else:
            if y_actual[i]==1:
                FN += 1
            else:
                FP += 1

    return TP, FP, TN, FN


def QCT_PFAT(y_true, o_x,alpha,beta,pFA):

    P=len(np.where(y_true==+1)[0])
    N=len(np.where(y_true==-1)[0])

    B=sorted(range(len(o_x)),key=lambda x:o_x[x],reverse=True)

    o_x_sorted=o_x[B]
    y_sorted=y_true[B]
    idx=0
    UT=2
    FP=0
    pFAT=0

    while pFAT<=pFA and idx < (P+N):
        if UT != o_x_sorted[idx]:
            pFAT=((FP+1)/float(N))
            UT=o_x_sorted[idx]
        if y_sorted[idx]==-1:
            FP+=1
        idx+=1
    pFAT=FP/float(N)

    QCT=(UT+1-2*alpha)/(1-2*beta-UT)

    return QCT, UT


def ROC_points(L, f):

    """
    Calculates the value of the discriminant function for a dx1 dimensional
    sample given covariance matrix and mean vector.
    """

    FPR=[]
    TPR=[]
    P=len(np.where(L==+1)[0])
    N=len(np.where(L==-1)[0])

    B=sorted(range(len(f)),key=lambda x:f[x],reverse=True)

    L_sorted=L[B]
    FP=0
    TP=0
    f_prev=1e45
    i=0

    while i < len(L_sorted):
        if f[i] !=f_prev:
            FPR.append(FP/float(N))
            TPR.append(TP/float(P))
            f_prev=f[i]
        if L_sorted[i]==+1:
            TP+=1
        else:
            FP+=1
        i+=1
        
    FPR.append(FP/float(N))
    TPR.append(TP/float(P))
    
    return FPR, TPR

def center_outputs(outputs, alpha, beta, hard=None):
    centered_outputs = torch.zeros_like(outputs) # Outputs corrigiendo el SW

    if isinstance(alpha, np.ndarray):
        for i in range(len(alpha)):
            centered_outputs[:,i] = (outputs[:,i]-alpha[i]+beta)/((1-alpha[i]-beta))
    else:
        for j in range(outputs.shape[1]):
            centered_outputs[:,j] = (outputs[:,j]-alpha+beta)/((1-alpha-beta))
    if hard==True:
        return torch.sign(centered_outputs)
        # return torch.tanh(centered_outputs/((1-alpha[i]-beta)**10))
    else:
        return centered_outputs
    
def logsumexp(x):
    y = torch.max(x, dim=1)[0]
    return y+torch.log(torch.exp(torch.sub(x, y[:, None])).sum(dim=1))

def convergenceTest(fval, previous_fval, threshold=1e-3, warn=False):
# Check if an objective function has converged
# 
# We have converged if the slope of the function falls below 'threshold',
# i.e., |f(t) - f(t-1)| / avg < threshold,
# where avg = (|f(t)| + |f(t-1)|)/23
# 'threshold' defaults to 1e-4.
# This stopping criterion is from Numerical Recipes in C p423
# This file is from pmtk3.googlecode.com

    converged = 0;
    delta_fval = torch.abs(fval - previous_fval)
    avg_fval = (torch.abs(fval) + torch.abs(previous_fval) + eps)/2
    if all(delta_fval / avg_fval < threshold):
        converged = 1;
    
    if warn and (fval-previous_fval) < -2*eps:  # fval < previous_fval
        print('convergenceTest:fvalDecrease', 'objective decreased!')
    
    return converged

############################## FUNCTIONS ##############################

######################## Datos ImbLearn #######################################

"""
ID 	Name 	        Repository & Target 	         Ratio 	#S 	     #F
1 	ecoli 	        UCI, target: imU 	             8.6:1 	336 	 7
2 	optical_digits 	UCI, target: 8 	                 9.1:1 	5,620 	 64
3 	satimage 	    UCI, target: 4 	                 9.3:1 	6,435 	 36
4 	pen_digits 	    UCI, target: 5 	                 9.4:1 	10,992 	 16
5 	abalone 	    UCI, target: 7 	                 9.7:1 	4,177 	 10
6 	sick_euthyroid 	UCI, target: sick euthyroid 	 9.8:1 	3,163 	 42
7 	spectrometer 	UCI, target: >=44 	             11:1 	531 	 93
8 	car_eval_34 	UCI, target: good, v good 	     12:1 	1,728 	 21
9 	isolet 	        UCI, target: A, B 	             12:1 	7,797 	 617
10 	us_crime 	    UCI, target: >0.65 	             12:1 	1,994 	 100
11 	yeast_ml8 	    LIBSVM, target: 8 	             13:1 	2,417 	 103
12 	scene 	        LIBSVM, target: >one label 	     13:1 	2,407 	 294
13 	libras_move 	UCI, target: 1 	                 14:1 	360 	 90
14 	thyroid_sick 	UCI, target: sick 	             15:1 	3,772 	 52
15 	coil_2000 	    KDD, CoIL, target: minority 	 16:1 	9,822 	 85
16 	arrhythmia 	    UCI, target: 06 	             17:1 	452 	 278
17 	solar_flare_m0 	UCI, target: M->0 	             19:1 	1,389 	 32
18 	oil 	        UCI, target: minority 	         22:1 	937 	 49
19 	car_eval_4 	    UCI, target: vgood 	             26:1 	1,728 	 21
20 	wine_quality 	UCI, wine, target: <=4 	         26:1 	4,898 	 11
21 	letter_img 	    UCI, target: Z 	                 26:1 	20,000 	 16
22 	yeast_me2 	    UCI, target: ME2 	             28:1 	1,484 	 8
23 	webpage 	    LIBSVM, w7a, target: minority 	 33:1 	34,780 	 300
24 	ozone_level 	UCI, ozone, data 	             34:1 	2,536 	 72
25 	mammography 	UCI, target: minority 	         42:1 	11,183 	 6
26 	protein_homo 	KDD CUP 2004, minority 	         111:1 	145,751  74
27 	abalone_19 	    UCI, target: 19 	             130:1 	4,177 	 10
"""

"""
fnames = 'data/'+filename_i+'.npz'
data = np.load(fnames)

X=data['data']
y=data['label']

dim = X.shape[1]
"""
    
filenames_in = ['ecoli', 'satimage', 'abalone', 'Ringnorm10', 
                'balance', 'ecoli4', 'abalone9-18', 'solar_flare_m0',
                'Ringnorm20', 'oil', 'flare-F', 'wine_quality', 'letter_img',
                'yeast4', 'abalone17vs7-8-9-10', 'Ringnorm40', 'yeast6']  #     'winequality-red-4' # 'ozone_level' # 'protein_homo' #  'yeast_me2' # 'yeast_ml8' # 'car_eval_34' # 'page-blocks0' # 'yeast-2_vs_4' # 'glass4' # 'abalone_19' #   'poker-8-9_vs_6' # 'winequality-white-3-9_vs_5' #  'vowel0' # 'thyroid_sick' # 'us_crime' #  'spectrometer' # 'scene' # 'sick_euthyroid' # 'optical_digits' # 'isolet' # 'pen_digits' # 'page-blocks-1-3_vs_4' # 'mammography' #  'yeast3' # 'yeast1' # 'libras_move' #  'car_eval_4' #  'glass-0-1-6_vs_5' # 'glass5' # 'led7digit-0-2-4-5-6-7-8-9_vs_1' #    # 'coil_2000' #  'abalone17vs7-8-9-10' #   'yeast_me2' # 'us_crime' #  'oil' # 'glass4' # 'car_eval_34' #  'yeast5' 
    


    
#######################################################################
#######################################################################
#######################################################################
    
################################ CODE #################################


    
# Fase de entrenamiento
for filename_i in filenames_in:
    
    ############################## DATA LOAD ##############################
    path_prefix = '../data/'+filename_i+'/'
    
    if filename_i.startswith('Ringnorm'):
        dim = 20
        mean1 = ((2/math.sqrt(dim))*np.ones(dim))
        cov1 = (np.identity(dim))
        mean0 = np.zeros(dim)
        cov0 = (4*np.identity(dim)) 
        N1 = 150
        
        if filename_i == 'Ringnorm10':
            N0 = 10*N1
        elif filename_i =='Ringnorm20':
            N0 = 20*N1
        elif filename_i =='Ringnorm40':
            N0 = 40*N1
            
        ####### DATOS SINTETICOS #######
        
        X0 = np.random.multivariate_normal(mean0, cov0, N0)
        y0 = -np.ones(N0)
        X1 = np.random.multivariate_normal(mean1, cov1, N1)
        y1 = np.ones(N1)

        X_o = np.append(X0,X1,axis=0)
        y_o = np.append(y0,y1,axis=0)

        idx_random = np.random.choice(N0 + N1, N0 + N1, replace=False)
        X = X_o[idx_random]
        y = y_o[idx_random]
    elif filename_i == 'yeast1' or filename_i == 'yeast3' or filename_i == 'yeast4' \
            or filename_i == 'yeast5' or filename_i == 'yeast6':
        fnames = path_prefix+filename_i+'.dat'
        d_x = 8
        
        classes = {b' negative': -1, b' positive': +1}
        
        Xy = np.loadtxt(fnames, skiprows = 0, delimiter = ",", comments = "@", converters = {d_x: lambda x: classes[x]})
        
        X=Xy[:,range(d_x)]
        y=Xy[:,d_x]
        
        dim = X.shape[1]
    elif filename_i == 'yeast-2_vs_4':
        fnames = path_prefix+filename_i+'.dat'
        d_x = 8
        
        classes = {b'negative': -1, b'positive': +1}
        
        Xy = np.loadtxt(fnames, skiprows = 0, delimiter = ",", comments = "@", converters = {d_x: lambda x: classes[x]})
        
        X=Xy[:,range(d_x)]
        y=Xy[:,d_x]
        
        dim = X.shape[1]  
    elif filename_i == 'abalone9-18' or filename_i == 'abalone17vs7-8-9-10' or filename_i == 'abalone19' or filename_i == 'abalone-19_vs_10-11-12-13' or filename_i == 'abalone-20_vs_8-9-10':
        fnames = path_prefix+filename_i+'.dat'
        d_x = 8
        
        classes = {b' negative': -1, b' positive': +1}
        
        gender = {b'M': -1, b'F': +1, b'I': 0}
        
        Xy = np.loadtxt(fnames, skiprows = 0, delimiter = ",", comments = "@", converters = {0: lambda x: gender[x], d_x: lambda x: classes[x]})
        
        X=Xy[:,range(d_x)]
        y=Xy[:,d_x]
        
        dim = X.shape[1]
    elif filename_i == 'balance':
        fnames = path_prefix+filename_i+'.dat'
        d_x = 4
        
        classes = {b' B': +1, b' R': -1, b' L': -1}
        
        #  Old Xy = np.loadtxt(fnames, skiprows = 0, delimiter = ", ", comments = "@", converters = {d_x: lambda x: classes[x]})
        Xy = np.loadtxt(fnames, skiprows = 0, delimiter = ",", comments = "@", converters = {d_x: lambda x: classes[x]})
        
        X=Xy[:,range(d_x)]
        y=Xy[:,d_x]
        
        dim = X.shape[1]
    elif filename_i == 'page-blocks-1-3_vs_4':
        fnames = path_prefix+filename_i+'.dat'
        d_x = 10
        
        classes = {b'negative': -1, b'positive': +1}
        
        Xy = np.loadtxt(fnames, skiprows = 0, delimiter = ",", comments = "@", converters = {d_x: lambda x: classes[x]})
        
        X=Xy[:,range(d_x)]
        y=Xy[:,d_x]
        
        dim = X.shape[1]
    elif filename_i == 'page-blocks0':
        fnames = path_prefix+filename_i+'.dat'
        d_x = 10
        
        classes = {b' negative': -1, b' positive': +1}
        
        Xy = np.loadtxt(fnames, skiprows = 0, delimiter = ",", comments = "@", converters = {d_x: lambda x: classes[x]})
        
        X=Xy[:,range(d_x)]
        y=Xy[:,d_x]
        
        dim = X.shape[1] 
    elif filename_i == 'glass4' or filename_i == 'glass5':
        fnames = path_prefix+filename_i+'.dat'
        d_x = 9
        
        classes = {b'positive': +1, b'negative': -1}
        
        Xy = np.loadtxt(fnames, skiprows = 0, delimiter = ", ", comments = "@", converters = {d_x: lambda x: classes[x]})
        
        X=Xy[:,range(d_x)]
        y=Xy[:,d_x]
        
        dim = X.shape[1]
    elif filename_i == 'poker-8_vs_6' or filename_i == 'poker-8-9_vs_6' or filename_i == 'poker-8-9_vs_5':
        fnames = path_prefix+filename_i+'.dat'
        d_x = 10
        
        classes = {b'positive': +1, b'negative': -1}
        
        Xy = np.loadtxt(fnames, skiprows = 0, delimiter = ",", comments = "@", converters = {d_x: lambda x: classes[x]})
        
        X=Xy[:,range(d_x)]
        y=Xy[:,d_x]
        
        dim = X.shape[1]
    elif filename_i == 'winequality-white-3-9_vs_5' or filename_i == 'winequality-red-4':
        fnames = path_prefix+filename_i+'.dat'
        d_x = 11
        
        classes = {b'positive': +1, b'negative': -1}
        
        Xy = np.loadtxt(fnames, skiprows = 0, delimiter = ",", comments = "@", converters = {d_x: lambda x: classes[x]})
        
        X=Xy[:,range(d_x)]
        y=Xy[:,d_x]
        
        dim = X.shape[1]
    elif filename_i == 'kr-vs-k-zero_vs_eight' or filename_i == 'kr-vs-k-zero_vs_fifteen':
        fnames = path_prefix+filename_i+'.dat'
        d_x = 6
        
        classes = {b'positive': +1, b'negative': -1}
        Position = {b'a': 1, b'b': 2, b'c': 3, b'd': 4, b'e': 5, b'f': 6, b'g': 7, b'h': 8}
        
        Xy = np.loadtxt(fnames, skiprows = 0, delimiter = ",", comments = "@", converters = {0: lambda r: Position[r], 2: lambda s: Position[s], 4: lambda t: Position[t], d_x: lambda x: classes[x]})
        
        X=Xy[:,range(d_x)]
        y=Xy[:,d_x]
        
        dim = X.shape[1]
    elif filename_i == 'glass-0-1-6_vs_5' or filename_i == 'shuttle-2_vs_5':
        fnames = path_prefix+filename_i+'.dat'
        d_x = 9
        
        classes = {b'positive': +1, b'negative': -1}
        
        Xy = np.loadtxt(fnames, skiprows = 0, delimiter = ",", comments = "@", converters = {d_x: lambda x: classes[x]})
        
        X=Xy[:,range(d_x)]
        y=Xy[:,d_x]
        
        dim = X.shape[1]
    elif filename_i == 'ecoli4' or filename_i=='ecoli3':
        fnames = path_prefix+filename_i+'.dat'
        d_x = 7
        
        classes = {b' positive': +1, b' negative': -1}
        
        Xy = np.loadtxt(fnames, skiprows = 0, delimiter = ",", comments = "@", converters = {d_x: lambda x: classes[x]})
        
        X=Xy[:,range(d_x)]
        y=Xy[:,d_x]
        
        dim = X.shape[1]
    elif filename_i == 'flare-F':
        fnames = path_prefix+filename_i+'.dat'
        d_x=11
        
        classes = {b' negative': -1, b' positive': +1}
        LargestSpotSize = {b'A': 0, b'R': 1, b'S': 2, b'X': 3, b'K': 4, b'H': 5}
        SpotDistribution = {b' X': 0, b' O': 1, b' I': 2, b' C': 3}
        
        Xy = np.loadtxt(fnames, skiprows = 0, delimiter = ",", comments = "@", converters = {0: lambda s: LargestSpotSize[s], 1: lambda q: SpotDistribution[q], d_x: lambda x: classes[x]})
        X=Xy[:,range(d_x)]
        y=Xy[:,d_x]
        
        dim = X.shape[1]
    elif filename_i == 'led7digit-0-2-4-5-6-7-8-9_vs_1':
        fnames = path_prefix+filename_i+'.dat'
        d_x = 7
        
        classes = {b'positive': +1, b'negative': -1}
        
        Xy = np.loadtxt(fnames, skiprows = 0, delimiter = ",", comments = "@", converters = {d_x: lambda x: classes[x]})
        
        X=Xy[:,range(d_x)]
        y=Xy[:,d_x]
        
        dim = X.shape[1]
    elif filename_i == 'vowel0':
        fnames = path_prefix+filename_i+'.dat'
        d_x = 13
        
        classes = {b'positive': +1, b'negative': -1}
        
        Xy = np.loadtxt(fnames, skiprows = 0, delimiter = ", ", comments = "@", converters = {d_x: lambda x: classes[x]})
        
        X=Xy[:,range(d_x)]
        y=Xy[:,d_x]
        
        dim = X.shape[1]
    elif filename_i == 'satimage' or filename_i == 'ecoli' or filename_i == 'webpage' \
            or filename_i == 'optical_digits'  or filename_i == 'pen_digits' \
            or filename_i == 'oil' or filename_i == 'solar_flare_m0' \
            or filename_i == 'arrhythmia' or filename_i == 'letter_img' \
            or filename_i == 'wine_quality' or filename_i ==  'abalone' \
            or filename_i == 'car_eval_34' or filename_i == 'sick_euthyroid' \
            or filename_i == 'coil_2000' or filename_i == 'us_crime' \
            or filename_i == 'libras_move' or filename_i == 'car_eval_4' \
            or filename_i == 'yeast_me2'  or filename_i == 'yeast_ml8' \
            or filename_i == 'ozone_level'  or filename_i == 'spectrometer' \
            or filename_i == 'protein_homo' or filename_i == 'abalone_19' \
            or filename_i == 'scene'  or filename_i == 'thyroid_sick' \
            or filename_i == 'mammography' or filename_i == 'isolet'   :
        path_prefix = '../data/'
        fnames = glob(path_prefix+filename_i+'.npz')
        data = [np.load(f) for f in fnames][0]
        
        X=data['data']
        y=data['label']
        
        dim = X.shape[1]
        if filename_i == 'protein_homo' or filename_i == 'webpage':
            idx_sel = np.random.choice(range(len(X)), np.min([len(X), 10000]), replace=False)
            X = X[idx_sel]
            y = y[idx_sel]
    
    
    N0 = len(np.where(y==-1)[0])
    N1 = len(np.where(y==1)[0])
    
    index0 = np.where(y==-1)[0]
    index1 = np.where(y==1)[0]
    
    
    l_test0 = int(N0*0.01*Test_size)
    l_test1 = int(N1*0.01*Test_size)

    print('Dataset: '+filename_i+'. N0: %d. N1: %d. IR = %.2f' % (N0 , N1, N0/N1))
    
    ######## Costes y Probabilidades ########
    P0=float(N0)/float(N0+N1)
    P1=float(N1)/float(N0+N1)
    QP=P0/P1

    P0_1=0.5 # Para la balanced accuracy
    P1_1=0.5

    QPT=QP ## Sin Submuestreo

    C10=1 # False Positive (FP)
    C00=0 # True Negative (TN)
    C01=1 # False Negative (FN)   : QC=1/5
    C11=0 # True Positive (TP)

    QC=float(C10-C00)/float(C01-C11)
    QCT=QC

    Q=QC*QP
    ######## Costes y Probabilidades ########

    ############################## DATA LOAD ##############################

    perf_sw_K1 = np.zeros([Nruns, len(Alpha), len(Beta), 8]) # Confusion Matrix + MCC + F1
    perf_sw_K2 = np.zeros([Nruns,len(Alpha), len(Beta)])   # CBL        
    perf_sw_K1_ng = np.zeros([Nruns, len(Alpha), len(Beta), 8]) # Confusion Matrix + MCC + F1
    perf_sw_K2_ng = np.zeros([Nruns, len(Alpha), len(Beta)])   # CBL
    perf_sw_K3 = np.zeros([Nruns, len(Alpha), len(Beta)])   # AuC
    perf_sw_K3_ng = np.zeros([Nruns, len(Alpha), len(Beta)])   # AuC
    
    lROC=1001
    mean_fpr = np.logspace(-4, 0, lROC)
    
    perf_sw_K4=np.zeros([len(Alpha), len(Beta), lROC, 2])     # Test:  ROC(FPR, TPR)
    perf_sw_K4_ng=np.zeros([len(Alpha), len(Beta), lROC, 2])     # Test:  ROC(FPR, TPR)
    
    perf_sw_K5 = np.zeros([Nruns, len(Alpha), len(Beta)])       # Training Time: Gating Network
    perf_sw_K5_ng = np.zeros([Nruns, len(Alpha), len(Beta)])    # Training Time: Averaging
        
    for nruns in range(Nruns):
        if verbose:
            print("Run: ", nruns)
            print()
        
        ####### DATOS REALES #######
        
        idx0_test = np.random.choice(index0, l_test0, replace=False)
        idx1_test = np.random.choice(index1, l_test1, replace=False)
        idx_test = np.sort(np.concatenate((idx0_test, idx1_test), axis=0))
        
        X_train = np.delete(X, idx_test, axis=0)
        X_test = X[idx_test]
        y_train_n = np.delete(y, idx_test, axis=0)
        y_test_n = y[idx_test]
        
        ####### DATOS REALES #######
        
        scaler = StandardScaler() # MinMaxScaler(feature_range=(-1, 1))  # MinMaxScaler
        scaler.fit(X_train)
        X_train_n = scaler.transform(X_train)
        X_test_n = scaler.transform(X_test)
            
        N0_tr = len(np.where(y_train_n==-1)[0])
        N1_tr = len(np.where(y_train_n==1)[0])
        
        P0_tr = float(N0_tr)/float(N0_tr+N1_tr)
        P1_tr = float(N1_tr)/float(N0_tr+N1_tr)
        
        QP_tr = P0_tr/P1_tr
        
        # Dos conjuntos de train con resampling. Uno para los conjuntos
        # y otro para la puerta. Solo se utilizan cuando resample = True
        
        y_true = y_test_n
        
        #################### SMOTE ####################
        # Resampleamos SOLO 1 vez (mismo resampling para cada experto)
        if Q_RB_S==0 or Q_RB_S==[0]:
            ind_ord = np.arange(len(y_train_n))
            ind_rnd = np.random.choice(ind_ord, size=len(y_train_n), replace=False)
            X_train_SM, y_train_SM = X_train_n[ind_rnd,:], y_train_n[ind_rnd]

        # No se utiliza para costes
        N0_sm = len(np.where(y_train_SM==-1)[0])
        N1_sm = len(np.where(y_train_SM==1)[0])
        
        P0_sm = float(N0_sm)/float(N0_sm+N1_sm)
        P1_sm = float(N1_sm)/float(N0_sm+N1_sm)
        
        QP_sm = P0_sm/P1_sm
        #################### SMOTE ####################
        
        if Q_RB_C == 0 or Q_RB_C == [0]:
            Q_RB = QP_tr

                
        if resample==True:
            X_train_e, X_train_g, y_train_e, y_train_g = train_test_split(X_train_SM, y_train_SM, test_size=0.33, random_state=42)
            
            Xy_train = np.concatenate((X_train_e,y_train_e.reshape((len(y_train_e),1))),axis=1)
            X_train_torch = torch.from_numpy(X_train_e)
            y_train_torch_orig = torch.from_numpy(y_train_e)
            X_train_gate_torch = torch.from_numpy(X_train_g)
            y_train_gate_torch = torch.from_numpy(y_train_g)
        else:
            if validate==True:
                X_train_g, X_val, y_train_g, y_val = train_test_split(X_train_SM, y_train_SM, test_size=0.33, random_state=42)
                
                Xy_train = np.concatenate((X_train_g,y_train_g.reshape((len(y_train_g),1))),axis=1)
                if full_Train:
                    X_train_torch = torch.from_numpy(X_train_SM)
                    y_train_torch_orig = torch.from_numpy(y_train_SM)
                else:
                    X_train_torch = torch.from_numpy(X_train_g)
                    y_train_torch_orig = torch.from_numpy(y_train_g)                    
                X_train_gate_torch = torch.from_numpy(X_train_g)
                y_train_gate_torch = torch.from_numpy(y_train_g)
                X_val_torch = torch.from_numpy(X_val)
                M_val = X_val_torch.shape[0]
                y_val_torch = torch.from_numpy(y_val)
            else:
                Xy_train = np.concatenate((X_train_SM,y_train_SM.reshape((len(y_train_SM),1))),axis=1)
                X_train_torch = torch.from_numpy(X_train_SM)
                y_train_torch_orig = torch.from_numpy(y_train_SM)
                X_train_gate_torch = X_train_torch
                y_train_gate_torch = y_train_torch_orig
        
        X_test_torch = torch.from_numpy(X_test_n)
        y_test_torch = torch.from_numpy(y_test_n)
        
        M_tr_gate = X_train_gate_torch.shape[0]
        M_tr = X_train_torch.shape[0]
        M_tst = X_test_torch.shape[0]

        d_ =  X_train_torch.shape[1]
        Ne_ = K_ens[0]
        weights_E = compute_weights(y_train_torch_orig.detach().numpy(), RB = 1, IR = Weight_MoE, mode = 'Normal')
        weights_G = compute_weights(y_train_gate_torch.detach().numpy(), RB = 1, IR = Weight_MoE, mode = 'Normal')
        a_idx=0
        for alpha in Alpha:
            b_idx=0
            for beta in Beta:
                miMoE_ng = 0
                miMoE_ng = BigMoE(input_size=X_train.shape[1], hidden_size=nh, num_outputs=1, 
                               num_experts=K_ens[0], alpha=alpha, beta=beta, 
                               RB_e = Q_RB, IR_e = QP_tr,
                               RB_g = 1, IR_g = Weight_MoE,
                               clusters = Clusters)                
                
                miMoE_ng.alpha = alpha
                miMoE_ng.beta = beta
                
                y_train_torch = torch.zeros((M_tr, Ne_))
                if alpha == 0 and beta==0:
                    for jj in range(Ne_):
                        y_train_torch[:,jj] = y_train_torch_orig
                else:
                    for jj in range(Ne_):
                        y_train_torch_sw_pm1 = label_switching(y_train_torch_orig, miMoE_ng.alpha, miMoE_ng.beta)
                        beta_sw = (1-2*beta)*torch.ones_like(y_train_torch_sw_pm1).float()
                        alpha_sw = -(1-2*alpha)*torch.ones_like(y_train_torch_sw_pm1).float()
                        y_train_torch[:,jj] = torch.where(y_train_torch_sw_pm1>0, beta_sw, alpha_sw)
                
                pi_0 = torch.ones((M_tr, Ne_))/Ne_ # \g_p (x) Eq 5.
                resp_0 = pi_0                     # \gamma_p (x) Eq 5.
                if validate==True:
                    pi_val = torch.ones((M_val, Ne_))/Ne_ # \g_p (x) Eq 5.
                pi_tst = torch.ones((M_tst, Ne_))/Ne_ # \g_p (x) Eq 5.
                g_p = pi_0
                gamma_p = resp_0
                
                # v_best = torch.hstack([torch.zeros(Ne_).unsqueeze(1), torch.zeros(Ne_,d_) ]).double()
                v_best = torch.cat([torch.zeros(Ne_).unsqueeze(1), torch.zeros(Ne_,d_) ], dim=1).double()   # a, b - 2d torch.Tensors
                """
                miMoE_ng.gate_layer.bias.data = v_best[:,0].float()
                miMoE_ng.gate_layer.weight.data = v_best[:,1:].float()
                """
                start_avr = time.time()
                miMoE_ng = miMoE_ng.fit_expert_model(Xy_train, X_train_torch, y_train_torch, gamma_p, lbfgs=True)
                # miMoE = miMoE.fit_expert_model(Xy_train, X_train_torch, y_train_torch, gamma_p, lbfgs=True)
                end_avr = time.time()
                perf_sw_K5_ng[nruns,a_idx,b_idx] = end_avr-start_avr
                if verbose:
                    print('['+filename_i+'] TRAIN TIME average: %.2gs'%(end_avr-start_avr))
                    print()
                
                o_pred_sw = miMoE_ng.predict_nogate(X_train_torch)
                pyX_p = (0.5*(o_pred_sw+1)-alpha)/(1-alpha-beta)
                
                centered_o_pred_sw = center_outputs(o_pred_sw, alpha, beta, False)
                o_tr = torch.sum(centered_o_pred_sw * g_p,dim=1)

                ##### TEST NO GATE
                
                o_pred_test_ng = torch.sum(miMoE_ng.predict_nogate(X_test_torch)*pi_tst,dim=1)
                o_x_ng = np.ones_like(o_pred_test_ng)
                o_x_ng[o_pred_test_ng <  (2 * (alpha + (1 - alpha - beta) * (QCT / (QCT + (Q_RB/QP_tr)))) - 1)] = -1
                
                # Computo de la curva ROC y el AUC
                pyX_test=(0.5*(o_pred_test_ng+1)-alpha)/(1-alpha-beta)
                
                FPR_vec, TPR_vec, thresholds = roc_curve(y_true, pyX_test)
                roc_auc = auc(FPR_vec, TPR_vec)
                perf_sw_K3_ng[nruns,a_idx,b_idx] = roc_auc
                
                f_ROC = interp1d(FPR_vec, TPR_vec)
                
                perf_sw_K4_ng[a_idx,b_idx,:,0] += mean_fpr
                perf_sw_K4_ng[a_idx,b_idx,:,1] += f_ROC(mean_fpr)
                
                # Obtención de prestaciones
                TP, FP, TN, FN = perf_measure(y_true, o_x_ng) 
                
                # Sensitivity, hit rate, recall, or true positive rate
                TPR=TP/float(TP+FN)
                
                # Specificity (SPC) or true negative rate
                SPC=TN/float(TN+FP)
                
                # False positive rate: FPR=FP/float(FP+TN) = 1-SPC
                FPR=1-SPC
                
                # False negative rate
                FNR=1-TPR
                
                # Mathhews correlation coefficient
                if (((TP + FP) == 0) or ((TP + FN) == 0) or ((TN + FP) == 0) or ((TN + FN) == 0)):
                    MCC = 0
                else:
                    MCC = (TP * TN - FP * FN) / np.sqrt(float(TP + FP) * float(TP + FN) * float(TN + FP) * float(TN + FN))
                
                # Accuracy
                ACC=(TP+TN)/float(TP+FN+FP+TN)
                
                # F1 Score
                F1=2*TP/float(2*TP+FP+FN) 
                
                perf_sw_K1_ng[nruns,a_idx,b_idx,0]=FPR
                perf_sw_K1_ng[nruns,a_idx,b_idx,1]=TPR
                perf_sw_K1_ng[nruns,a_idx,b_idx,2]=SPC
                perf_sw_K1_ng[nruns,a_idx,b_idx,3]=ACC
                perf_sw_K1_ng[nruns,a_idx,b_idx,4]=MCC
                perf_sw_K1_ng[nruns,a_idx,b_idx,5]=F1
                if TP+FP == 0:
                    perf_sw_K1_ng[nruns,a_idx,b_idx,6]=0  # Precision
                    perf_sw_K1_ng[nruns,a_idx,b_idx,7]=0  # G-Mean
                else:
                    perf_sw_K1_ng[nruns,a_idx,b_idx,6]= TP/float(TP+FP) # Precision
                    perf_sw_K1_ng[nruns,a_idx,b_idx,7]= np.sqrt((TP/float(TP+FP))*TPR) # G-Mean
                
                perf_sw_K2_ng[nruns,a_idx,b_idx]=(FPR*P0_1+FNR*P1_1) # Balanced Accuracy
                
                if validate==True:
                    o_pred_val_ng = torch.sum(miMoE_ng.predict_nogate(X_val_torch)*pi_val,dim=1)
                    o_x_val_ng = np.ones_like(o_pred_val_ng)
                    o_x_val_ng[o_pred_val_ng <  (2 * (alpha + (1 - alpha - beta) * (QCT / (QCT + (Q_RB/QP_tr)))) - 1)] = -1
                    bic_val_ng = 1-f_sel(y_val_torch,o_x_val_ng)
                    
                bic_train_ng = 1-f_sel(y_train_torch_orig,torch.sign(o_tr))
                bic_tst_ng = 1-f_sel(y_true,o_x_ng)
                if verbose:
                    if validate==True:
                        print('No Gate: BIC_train_ng = %.3f | BIC_val_ng = %.3f | BIC_tst_ng = %.3f ' %(bic_train_ng, bic_val_ng, bic_tst_ng))
                    else:
                        print('No Gate: BIC_train_ng = %.3f | BIC_tst_ng = %.3f ' %(bic_train_ng, bic_tst_ng))

                    
                #______________________________________________________
                # Expectation-Maximization Loop

                ii_max = 0
                # Expectation-Maximization Algorithm 

                X_tr_gate = torch.cat([Bias*torch.ones(M_tr_gate).unsqueeze(1).double(), cX*X_train_gate_torch], dim=1)   # a, b - 2d torch.Tensors 
                X_tr_E = torch.cat([Bias*torch.ones(M_tr).unsqueeze(1).double(), cX*X_train_torch], dim=1)   # a, b - 2d torch.Tensors 
                
                v_ = torch.cat([v_sig*torch.randn(Ne_).unsqueeze(1), cX*v_sig*torch.randn(Ne_,d_)], dim=1).double()
                v_[0,:] =  0*v_[0,:]

                miMoE = copy.deepcopy(miMoE_ng)
                # miMoE.gate_layer[0].bias.data = v_[:,0].float()
                # miMoE.gate_layer[0].weight.data = v_[:,1:].float()
                
                if pyX == 'Norm':
                    if pyX_sig == 'Auto':
                        pyX_sig_2 = 2
                    else:
                        pyX_sig_2 = pyX_sig
                    gv_p = pyX_sig_2*torch.ones(Ne_)
                    tol = 1e-4
                
                if validate==True:
                    bic_min = bic_val_ng
                else:
                    bic_min = bic_train_ng
                    
                ii_min = 0
                                        
                miMoE_best = copy.deepcopy(miMoE_ng)
                
                flag = 1
                
                start_gate = time.time()
                for ii in range(Niter_EM):
                    # Inspired by: 
                    #    Mixture of experts: a literature survey
                    #    Saeed Masoudnia & Reza Ebrahimpour 
                    #    Artificial Intelligence Review volume 42, pages275–293 (2014)
                    #    https://link.springer.com/article/10.1007/s10462-012-9338-y
                    # and
                    #    A Regularized Mixture of Linear Experts for 
                    #    Quality Prediction in Multimode and 
                    #    Multiphase Industrial Processes
                    #    Francisco Souza,Jérôme Mendes Rui Araújo
                    #    https://www.mdpi.com/2076-3417/11/5/2040
                    

 
                    o_pred_sw = miMoE.predict_nogate(X_train_gate_torch)   
                    pyX_p = (0.5*(o_pred_sw+1)-alpha)/(1-alpha-beta)
                    centered_o_pred_sw = center_outputs(o_pred_sw, alpha, beta, False)   # False: solarflare
                    
                    o_pred_sw_E = miMoE.predict_nogate(X_train_torch)   
                    pyX_p_E = (0.5*(o_pred_sw_E+1)-alpha)/(1-alpha-beta)
                    centered_o_pred_sw_E = center_outputs(o_pred_sw_E, alpha, beta, False)   # False: solarflare

                    
                    if pyX == 'BCE':
                        y_train_torch_01 = 0.5*(y_train_gate_torch.double()+1)
                        p1 = torch.pow(pyX_p,y_train_torch_01[:,None])
                        p2 = torch.pow(1-pyX_p,1-y_train_torch_01[:,None])
                        prob = torch.clamp(torch.mul(p1,p2), min=eps, max=1-eps)
                    
                        y_train_torch_01_E = 0.5*(y_train_torch_orig.double()+1)
                        p1_E = torch.pow(pyX_p_E,y_train_torch_01_E[:,None])
                        p2_E = torch.pow(1-pyX_p_E,1-y_train_torch_01_E[:,None])
                        prob_E = torch.clamp(torch.mul(p1_E,p2_E), min=eps, max=1-eps)
                    elif pyX == 'Norm': 
                        prob = torch.from_numpy(norm.pdf(centered_o_pred_sw, loc=y_train_gate_torch[:,None],scale=np.sqrt(gv_p)+tol))
                        prob_E = torch.from_numpy(norm.pdf(centered_o_pred_sw_E, loc=y_train_torch_orig[:,None],scale=np.sqrt(gv_p)+tol))

                    gate = X_tr_gate @ v_.T
                    pi = torch.exp(gate-logsumexp(gate)[:, None]) # Identical to gates_train. g_p Eq 2.
                    tmp = torch.diag(prob @ pi.double().T)
                    resp_tr = torch.div((prob * pi).T + eps,tmp + eps).T   # \gamma_p (x) Eq 5.
                    
                    gamma_p = resp_tr
                    g_p = pi

                    gate_E = X_tr_E @ v_.T
                    pi_E = torch.exp(gate_E-logsumexp(gate_E)[:, None]) # Identical to gates_train. g_p Eq 2.
                    tmp = torch.diag(prob_E @ pi_E.double().T)
                    resp_tr_E = torch.div((prob_E * pi_E).T + eps,tmp + eps).T   # \gamma_p (x) Eq 5.
                    
                    gamma_p_E = resp_tr_E
                    

                    # M-step

                    
                    miMoE = miMoE.fit_expert_model(Xy_train, X_train_torch, y_train_torch, gamma_p_E, lbfgs=True)  # Solarflare: gamma_p**-2
                    
                     
                    # Gate Function Update
                    
                    y_gate = resp_tr

                    # Initialization
                    
                    # Single Loop EM algorithm for Mixture of Experts
                    # https://link.springer.com/chapter/10.1007/978-3-642-01510-6_109
                    y_gate_0 = y_gate[:,0]

                    Hk = torch.log(torch.div(y_gate[:,1:], y_gate_0[:,None]))
                    

                    x_np = X_tr_gate.detach().numpy()
                    y_np = Hk.detach().numpy()
                    
                    w_np = weights_G.detach().numpy()
 
                    mod_Lasso =  ElasticNet(alpha=0.01) # LinearRegression() # Lasso(alpha=lamda_ini) #  Lasso(alpha=0.01) # LinearRegression()

                    mod_Lasso.fit(x_np[:,1:], y_np, sample_weight=w_np)
                    v_[1:,1:] = torch.from_numpy(mod_Lasso.coef_)
                    v_[1:,0] = torch.from_numpy(mod_Lasso.intercept_)
                    
                    v_[0,:] =  0*v_[0,:]

                                                 
                    gate = X_tr_gate @ v_.T
                    pi = torch.exp(gate-logsumexp(gate)[:, None]) # Identical to gates_train. g_p Eq 2.
                    g_p = pi
                    centered_o_pred_sw = center_outputs(o_pred_sw, alpha, beta, False)
                    o_tr = torch.sum(centered_o_pred_sw * g_p,dim=1)

                    if pyX == 'Norm':
                        if pyX_sig == 'Auto':
                            er_p = weights_G[:,None]*(y_train_gate_torch[:, None]-centered_o_pred_sw)**2
                            # er_p = (y_train_torch_orig[:, None]-o_pred_sw*g_p)**2
                            Er_p = (resp_tr*er_p).sum(dim=0)
                            gv_p = 2*torch.div(Er_p+eps,resp_tr.sum(dim=0)+eps)
                        else:
                            gv_p = pyX_sig_2*torch.ones(Ne_)
                    

                    
                    if validate==True:
                        X_val_gate = torch.cat([Bias*torch.ones(M_val).unsqueeze(1).double(), X_val_torch], dim=1)
        
                        gate_val = X_val_gate @ v_.T
                        pi_val = torch.exp(gate_val-logsumexp(gate_val)[:, None]) # Identical to gates_train. g_p Eq 2.
       
                        o_pred_val_sw = torch.sum(miMoE.predict_nogate(X_val_torch)*pi_val,dim=1)
                        o_x_val = np.ones_like(o_pred_val_sw)
                        o_x_val[o_pred_val_sw <  (2 * (alpha + (1 - alpha - beta) * (QCT / (QCT + (Q_RB/QP_tr)))) - 1)] = -1
                        bic = 1-f_sel(y_val_torch,o_x_val)
                    else:   
                        bic = 1-f_sel(y_train_gate_torch,torch.sign(o_tr))
                    
                    if bic < (1-np.sign(bic_min)*0.03*flag)*bic_min: 
                        
                        miMoE_best = copy.deepcopy(miMoE)
                        v_best = v_


                        flag = 0
                        bic_min = bic
                        ii_min = ii
                        
                    

                if flag:
                    if verbose:
                        print('\nNo Improvement: average\n\n')
                    
                    pi = torch.ones((M_tr, Ne_))/Ne_ # \gamma_p (x) Eq 5.
                    g_p = pi
                    v_best = torch.cat([torch.zeros(Ne_).unsqueeze(1), torch.zeros(Ne_,d_) ], dim=1).double()   # a, b - 2d torch.Tensors
                    miMoE = copy.deepcopy(miMoE_ng)
                else:
                    if verbose:
                        print('\nImprovement: gate\n\n')
                    miMoE = copy.deepcopy(miMoE_best)
 
                end_gate = time.time()
                perf_sw_K5[nruns,a_idx,b_idx] = end_gate-start_gate+0*(end_avr-start_avr)
                if verbose:
                    print('['+filename_i+'] TRAIN TIME gating: %.2gs'%(end_gate-start_gate+0*(end_avr-start_avr)))
                    print()
                
                # Test Stage
                v_ = v_best
                X_tst_gate = torch.cat([Bias*torch.ones(M_tst).unsqueeze(1).double(), X_test_torch], dim=1)

                gate_tst = X_tst_gate @ v_.T
                pi_tst = torch.exp(gate_tst-logsumexp(gate_tst)[:, None]) # Identical to gates_train. g_p Eq 2.
   
                o_pred_test_sw = torch.sum(miMoE.predict_nogate(X_test_torch)*pi_tst,dim=1)
                o_x = np.ones_like(o_pred_test_sw)
                o_x[o_pred_test_sw <  (2 * (alpha + (1 - alpha - beta) * (QCT / (QCT + (Q_RB/QP_tr)))) - 1)] = -1

                pyX_test_sw=(0.5*(o_pred_test_sw+1)-alpha)/(1-alpha-beta)

                # Computo de la curva ROC y el AUC
                
                pyX_test = pyX_test_sw # 0.5*(o_pred+1)
                
                FPR_vec, TPR_vec, thresholds = roc_curve(y_true, pyX_test)
                roc_auc = auc(FPR_vec, TPR_vec)
                perf_sw_K3[nruns,a_idx,b_idx] = roc_auc
                
                f_ROC = interp1d(FPR_vec, TPR_vec)
                
                perf_sw_K4[a_idx,b_idx,:,0] += mean_fpr
                perf_sw_K4[a_idx,b_idx,:,1] += f_ROC(mean_fpr)
                
                # Obtención de prestaciones
                TP, FP, TN, FN = perf_measure(y_true, o_x) 
                
                # Sensitivity, hit rate, recall, or true positive rate
                TPR=TP/float(TP+FN)
                
                # Specificity (SPC) or true negative rate
                SPC=TN/float(TN+FP)
                
                # False positive rate: FPR=FP/float(FP+TN) = 1-SPC
                FPR=1-SPC
                
                # False negative rate
                FNR=1-TPR
                
                # Mathhews correlation coefficient
                if (((TP + FP) == 0) or ((TP + FN) == 0) or ((TN + FP) == 0) or ((TN + FN) == 0)):
                    MCC = 0
                else:
                    MCC = (TP * TN - FP * FN) / np.sqrt(float(TP + FP) * float(TP + FN) * float(TN + FP) * float(TN + FN))
                
                # Accuracy
                ACC=(TP+TN)/float(TP+FN+FP+TN)
                
                # F1 Score
                F1=2*TP/float(2*TP+FP+FN) 
                
                perf_sw_K1[nruns,a_idx,b_idx,0]=FPR
                perf_sw_K1[nruns,a_idx,b_idx,1]=TPR
                perf_sw_K1[nruns,a_idx,b_idx,2]=SPC
                perf_sw_K1[nruns,a_idx,b_idx,3]=ACC
                perf_sw_K1[nruns,a_idx,b_idx,4]=MCC
                perf_sw_K1[nruns,a_idx,b_idx,5]=F1
                if TP+FP == 0:
                    perf_sw_K1[nruns,a_idx,b_idx,6]=0 # Precision
                    perf_sw_K1[nruns,a_idx,b_idx,7]==0  # G-Mean
                else:
                    perf_sw_K1[nruns,a_idx,b_idx,6]= TP/float(TP+FP) # Precision
                    perf_sw_K1[nruns,a_idx,b_idx,7]= np.sqrt((TP/float(TP+FP))*TPR) # G-Mean
                
                perf_sw_K2[nruns,a_idx,b_idx]=(FPR*P0_1+FNR*P1_1) # Balanced Accuracy
                """
                print('Run: %d. Performance on Test data. Alpha = %.2f. Beta = %.2f' %(nruns, alpha, beta))
                print('No Gate. F1: %f +- %f | MCC: %f +- %f | TPR: %f | FPR: %f' %(np.mean(perf_sw_K1_ng[nruns,a_idx,b_idx,5]), np.std(perf_sw_K1_ng[nruns,a_idx,b_idx,5]), np.mean(perf_sw_K1_ng[nruns,a_idx,b_idx,4]), np.std(perf_sw_K1_ng[nruns,a_idx,b_idx,4]), np.mean(perf_sw_K1_ng[nruns,a_idx,b_idx,1]), np.mean(perf_sw_K1_ng[nruns,a_idx,b_idx,0])))
                print('Gate. F1: %f +- %f | MCC: %f +- %f | TPR: %f | FPR: %f' %(np.mean(perf_sw_K1[nruns,a_idx,b_idx,5]), np.std(perf_sw_K1[nruns,a_idx,b_idx,5]), np.mean(perf_sw_K1[nruns,a_idx,b_idx,4]), np.std(perf_sw_K1[nruns,a_idx,b_idx,4]), np.mean(perf_sw_K1[nruns,a_idx,b_idx,1]), np.mean(perf_sw_K1[nruns,a_idx,b_idx,0])))
                print('________________________\n')
                """
                b_idx += 1
            a_idx +=1
    """
    end = time.time()
    print('['+filename_i+'] TRAIN TIME:')
    print('%.2gs'%(end-start))
    print()
    """
    perf_sw_K1_std = np.std(perf_sw_K1, axis=0)
    perf_sw_K1 = np.mean(perf_sw_K1, axis=0)
    
    perf_sw_K2_std = np.std(perf_sw_K2, axis=0)
    perf_sw_K2 = np.mean(perf_sw_K2, axis=0)
    
    perf_sw_K1_ng_std = np.std(perf_sw_K1_ng, axis=0)
    perf_sw_K1_ng = np.mean(perf_sw_K1_ng, axis=0)
    
    perf_sw_K2_ng_std = np.std(perf_sw_K2_ng, axis=0)
    perf_sw_K2_ng = np.mean(perf_sw_K2_ng, axis=0)
    
    perf_sw_K3_std = np.std(perf_sw_K3, axis=0)
    perf_sw_K3 = np.mean(perf_sw_K3, axis=0)
    perf_sw_K3_ng_std = np.std(perf_sw_K3_ng, axis=0)
    perf_sw_K3_ng = np.mean(perf_sw_K3_ng, axis=0)
    
    perf_sw_K4 /= n_folds*Nruns
    perf_sw_K4_ng /= n_folds*Nruns

    perf_sw_K5_std = np.std(perf_sw_K5, axis=0)
    perf_sw_K5 = np.mean(perf_sw_K5, axis=0)
    perf_sw_K5_ng_std = np.std(perf_sw_K5_ng, axis=0)
    perf_sw_K5_ng = np.mean(perf_sw_K5_ng, axis=0)
    
    a_idx = 0
    for alpha in Alpha:
        b_idx = 0
        for beta in Beta:
            """
            print('Performance on Test data. Alpha = %.2f. Beta = %.2f' %(alpha, beta))
            print('No Gate. F1: %f +- %f | MCC: %f +- %f | TPR: %f | FPR: %f' %(perf_sw_K1_ng[a_idx,b_idx,5], perf_sw_K1_ng_std[a_idx,b_idx,5], perf_sw_K1_ng[a_idx,b_idx,4], perf_sw_K1_ng_std[a_idx,b_idx,4], perf_sw_K1_ng[a_idx,b_idx,1], perf_sw_K1_ng[a_idx,b_idx,0]))
            print('Gate. F1: %f +- %f | MCC: %f +- %f | TPR: %f | FPR: %f' %(perf_sw_K1[a_idx,b_idx,5], perf_sw_K1_std[a_idx,b_idx,5], perf_sw_K1[a_idx,b_idx,4], perf_sw_K1_std[a_idx,b_idx,4], perf_sw_K1[a_idx,b_idx,1], perf_sw_K1[a_idx,b_idx,0]))
            print('________________________\n')
            """
            print('Training Time for '+filename_i+'. Alpha = %.2f. Beta = %.2f' %(alpha, beta))
            print('Averaging: %f +- %f' %(perf_sw_K5_ng[a_idx,b_idx], perf_sw_K5_ng_std[a_idx,b_idx]))
            print('Gating: %f +- %f' %(perf_sw_K5[a_idx,b_idx], perf_sw_K5_std[a_idx,b_idx]))
            print('________________________\n')

            b_idx += 1
        a_idx += 1

            