# -*- coding: utf-8 -*-
"""
Created on Tue May 17 15:50:25 2016

@author: hossam
"""
import PSO as pso
import MVO as mvo
import GWO as gwo
import MFO as mfo
import CS as cs
import BAT as bat
import WOA as woa
import FFA as ffa
import SSA as ssa
import GA as ga
import HHO as hho
import SCA as sca
import JAYA as jaya
import benchmarks
import csv
import time
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import pandas as pd 
import numpy
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.svm import SVC

import warnings
warnings.filterwarnings('ignore')
from lightgbm import LGBMClassifier
import math
from sklearn.model_selection import train_test_split
from math import sqrt
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
import numpy as np


train = pd.read_csv('data/svd_train.tsv', sep='\t')
test = pd.read_csv('data/svd_test.tsv', sep='\t')


print(train.head())
print('The number of features %s' % (len(train.columns)-1))


features = len(train.columns)
    
def selector(algo, func_details, popSize, Iter, alpha):
    function_name=func_details[0]
    
    lb = func_details[1]
    ub = func_details[2]

    print(lb)
    print(ub)
    #dim=func_details[3]
    #dim=len(train.columns)+1                            #excluding the target class, and including cost & gama
    dim = features + 1
    #dim = features
    #Alpha
    #alpha=numpy.random.uniform(0.001,1)                 #if you don't want to use alpha put it 0

    fid = 2
    model_params = []
   
    if(algo==0):
        x=pso.PSO(getattr(benchmarks, function_name),lb,ub,dim,popSize,Iter,fid, alpha)
    if(algo==1):
        x=mvo.MVO(getattr(benchmarks, function_name),lb,ub,dim,popSize,Iter)
    if(algo==2):
        x=gwo.GWO(getattr(benchmarks, function_name),lb,ub,dim,popSize,Iter)
    if(algo==3):
        x=mfo.MFO(getattr(benchmarks, function_name),lb,ub,dim,popSize,Iter)
    if(algo==4):
        x=cs.CS(getattr(benchmarks, function_name),lb,ub,dim,popSize,Iter)
    if(algo==5):
        x=bat.BAT(getattr(benchmarks, function_name),lb,ub,dim,popSize,Iter)
    if(algo==6):
        x=woa.WOA(getattr(benchmarks, function_name),lb,ub,dim,popSize,Iter, alpha)
    if(algo==7):
        x=ffa.FFA(getattr(benchmarks, function_name),lb,ub,dim,popSize,Iter)
    if(algo==8):
        x=ssa.SSA(getattr(benchmarks, function_name),lb,ub,dim,popSize,Iter, fid)
    if(algo==9):
        x=ga.GA(getattr(benchmarks, function_name),lb,ub,dim,popSize,Iter)
    if(algo==10):
        x=hho.HHO(getattr(benchmarks, function_name),lb,ub,dim,popSize,Iter)
    if(algo==11):
        x=sca.SCA(getattr(benchmarks, function_name),lb,ub,dim,popSize,Iter)
    if(algo==12):
        x=jaya.JAYA(getattr(benchmarks, function_name),lb,ub,dim,popSize,Iter)
    return x
    
    
# Select optimizers
GWO = False
PSO= True
MVO= False#
MFO = False
CS = False
BAT = False#  #try it again. keep reach a number of selected features 0
WOA = True
FFA = False#
SSA = False
GA = False#
HHO = False#
SCA = False
JAYA = False


# Select benchmark function
F1=False
F2=True
F3=False
F4=False
F5=False
F6=False
F7=False
F8=False
F9=False
F10=False
F11=False
F12=False
F13=False
F14=False
F15=False
F16=False
F17=False
F18=False
F19=False



optimizer=[PSO, MVO, GWO, MFO, CS, BAT, WOA, FFA, SSA, GA, HHO, SCA, JAYA]
benchmarkfunc=[F1,F2,F3,F4,F5,F6,F7,F8,F9,F10,F11,F12,F13,F14,F15,F16,F17,F18,F19] 
        
# Select number of repetitions for each experiment. 
# To obtain meaningful statistical results, usually 30 independent runs 
# are executed for each algorithm.
NumOfRuns = 1

# Select general parameters for all optimizers (population size, number of iterations)
PopulationSize = 5
Iterations= 3

#Export results ?
Export=True


#ExportToFile="YourResultsAreHere.csv"
#Automaticly generated name by date and time

# Check if it works at least once
Flag=False

CnvgHeader=[]

ACC = numpy.zeros(Iterations)
RECALL = numpy.zeros(Iterations)
PREC = numpy.zeros(Iterations)


P0 = numpy.zeros(Iterations)
P1 = numpy.zeros(Iterations)
R0 = numpy.zeros(Iterations)
R1 = numpy.zeros(Iterations)
F0 = numpy.zeros(Iterations)
F1 = numpy.zeros(Iterations)
Gm = numpy.zeros(Iterations)
FPR = numpy.zeros(Iterations)
FNR = numpy.zeros(Iterations)
AUC = numpy.zeros(Iterations)

TN = numpy.zeros(Iterations)
TP = numpy.zeros(Iterations)
FP = numpy.zeros(Iterations)
FN = numpy.zeros(Iterations)

FPR = np.zeros(Iterations)
FNR = np.zeros(Iterations)
AUC = np.zeros(Iterations)

ALPHA = [0.2, 0.5, 0.8, 0.85, 0.9, 0.95, 0.999]


for value in ALPHA:

    ExportToFile = f"results/WOA_PSO/svd/alpha/experiment_svd_alpha={value}_pop={PopulationSize}_iter={Iterations}_runs={NumOfRuns}_" + time.strftime(
        "%Y-%m-%d-%H-%M-%S") + ".csv"

    for l in range(0,Iterations):
        CnvgHeader.append("Iter"+str(l+1))

    global k
    for i in range (0, len(optimizer)):
        for j in range (0, len(benchmarkfunc)):
            if((optimizer[i]==True) and (benchmarkfunc[j]==True)): # start experiment if an optimizer and an objective function is selected
                for k in range (0, NumOfRuns):
                    print(['At run '+ str(k+1)])

                    func_details = benchmarks.getFunctionDetails(j)
                    x = selector(i, func_details, PopulationSize, Iterations, value)

                    #************************************************************

                    x2 = x.bestIndividual

                    try:

                        print("~~~~~~~~~~~~~~ outer stage ~~~~~~~~~~~~~~~~~")
                        print(x2)


                        cost = abs(float(x2[-1] * 34.9) + 0.1)
                        gama = abs(float(x2[-2] * 0.00009999) + 0.00000001) # #[0.00000001 - 0.0001]

                        #gama = float(x2[len(x2) - 1] * (1 - 0.0000001)) + 0.0000001  # gm_actual = gm_scaled * (max-min)+min
                        #cost = float(x2[len(x2) - 2] * (35 - 1) + 1)

                        print(f'At outer-stage evaluation: Cost = {cost}, and gamma = {gama}')

                        #neighbors = abs(int(x2[-1] * 19) + 1)

                        #leaves = abs(int(x2[-1]*25)+10) #[10-35]#https://www.sciencedirect.com/science/article/pii/S016974392100229X
                        #lr = abs(float(x2[-2]*0.999)+0.001) #0.001-1

                        #lgbm = LGBMClassifier(num_leaves=leaves, learning_rate=lr)  # num_leaves learning_rate

                        #trees = abs(int(x2[-1] * 90)+ 10) #[10:500]
                        #rf = RandomForestClassifier(n_estimators=trees)

                        print(len(x2[0:-2]))


                        #x_features = [round(i) for i in x2[0:-2]]
                        x_features = x2[0:-2].copy()


                        # if np.count_nonzero(x_features) == 0:
                        #     x_features[np.random.randint(len(x_features))] = 1

                        X_train = train.loc[:, train.columns != 'label']
                        selectedCols = [x for i, x in enumerate(X_train.columns) if x_features[i]==1]
                        X_train = X_train.loc[:, selectedCols]
                        print('in optimizer the num of features after FS %s' % len(X_train.columns))
                        y_train = train['label'].values.tolist()

                        X_test = test.loc[:, test.columns != 'label']
                        selectedCols = [x for i, x in enumerate(X_test.columns) if x_features[i] == 1]
                        X_test = X_test.loc[:, selectedCols]
                        y_test = test['label'].values.tolist()

                        #knn = KNeighborsClassifier(n_neighbors=k)

                        svm = SVC(C=cost, gamma=gama, random_state=int(time.time()))

                        # if len(X_train.columns) == 0:
                        #     X_train = train.loc[:, train.columns != 'label']

                        svm.fit(X_train, y_train)
                        y_pred = svm.predict(X_test)

                        accuracy = accuracy_score(y_test, y_pred)
                        f1_score1 = f1_score(y_test, y_pred,average='macro')
                        recall = recall_score(y_test, y_pred, average='macro')
                        precision = precision_score(y_test, y_pred, average='macro')

                        print("Accuracy: %.3f%%" % (accuracy * 100.0))
                        print("F1-score: %.3f%%" % (f1_score1 * 100.0))

                        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
                        TN[0] = tn
                        TP[0] = tp
                        FP[0] = fp
                        FN[0] = fn

                        fpr = fp / (fp+tn)
                        fnr = fn / (fn+tp)
                        tpr = tp / (tp+fn)
                        auc = (((1-fpr)*(1+tpr))/2) + ((fpr*tpr)/2)


                        report = classification_report(y_test, y_pred, digits=3, output_dict=True)

                        print(report)
                        P0[0] = report['0']['precision']
                        P1[0] = report['1']['precision']
                        R0[0] = report['0']['recall']
                        R1[0] = report['1']['recall']
                        F0[0] = report['0']['f1-score']
                        F1[0] = report['1']['f1-score']
                        Gm[0] = math.sqrt(report['0']['recall'] * report['1']['recall'])
                        FPR[0] = fpr
                        FNR[0] = fnr
                        AUC[0] = auc


                        ACC[0] = accuracy
                        F1[0] = f1_score1
                        RECALL[0] =recall
                        PREC[0] = precision

                        if(Export==True):
                            with open(ExportToFile, 'a', newline='\n') as out:
                                writer = csv.writer(out, delimiter=',')
                                if (Flag==False): # just one time to write the header of the CSV file
                                    header= numpy.concatenate([["Optimizer","objfname","startTime","EndTime","ExecutionTime","Measure "],CnvgHeader])
                                    writer.writerow(header)

                                a=numpy.concatenate([[x.optimizer,x.objfname,x.startTime,x.endTime,x.executionTime,'Convergence'],x.convergence])
                                b=numpy.concatenate([[x.optimizer,x.objfname,x.startTime,x.endTime,x.executionTime,'Best ACC'],ACC])
                                c=numpy.concatenate([[x.optimizer,x.objfname,x.startTime,x.endTime,x.executionTime,'Best F1'],F1])
                                e=numpy.concatenate([[x.optimizer,x.objfname,x.startTime,x.endTime,x.executionTime,'Best Recall'],RECALL])
                                d=numpy.concatenate([[x.optimizer,x.objfname,x.startTime,x.endTime,x.executionTime,'Best Precision'],PREC])

                                tn = numpy.concatenate([[x.optimizer, x.objfname, x.startTime, x.endTime, x.executionTime, 'TN'], TN])
                                tp = numpy.concatenate([[x.optimizer, x.objfname, x.startTime, x.endTime, x.executionTime, 'TP'], TP])
                                fp = numpy.concatenate([[x.optimizer, x.objfname, x.startTime, x.endTime, x.executionTime, 'FP'], FP])
                                fn = numpy.concatenate([[x.optimizer, x.objfname, x.startTime, x.endTime, x.executionTime, 'FN'], FN])

                                p0 = numpy.concatenate([[x.optimizer, x.objfname, x.startTime, x.endTime, x.executionTime, 'P0'], P0])
                                p1 = numpy.concatenate([[x.optimizer, x.objfname, x.startTime, x.endTime, x.executionTime, 'P1'], P1])
                                r0 = numpy.concatenate([[x.optimizer, x.objfname, x.startTime, x.endTime, x.executionTime, 'R0'], R0])
                                r1 = numpy.concatenate([[x.optimizer, x.objfname, x.startTime, x.endTime, x.executionTime, 'R1'], R1])
                                f0 = numpy.concatenate([[x.optimizer, x.objfname, x.startTime, x.endTime, x.executionTime, 'F0'], F0])
                                f1 = numpy.concatenate([[x.optimizer, x.objfname, x.startTime, x.endTime, x.executionTime, 'F1'], F1])
                                gm = numpy.concatenate([[x.optimizer, x.objfname, x.startTime, x.endTime, x.executionTime, 'GM'], Gm])
                                fprr = numpy.concatenate([[x.optimizer, x.objfname, x.startTime, x.endTime, x.executionTime, 'FPR'], FPR])
                                fnrr = numpy.concatenate([[x.optimizer, x.objfname, x.startTime, x.endTime, x.executionTime, 'FNR'], FNR])
                                aucc = numpy.concatenate([[x.optimizer, x.objfname, x.startTime, x.endTime, x.executionTime, 'AUC'], AUC])


                                writer.writerow(a)
                                writer.writerow(b)
                                writer.writerow(c)
                                writer.writerow(e)
                                writer.writerow(d)

                                writer.writerow(tn)
                                writer.writerow(tp)
                                writer.writerow(fp)
                                writer.writerow(fn)
                                writer.writerow(p0)

                                writer.writerow(p1)
                                writer.writerow(r0)
                                writer.writerow(r1)
                                writer.writerow(f0)
                                writer.writerow(f1)
                                writer.writerow(gm)

                                writer.writerow(fprr)
                                writer.writerow(fnrr)
                                writer.writerow(aucc)

                                writer.writerow(['Selected Columns', selectedCols])
                                writer.writerow(['len Columns', len(selectedCols)])
                                writer.writerow(['feature reduction', (847 - len(selectedCols)) / 847])
                                writer.writerow(['cost', cost])
                                writer.writerow(['gamma', gama])
                                writer.writerow(['y_test', y_test])
                                writer.writerow(['y_pred', list(np.array(y_pred))])
                                writer.writerow("")

                                out.close()
                        Flag=True # at least one experiment

                    except Exception as e:
                        print(e)
                        print("The len of best individual is zero")
                        continue

    if (Flag==False): # Faild to run at least one experiment
        print("No Optomizer or Cost function is selected. Check lists of available optimizers and cost functions")


