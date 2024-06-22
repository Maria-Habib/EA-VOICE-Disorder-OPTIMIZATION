# -*- coding: utf-8 -*-
"""
Created on Sun May 15 22:37:00 2016

@author: Hossam Faris
"""

import random
import numpy
import statistics
import math
from colorama import Fore, Back, Style
from solution import solution
import time
from transferFun import transferFun
import datetime
import pandas as pd
from transferFun import sigmoid_transfer


def PSO(objf,lb,ub,dim,PopSize,iters,fid, alpha):
    # PSO parameters
    Vmax=6
    wMax=0.9
    wMin=0.2
    c1=2
    c2=2
    s=solution()

    if not isinstance(lb, list):
        lb = [lb] * dim
    if not isinstance(ub, list):
        ub = [ub] * dim


    ######################## Initializations ##############################

    vel=numpy.zeros((PopSize,dim)) #dim=fs#+cost+gama
    pos = numpy.zeros((PopSize, dim))

    pBestScore=numpy.zeros(PopSize)
    #pBestScore.fill(float(0))
    pBestScore.fill(float("inf"))

    pBest=numpy.zeros((PopSize,dim))

    gBest=numpy.zeros(dim)
    gBestScore=float("inf")
    #gBestScore=float(0)


    for i in range(dim):
        pos[:, i] = numpy.random.uniform(0.0000001,1, PopSize) * (ub[i] - lb[i]) + lb[i]

    for j in range(0,PopSize):
        for i in range(dim-2):
            posOut=transferFun(pos[j,i],pos[j,i],fid) # use the transfer function S2 to convert to binary
            pos[j,i]=posOut

        if i==(dim-2):
            pos[:, i]=numpy.random.uniform(0,1)
        if i==(dim-1):
            pos[:, i]=numpy.random.uniform(0,1)

    print(pos)
    convergence_curve=numpy.zeros(iters)

    ############################################
    print("PSO is optimizing  \""+objf.__name__+"\"")

    timerStart=time.time()
    s.startTime=time.strftime("%Y-%m-%d-%H-%M-%S")

    lst = []

    for l in range(0,iters):
        for i in range(0,PopSize):
            #pos[i,:]=checkBounds(pos[i,:],lb,ub)
            for j in range(dim):
                pos[i, j] = numpy.clip(pos[i,j], lb[j], ub[j])

            while numpy.sum(pos[i,:])==0:
                 pos[i,:]=numpy.random.randint(2, size=(1, dim))

            print("particle length: "+ str(len(pos[i,:])))
            #print(pos[i,:])

            fitness=objf(pos[i,:], alpha)
            print("the fitness is ", fitness)

            lst.append([list(pos[i, :]), fitness])

            if(pBestScore[i]>fitness):
                print(f"******* pBestScore[i]= {pBestScore[i]}, fitness= {fitness} *********")

                pBestScore[i]=fitness
                pBest[i,:]=pos[i,:].copy()

            if(gBestScore>fitness):
                print(f"******* pBestScore= {pBestScore}, fitness= {fitness} *********")
            #if(gBestScore<fitness):
                gBestScore=fitness
                gBest=pos[i,:].copy()

        #lst = [item for sublist in lst for item in sublist]



        #Update the W of PSO
        w=wMax-l*((wMax-wMin)/iters)

        for i in range(0,PopSize):
            for j in range (0, dim):
                r1=random.random()
                r2=random.random()
                vel[i,j]=w*vel[i,j]+c1*r1*(pBest[i,j]-pos[i,j])+c2*r2*(gBest[j]-pos[i,j])

                if(vel[i,j]>Vmax):
                    vel[i,j]=Vmax

                if(vel[i,j]<-Vmax):
                    vel[i,j]=-Vmax

                #pos[i,j]=pos[i,j]+vel[i,j]
                tempV=vel[i,j]
                tempP=pos[i,j]

                posOut=transferFun(tempP,tempV,fid) # convert to binary using S2
                pos[i,j] = posOut

                #if j==(dim-2):
                pos[:, -2] = numpy.random.uniform(0,1)
                    #if j==(dim-1):
                pos[:, -1] = numpy.random.uniform(0,1)



        convergence_curve[l]=gBestScore
        #store all global best particles

        if (l%1==0):
               print(['At iteration '+ str(l+1) + ' the best fitness is '+ str(gBestScore)])

    pd.DataFrame(lst).to_csv(f'results/WOA_PSO/svd/alpha/population_fitness_PSO_alpha={alpha}_{datetime.datetime.now()}_run.csv', index=False)

    #print(statistics.mean(convergence_curve))
    timerEnd=time.time()
    s.endTime=time.strftime("%Y-%m-%d-%H-%M-%S")
    s.executionTime=timerEnd-timerStart
    s.convergence=convergence_curve
    s.bestIndividual=gBest
    s.best=statistics.mean(convergence_curve)
    s.optimizer="PSO"
    s.objfname=objf.__name__

    return s


