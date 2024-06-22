# -*- coding: utf-8 -*-
"""
Created on Mon May 16 14:19:49 2016

@author: hossam
"""
import random
import numpy
import math
from solution import solution
import time

from transferFun import sigmoid_transfer
import pandas as pd
import datetime

def WOA(objf,lb,ub,dim,SearchAgents_no,Max_iter, alpha):


    #dim=30
    #SearchAgents_no=50
    #lb=-100
    #ub=100
    #Max_iter=500
    if not isinstance(lb, list):
        lb = [lb] * dim
    if not isinstance(ub, list):
        ub = [ub] * dim
        
    
    # initialize position vector and score for the leader
    Leader_pos=numpy.zeros(dim)
    Leader_score=float("inf")  #change this to -inf for maximization problems
    
    
    #Initialize the positions of search agents
    Positions = numpy.zeros((SearchAgents_no, dim))
    for i in range(dim):
        #Positions[:, i] = numpy.random.uniform(0,1,SearchAgents_no) *(ub[i]-lb[i])+lb[i]
        #rc1 = abs(float(random.random())) # I modified to this
        #rc0 = 1 - rc1
        Positions[:, i] = numpy.random.choice(2, SearchAgents_no) #, p=[rc1, rc0])
        #Positions[:, -1] = numpy.random.uniform(0.000001, 0.99999, 1)
        #Positions[:, -2] = numpy.random.uniform(0.000001, 0.99999, 1)
        print(Positions[:, i])

        #Initialize convergence
    convergence_curve=numpy.zeros(Max_iter)
    
    
    ############################
    s=solution()

    print("WOA_PSO is optimizing  \""+objf.__name__+"\"")

    timerStart=time.time() 
    s.startTime=time.strftime("%Y-%m-%d-%H-%M-%S")
    ############################
    
    t=0  # Loop counter
    
    # Main loop

    lst = []

    while t<Max_iter:
        for i in range(0,SearchAgents_no):
            
            # Return back the search agents that go beyond the boundaries of the search space
            
            #Positions[i,:]=checkBounds(Positions[i,:],lb,ub)
            for j in range(dim): #except the last two, C and gama
                Positions[i,j]=numpy.clip(Positions[i,j], lb[j], ub[j])
            
            # Calculate objective function for each search agent

            Positions[i, -2] = numpy.random.uniform(0.0000000001, 0.99999999999)
            Positions[i, -1] = numpy.random.uniform(0.0000000001, 0.99999999999)

            fitness=objf(Positions[i,:], alpha)
            lst.append([list(Positions[i, :]), fitness])

            print('Fitness=', fitness)
            
            # Update the leader
            if fitness<Leader_score: # Change this to > for maximization problem
                Leader_score=fitness # Update alpha
                Leader_pos=Positions[i,:].copy() # copy current whale position into the leader position

        
        
        a=2-t*((2)/Max_iter) # a decreases linearly fron 2 to 0 in Eq. (2.3)
        
        # a2 linearly decreases from -1 to -2 to calculate t in Eq. (3.12)
        a2=-1+t*((-1)/Max_iter)
        
        # Update the Position of search agents 
        for i in range(0,SearchAgents_no):
            r1=random.random() # r1 is a random number in [0,1]
            r2=random.random() # r2 is a random number in [0,1]
            
            A=2*a*r1-a  # Eq. (2.3) in the paper
            C=2*r2      # Eq. (2.4) in the paper
            
            
            b=1              #  parameters in Eq. (2.5)
            l=(a2-1)*random.random()+1   #  parameters in Eq. (2.5)
            
            p = random.random()        # p in Eq. (2.6)
            
            for j in range(0, dim-2):
                
                if p<0.5:
                    if abs(A)>=1:
                        rand_leader_index = math.floor(SearchAgents_no*random.random())
                        X_rand = Positions[rand_leader_index, :]
                        D_X_rand=abs(C*X_rand[j]-Positions[i,j]) 
                        Positions[i,j]=X_rand[j]-A*D_X_rand      
                        
                    elif abs(A)<1:
                        D_Leader=abs(C*Leader_pos[j]-Positions[i,j]) 
                        Positions[i,j]=Leader_pos[j]-A*D_Leader      
                    
                    
                elif p>=0.5:
                  
                    distance2Leader=abs(Leader_pos[j]-Positions[i,j])
                    # Eq. (2.5)
                    Positions[i,j]=distance2Leader*math.exp(b*l)*math.cos(l*2*math.pi)+Leader_pos[j]

                #print("***************^^^^^^^^^^^^Use S2 function to convrert from continuous to discrete ^^^^^**********************")
                Positions[i, j] = sigmoid_transfer(Positions[i, j])

                #print(Positions[i, j])
        
        convergence_curve[t]=Leader_score
        if (t%1==0):
               print(['At iteration '+ str(t)+ ' the best fitness is '+ str(Leader_score)]);
        t=t+1

    pd.DataFrame(lst).to_csv(
        f'results/WOA_PSO/svd/alpha/svd_population_fitness_WOA_alpha_{alpha}_{datetime.datetime.now()}.csv', index=False)

    timerEnd=time.time()  
    s.endTime=time.strftime("%Y-%m-%d-%H-%M-%S")
    s.executionTime=timerEnd-timerStart
    s.convergence=convergence_curve
    s.optimizer="WOA"
    s.objfname=objf.__name__
    s.best = Leader_score
    s.bestIndividual = Leader_pos
    
    
    
    return s


