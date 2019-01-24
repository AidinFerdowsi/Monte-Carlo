# -*- coding: utf-8 -*-
"""
Created on Thu Jan 24 15:54:43 2019

@author: Aidin
"""
import numpy as np
import matplotlib.pyplot as plt
from gridWorld import standardGrid, negativeGrid
from iterativePolicyEvaluation import printValues, printPolicy
from monteCarloControl import argMax

GAMMA = 0.9
ACTIONS = {'U','D','L','R'}

def randomAction(a, grid, eps = 0.4):
    
    
    p = np.random.random()
    
    if p < (1 - eps):
        return a
    else:
#        return np.random.choice(list(ACTIONS))
        return np.random.choice(list(grid.actions[grid.currentState()]))
    
    

def playGame(grid,policy):
    s = (2,0)
    grid.setState(s)
    a = randomAction(policy[s],grid)
    
    
    statesActionRewards = [(s,a,0)]
    while True:
        r = grid.move(a)
        s = grid.currentState()
        
        if grid.gameOver():
            statesActionRewards.append((s,None,r))
            break
        else:
            a = randomAction(policy[s],grid)
            statesActionRewards.append((s,a,r))
    G = 0
    statesActionReturns = []
    firstVisit = True
    for s, a, r in reversed(statesActionRewards):
        if firstVisit:
            firstVisit = False
        else:
            statesActionReturns.append((s,a, G))
        G = r + GAMMA*G
    statesActionReturns.reverse()
    return statesActionReturns


if __name__ == '__main__':
#    grid = standardGrid()
    
    grid = negativeGrid(stepCost = - 0.1)
    
    
    print ("Rewards:")
    printValues(grid.rewards,grid)
    
    policy = {}    
    for s in grid.actions.keys():
        policy[s] = np.random.choice(list(grid.actions[s]))
    
    Q = {}
    returns = {}
    
    states = grid.allStates()
    
    for s in states:
        if s in grid.actions:
            Q[s] = {}
            for a in list(ACTIONS):
                Q[s][a] = 0
                returns[(s,a)] = []
        else:
            pass
        
    deltas = []
    
    
    for t in range(10000):
        if t%100 == 0:
            print(t)
        
        biggestChange = 0
        statesActionReturns = playGame(grid,policy)
        seenStateActions = set()
        for s, a, G in statesActionReturns:
#            print (s)
            stateAction = (s,a)
            if s == (2,2):
                print (a)
            if stateAction not in seenStateActions:
                oldQ = Q[s][a]
                returns[stateAction].append(G)
                Q [s][a] = np.mean(returns[stateAction])
                biggestChange = max(biggestChange,np.abs(oldQ - Q[s][a]))
                seenStateActions.add(stateAction)
        deltas.append(biggestChange)
        
        for s in policy.keys():
            policy[s] = argMax(Q[s])[0]
    
    plt.figure(figsize=(10,4))        
    plt.plot(deltas)
    plt.show()
    
    print("Final Policy")
    printPolicy(policy,grid)
    
    
    V = {}
    for s, Qs in Q.items():
        V[s] = argMax(Q[s])[1]
    
    print("Final Values")
    printValues(V,grid)