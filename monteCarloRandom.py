# -*- coding: utf-8 -*-
"""
Created on Mon Jan 21 11:57:49 2019

@author: Aidin
"""
import numpy as np
from gridWorld import standardGrid
from iterativePolicyEvaluation import printValues, printPolicy



EPSILON = 1e-5
GAMMA = 0.9
ACTIONS = {'U','D','L','R'}

def randomAction(a):
    
    
    p = np.random.random()
    
    if p < 0.5:
        return a
    else:
        temp = list(ACTIONS)
        temp.remove(a)
        return np.random.choice(temp)
    
    
def playGame(grid,policy):
    startStates = list(grid.actions.keys())
    startIndex = np.random.choice((len(startStates)))
    grid.setState(startStates[startIndex])
    
    s = grid.currentState()
    statesRewards = [(s,0)]
    while not grid.gameOver():
        a = policy[s]
        a = randomAction(a)
        r = grid.move(a)
        s = grid.currentState()
        statesRewards.append((s,r))
    G = 0
    statesReturns = []
    firstVisit = True
    for s, r in reversed(statesRewards):
        if firstVisit:
            firstVisit = False
        else:
            statesReturns.append((s,G))
        G = r+ GAMMA*G
    statesReturns.reverse()
    return statesReturns

if __name__ == '__main__':
    grid = standardGrid()
    
    
    print ("Rewards:")
    printValues(grid.rewards,grid)
    
    policy = {
            (2,0): "U",
            (1,0): "U",
            (0,0): "R",
            (0,1): "R",
            (0,2): "R",
            (1,2): "U",
            (2,1): "L",
            (2,2): "U",
            (2,3): "L",
    }
   
    V = {}
    returns = {}
    states = grid.allStates()
    for s in states:
        if s in grid.actions:
            returns[s] = []
        else:
            V[s] = 0
    
    
    for t in range (5000):
        statesReturns = playGame(grid,policy)
        visitedStates = set()
        
        for s , G in statesReturns:
            if s not in visitedStates:
                returns[s].append(G)
                V[s] = np.mean(returns[s])
                visitedStates.add(s)
    
    print("Values:")
    printValues(V,grid)
    
    print("Policy:")
    printPolicy(policy,grid)