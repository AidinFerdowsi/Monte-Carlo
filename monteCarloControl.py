# -*- coding: utf-8 -*-
"""
Created on Mon Jan 21 15:25:48 2019

@author: Aidin
"""
import numpy as np
from gridWorld import standardGrid
#import matplotlib.pyplot as plt
from iterativePolicyEvaluation import printValues, printPolicy



EPSILON = 1e-5
GAMMA = 0.9
ACTIONS = {'U','D','L','R'}

def playGame(grid,policy):
    startStates = list(grid.actions.keys())
    startIndex = np.random.choice((len(startStates)))
    grid.setState(startStates[startIndex])
    
    s = grid.currentState()
    a = np.random.choice(list(ACTIONS))
    
    statesActionsRewards = [(s,a,0)]
    while True:
        oldS = grid.currentState()
        r = grid.move(a)
        s = grid.currentState()
        if oldS == s:
            statesActionsRewards.append((s,None,-100))
            break
        elif grid.gameOver():
            statesActionsRewards.append((s,None,r))
            break
        else:
            a = policy[s]
            statesActionsRewards.append((s,a,r))
    
    G = 0
    statesReturns = []
    firstVisit = True
    for s, a, r in reversed(statesActionsRewards):
        if firstVisit:
            firstVisit = False
        else:
            statesReturns.append((s,a, G))
        G = r+ GAMMA*G
    statesReturns.reverse()
    return statesReturns

def argMax(dic):
    maxKey = None
    maxVal = float('-inf')
    
    for k, v in dic.items():
        if v> maxVal:
            maxVal = v
            maxKey = k
    return maxKey, maxVal

if __name__ == '__main__':
    grid = standardGrid()
    
    
    print ("Rewards:")
    printValues(grid.rewards,grid)
    
    policy = {}    
    for s in grid.actions.keys():
        policy[s] = np.random.choice(list(ACTIONS))
    
    Q = {}
    returns = {}
    
    states = grid.allStates()
    
    for s in states:
        if s in grid.actions:
            Q[s] = {}
            for a in ACTIONS:
                Q[s][a] = 0
                returns[(s,a)] = []
        else:
            pass
        
    deltas = []
    for t in range(10000):
        print(t)
        
        biggestChange = 0
        statesActionReturns = playGame(grid,policy)
        seenStateActions = set()
        for s, a, G in statesActionReturns:
            
            stateAction = (s,a)
            if stateAction not in seenStateActions:
                oldQ = Q[s][a]
                returns[stateAction].append(G)
                Q [s][a] = np.mean(returns[stateAction])
                biggestChange = max(biggestChange,np.abs(oldQ - Q[s][a]))
        deltas.append(biggestChange)
        
        for s in policy.keys():
            policy[s] = argMax(Q[s])[0]
#    plt.plot(deltas)
#    plt.show()
    
    print("Final Policy")
    printPolicy(policy,grid)
    
    
    V = {}
    for s, Qs in Q.items():
        V[s] = argMax(Q[s])[1]
    
    print("Final Values")
    printValues(V,grid)
    