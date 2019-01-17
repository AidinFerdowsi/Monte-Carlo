# -*- coding: utf-8 -*-
"""
Created on Fri Jan  4 14:51:45 2019

@author: Aidin
"""

import numpy as np
# import matplotlib.pyplot as plt
from gridWorld import standardGrid

EPSILON = 10e-4

def printValues(V,grid):
    for i in range(grid.height):
        print ("-------------------------------")
        for j in range(grid.width):
            v = V.get((i,j),0)
            if v>= 0:
                print(" %.2f|" % v, end="")
            else:
                print("%.2f|" % v, end="")
        print("")

def printPolicy(P,grid):
    for i in range(grid.height):
        print ("-------------------------------")
        for j in range(grid.width):
            a = P.get((i,j), ' ')
            print(" %s |" % a, end="")
        print("")
        
        
if __name__ == '__main__':
        
    grid = standardGrid()
    states = grid.allStates()
    
    V = {}
    
    for s in states:
        V[s] = 0
    gamma = 1.0
    while True:
        biggestChange = 0
        for s in states:
            oldV = V[s]
            if s in grid.actions:
                
                newV = 0
                pAction = 1.0/len(grid.actions[s])
                for a in grid.actions[s]:
                    grid.setState(s)
                    r = grid.move(a)
                    # print (r)
                    newV +=pAction * (r + gamma * V[grid.currentState()])
                    # print (newV)
                V[s] = newV
                biggestChange = max(biggestChange,np.abs(oldV-V[s]))
        if biggestChange < EPSILON:
            break
    print ("values for uniformly random actions:")
    printValues(V,grid)
    print("\n\n")
    
    ### fixed policy ###
    policy = {
            (2,0): "U",
            (1,0): "U",
            (0,0): "R",
            (0,1): "R",
            (0,2): "R",
            (1,2): "R",
            (2,1): "R",
            (2,2): "R",
            (2,3): "U",
    }
    
    printPolicy(policy,grid)
    
    V = {}
    
    for s in states:
        V[s] = 0
    
    gamma = 0.9
    
    while True:
        biggestChange = 0
        for s in states:
            oldV = V[s]
            if s in policy:
                a = policy[s]
                grid.setState(s)
                r = grid.move(a)
                V[s] = r + gamma * V[grid.currentState()]
                biggestChange = max(biggestChange,np.abs(oldV-V[s]))
        if biggestChange < EPSILON:
            break
    print ("values for fixed actions:")
    printValues(V,grid)
    print("\n\n")