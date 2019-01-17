# -*- coding: utf-8 -*-
"""
Created on Fri Jan  4 14:32:15 2019

@author: Aidin
"""

class Grid:
    def __init__(self, height, width, start):
        self.width = width
        self.height = height
        self.y = start[0]
        self.x = start[1]
        
    def set(self,rewards, actions):
        self.rewards = rewards
        self.actions = actions
    
    def setState(self,s):
        self.y = s[0]
        self.x = s[1]
    
    def currentState(self):
        return(self.y,self.x)
        
    def checkTerminal(self,s):
        return s not in self.actions
    
    def move(self,action):
        
        if action in self.actions[(self.y,self.x)]:
            if action == 'U':
                self.y -=1
            elif action == 'D':
                self.y +=1
            elif action == 'R':
                self.x +=1
            elif action == 'L':
                self.x -=1
        return self.rewards.get((self.y,self.x),0)
    
    def reverseMove(self,action):
        
        if action in self.actions[(self.y,self.x)]:
            if action == 'U':
                self.y +=1
            elif action == 'D':
                self.y -=1
            elif action == 'R':
                self.x -=1
            elif action == 'L':
                self.x +=1
        assert (self.currentState() in self.allStates())
        
    def gameOver(self):
        return (self.x,self.y) not in self.actions
    
    def allStates(self):
        return set(list(self.actions.keys()) + list(self.rewards.keys()))
    
def standardGrid():
    grid = Grid(3,4,(2,0))
    rewards = {(0,3): 1,(1,3): -1}
    actions = {
           (0,0) : ('D','R'),
           (0,1) : ('L','R'),
           (0,2) : ('L','D','R'),
           (1,0) : ('U','D'),
           (1,2) : ('U', 'D','R'),
           (2,0) : ('U','R'),
           (2,1) : ('L','R'),
           (2,2) : ('L','R','U'),
           (2,3) : ('L','U'),
     }
    grid.set(rewards,actions)
    return grid

def negativeGrid(stepCost = -0.1):
    grid = standardGrid()
    grid.rewards.update({
           (0,0) : stepCost,
           (0,1) : stepCost,
           (0,2) : stepCost,
           (1,0) : stepCost,
           (1,2) : stepCost,
           (2,0) : stepCost,
           (2,1) : stepCost,
           (2,2) : stepCost,
           (2,3) : stepCost,
           })
    return grid  
    