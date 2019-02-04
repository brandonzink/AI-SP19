# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):

    stack = util.Stack() #The stack to use
    
    state, actions, cost = problem.getStartState(), [], 0 #The initial starting condition

    visited = [state] #The visited spots

    while (not problem.isGoalState(state)): #Loop until you've reached the goal
      
      children = problem.getSuccessors(state) #Get the potential moves

      for next_pos, action, cost in children: #Loop through the children
        
        if (not next_pos in visited): #If we haven't already visited this node

          stack.push((next_pos, actions+[action], cost+cost)) #Add it to the stack
          visited.append(next_pos) #And add it to the visited states

      state, actions, cost = stack.pop() #Pop the next position

    return  actions #Return the set of steps


def breadthFirstSearch(problem):

    queue = util.Queue() #The queue to use
    
    state, actions, cost = problem.getStartState(), [], 0 #The initial starting condition

    visited = [state] #The visited spots

    while (not problem.isGoalState(state)): #Loop until you've reached the goal
      
      children = problem.getSuccessors(state) #Get the potential moves

      for next_pos, action, cost in children: #Loop through the children
        
        if (not next_pos in visited): #If we haven't already visited this node

          queue.push((next_pos, actions+[action], cost+cost)) #Add it to the queue
          visited.append(next_pos) #And add it to the visited states

      state, actions, cost = queue.pop() #Pop the next position

    return  actions

def uniformCostSearch(problem):
    
    p_queue = util.PriorityQueue() #The priority queue we are going to use
    
    state, actions, cost = problem.getStartState(), [], 0 #The starting state
 
    visited = [(state,0)]

    while (not problem.isGoalState(state)): #While you haven't yet reached the goal
      
      children = problem.getSuccessors(state) #Get the children

      for next_pos, action, cost in children: #Loop through the children

        #This loops through and chacks if we have visited this position before, and if we have,
        #checks to see if the cost is lower. If our new path has a lower cost, then we add that to
        #the priority queue via an update

        better_path = False

        total_cost = problem.getCostOfActions(actions+[action]) #The total cost of the path so far

        for i in range(0, len(visited)):

          prev_pos, prev_cost = visited[i]
          if (next_pos == prev_pos) and (total_cost >= prev_cost): #If we've been here and the previous route was lower cost
            better_path = True

        if better_path == False: #If we haven't been here or this path is more efficent

          p_queue.update((next_pos, actions+[action], total_cost), total_cost) #The dual total cost is redundant, it was just left over from using my previous code
          visited.append((next_pos, total_cost))

      state, actions, cost = p_queue.pop()
    
    return  actions

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

#aStar is exactly the same as UCS, but for the addition of the heuristic cost function
def aStarSearch(problem, heuristic=nullHeuristic):
    
    p_queue = util.PriorityQueue() #The priority queue we are going to use
    
    state, actions, cost = problem.getStartState(), [], 0 #The starting state

    visited=[(state,0)]

    while (not problem.isGoalState(state)): #While you haven't yet reached the goal
      
      children=problem.getSuccessors(state) #Get the children

      for next_pos,action,cost in children: #Loop through the children

        #This loops through and chacks if we have visited this position before, and if we have,
        #checks to see if the cost is lower. If our new path has a lower cost, then we add that to
        #the priority queue via an update

        better_path=False

        total_cost=problem.getCostOfActions(actions+[action]) #The total cost of the path so far

        for i in range(len(visited)):

          prev_pos,prev_cost = visited[i]
          if (next_pos==prev_pos) and (total_cost>=prev_cost): #If we've been here and the previous route was lower cost
            better_path=True

        if better_path == False: #If we haven't been here or this path is more efficent

          total_cost=problem.getCostOfActions(actions+[action]) #The total cost of the path so far
          p_queue.push((next_pos, actions+[action], total_cost), total_cost+heuristic(next_pos, problem))
          visited.append((next_pos, total_cost))

      state, actions, cost = p_queue.pop()
    
    return  actions


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
