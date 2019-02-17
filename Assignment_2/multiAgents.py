# multiAgents.py
# --------------
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util
import math

from game import Agent

class ReflexAgent(Agent):
	"""
	A reflex agent chooses an action at each choice point by examining
	its alternatives via a state evaluation function.

	The code below is provided as a guide.  You are welcome to change
	it in any way you see fit, so long as you don't touch our method
	headers.
	"""


	def getAction(self, gameState):
		"""
		You do not need to change this method, but you're welcome to.

		getAction chooses among the best options according to the evaluation function.

		Just like in the previous project, getAction takes a GameState and returns
		some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
		"""
		# Collect legal moves and successor states
		legalMoves = gameState.getLegalActions()

		# Choose one of the best actions
		scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
		bestScore = max(scores)
		bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
		chosenIndex = random.choice(bestIndices) # Pick randomly among the best

		"Add more of your code here if you want to"

		return legalMoves[chosenIndex]

	def evaluationFunction(self, currentGameState, action):
		"""
		Design a better evaluation function here.

		The evaluation function takes in the current and proposed successor
		GameStates (pacman.py) and returns a number, where higher numbers are better.

		The code below extracts some useful information from the state, like the
		remaining food (newFood) and Pacman position after moving (newPos).
		newScaredTimes holds the number of moves that each ghost will remain
		scared because of Pacman having eaten a power pellet.

		Print out these variables to see what you're getting, then combine them
		to create a masterful evaluation function.
		"""
		# Useful information you can extract from a GameState (pacman.py)
		successorGameState = currentGameState.generatePacmanSuccessor(action)
		newPos = successorGameState.getPacmanPosition()
		newFood = successorGameState.getFood()
		newGhostStates = successorGameState.getGhostStates()
		newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

		"*** YOUR CODE HERE ***"

		"""
		Three (sort of 4) things need to be considered here: where is the food at, where are the
		power up things at, and where are the ghosts at (plus what is our penalty for stopping)
		"""

		"""
		These scores (right below) were tweaked until we got an average score over 1,000. The implimentation
		for the decay rate was taken from lecture 8 slides, slide 23, meaning that each move required
		to get from point a to point b has exponentially less value
		"""
		foodScore = 1
		powerUpScore = 2
		ghostScore = -5.5

		decayRate = 0.25 #our decay rate for n moves
		stop = -25 #our penalty for stopping
		score = 0 #our score holder (higher is better)


		"""
		Here we score the food pellets, first checking if the next position has a food pellet (positive value),
		then looking at the rest of the food pellets and checking their distance (negative value). Since a higher score is better,
		I had to mess with the signs a bit so that farther away is worse (greater negative value)
		"""
		score += currentGameState.hasFood(newPos[0], newPos[1]) * foodScore #The next position

		for food in newFood.asList(): #Score the rest of them using the decay rate method from lecture slides
			score -= foodScore * (1 - math.exp(-1.0 * decayRate * util.manhattanDistance(newPos, food)))


		"""
		Here we look for the ghost, since being farther away from the ghosts is better, we assign a positive score, which
		kind of ends up being similar logic to something like minesweeper. If we are very close to a ghost, that's really bad
		so we assign a very large negative score since that means we can fail the game if the ghost moves onto us. If the ghosts
		are scared, we don't care where they are so we don't do this part of the score.
		"""
		if max(newScaredTimes) == 0:
			for ghostPos in newGhostStates:
				score += ghostScore * math.exp(-1.0 * decayRate * util.manhattanDistance(newPos, ghostPos.getPosition()))
				if (util.manhattanDistance(newPos, ghostPos.getPosition())<2): #If we are either 1 or 0 moves from a ghost
					score -= 99999


		"""
		This is the same method as the food, except with the power ups
		"""
		for powerUp in currentGameState.data.capsules:
			if newPos == powerUp:
				score += powerUpScore
			else:
				score -= powerUpScore * math.exp(-1.0 * decayRate * util.manhattanDistance(newPos, powerUp))


		"""
		If we decide to stop, we need to apply some penalty since nothing is being achieved
		"""
		if currentGameState.getPacmanPosition() == newPos:
			score += stop

		return score


def scoreEvaluationFunction(currentGameState):
	"""
	This default evaluation function just returns the score of the state.
	The score is the same one displayed in the Pacman GUI.

	This evaluation function is meant for use with adversarial search agents
	(not reflex agents).
	"""
	return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
	"""
	This class provides some common elements to all of your
	multi-agent searchers.  Any methods defined here will be available
	to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

	You *do not* need to make any changes here, but you can if you want to
	add functionality to all your adversarial search agents.  Please do not
	remove anything, however.

	Note: this is an abstract class: one that should not be instantiated.  It's
	only partially specified, and designed to be extended.  Agent (game.py)
	is another abstract class.
	"""

	def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
		self.index = 0 # Pacman is always agent index 0
		self.evaluationFunction = util.lookup(evalFn, globals())
		self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
	"""
	Your minimax agent (question 2)
	"""

	#Takes the current state, min/max behavior (str), agent # (int), and depth (int)
	def min_max_tree_builder(self, state, behavior, agent, depth):
		"""
		This function does the bulk of the work. It will switch back and forth between
		Pacman (maximizing agent) and the ghosts (minimizing agents) to build a min/max
		tree of some depth n (as described in the project writeup). As it iterates through,
		at each level it will return the min/max score and the corresponding min/max move.
		Once either the depth has been hit or Pacman wins/loses, it will return the state.
		"""

		#If we have won, lost, or reached the inputted depth, return the final score
		if (state.isWin() == True) or (state.isLose() == True) or (depth == 0):
			return self.evaluationFunction(state), Directions.STOP

		#Get the legal actions to test for the agent, 0 = Pacman, 1+ = ghosts
		legal_actions = state.getLegalActions(agent)

		#This holds the possible scores
		scores = []

		"""
		This function runs it for Pacman, building out a tree by calling the ghosts for every
		legal Pacman action and appending the scores to a list. It then takes the max score since
		it is a maximizing agent and returns that and the action. 
		"""
		if behavior == 'max':
			#Build out the tree, calling the next level with ghosts for ever Pacman action
			for actions in legal_actions:
				scores.append(self.min_max_tree_builder(state=state.generateSuccessor(agent, actions), behavior='min', agent=1, depth=depth-1)[0])
			max_score = max(scores) #The best score in the list
			max_index = scores.index(max_score) #The index of the best action to take
			return max_score, legal_actions[max_index]

		"""
		This is pretty much identical to the behvaior == 'max' above, but instead is for a 
		minimizing agent (ghosts) and calls Pacman for the next level of the tree. We also need
		to check to see if it is the last ghost of not, becuase if it is not then we haven't gone 
		down to the next depth and we don't call Pacman yet.
		"""
		if behavior == 'min':
			#If we are not at the last agent, run as above, calling the next agent (ghost) and not moving down
			if agent != state.getNumAgents() - 1:
				for actions in legal_actions:
					scores.append(self.min_max_tree_builder(state=state.generateSuccessor(agent, actions), behavior='min', agent=agent+1, depth=depth)[0])
			#If this is the last ghost
			if agent == state.getNumAgents() - 1:
				for actions in legal_actions:
					scores.append(self.min_max_tree_builder(state=state.generateSuccessor(agent, actions), behavior='max', agent=0, depth=depth-1)[0])
			min_score = min(scores) #The lowest score in the list
			min_index = scores.index(min_score) #The index of the lowest score
			return min_score, legal_actions[min_index]

		return "Failure"

	def getAction(self, gameState):
		"""
		Returns the minimax action from the current gameState using self.depth
		and self.evaluationFunction.

		Here are some method calls that might be useful when implementing minimax.

		gameState.getLegalActions(agentIndex):
		Returns a list of legal actions for an agent
		agentIndex=0 means Pacman, ghosts are >= 1

		gameState.generateSuccessor(agentIndex, action):
		Returns the successor game state after an agent takes an action

		gameState.getNumAgents():
		Returns the total number of agents in the game

		gameState.isWin():
		Returns whether or not the game state is a winning state

		gameState.isLose():
		Returns whether or not the game state is a losing state
		"""
		"*** YOUR CODE HERE ***"

		#Calls the function above, returns the direction (the [1] at the end)
		return self.min_max_tree_builder(state=gameState, behavior='max', agent=0, depth=self.depth*2)[1]

class AlphaBetaAgent(MultiAgentSearchAgent):
	"""
	Your minimax agent with alpha-beta pruning (question 3)
	"""
	#Takes the current state, min/max behavior (str), agent # (int), depth (int), max score (int), min score (int)
	def min_max_tree_builder(self, state, behavior, agent, depth, max_score_A, min_score_B):
		"""
		This function runs nearly identically to the min max agent from question 2,
		however it uses a min and max argument in the function call to determine if 
		the node should be considered or not. The code was copied from question 2,
		and then updates to fit the alpha beta pruning. Most of the changes in the code
		are the lines below the recursive function call. 
		"""

		#If we have won, lost, or reached the inputted depth, return the final score
		if (state.isWin() == True) or (state.isLose() == True) or (depth == 0):
			return self.evaluationFunction(state), Directions.STOP

		#Get the legal actions to test for the agent, 0 = Pacman, 1+ = ghosts
		legal_actions = state.getLegalActions(agent)

		"""
		This function runs it for Pacman, building out a tree by calling the ghosts for every
		legal Pacman action and appending the scores to a list. It then checks the score for that
		move against the max_score_A from the function call and updates the score if it is better.
		"""
		if behavior == 'max':
			max_score = -99999 #Set the intial max to a very low value
			max_action = []
			#Build out the tree, calling the next level with ghosts for ever Pacman action
			for actions in legal_actions:
				action_score = self.min_max_tree_builder(state=state.generateSuccessor(agent, actions), behavior='min', agent=1, depth=depth-1, max_score_A=max_score_A, min_score_B=min_score_B)[0]
				max_score_A = max(action_score, max_score_A)
				#If this is the best score, update the score
				if action_score > max_score:
					max_score = action_score
					max_action = actions
				#If our current value is better than the min value, return immediatley and
				#don't explore anymore nodes (defenition of alpha beta pruning)
				if max_score > min_score_B:
					break
			#Return the best score (or first best) and the appropriate action
			return max_score, max_action

		"""
		This is pretty much identical to the behvaior == 'max' above, but instead is for a 
		minimizing agent (ghosts) and calls Pacman for the next level of the tree. We also need
		to check to see if it is the last ghost of not, becuase if it is not then we haven't gone 
		down to the next depth and we don't call Pacman yet.
		"""
		if behavior == 'min':
			min_score = 99999
			min_action = []
			#If we are not at the last agent, run as above, calling the next agent (ghost) and not moving down
			if agent != state.getNumAgents() - 1:
				for actions in legal_actions:
					action_score = self.min_max_tree_builder(state=state.generateSuccessor(agent, actions), behavior='min', agent=agent+1, depth=depth, max_score_A=max_score_A, min_score_B=min_score_B)[0]
					min_score_B = min(action_score, min_score_B)
					#If this is the best (worst) score, update the min score
					if action_score < min_score:
						min_score = action_score
						min_action = actions
					#If our current value is better (worse) than the max value, return immediatley and
					#don't explore anymore nodes (defenition of alpha beta pruning)
					if max_score_A > min_score: 
						break
			#If this is the last ghost
			if agent == state.getNumAgents() - 1:
				for actions in legal_actions:
					action_score = self.min_max_tree_builder(state=state.generateSuccessor(agent, actions), behavior='max', agent=0, depth=depth-1, max_score_A=max_score_A, min_score_B=min_score_B)[0]
					min_score_B = min(action_score, min_score_B)
					#If this is the best (worst) score, update the min score
					if action_score < min_score:
						min_score = action_score
						min_action = actions
					#If our current value is better (worse) than the max value, return immediatley and
					#don't explore anymore nodes (defenition of alpha beta pruning)
					if max_score_A > min_score: 
						break
			#Return the minimum score (or first worst score) and the appropriate action
			return min_score, min_action

		return "Failure"

	def getAction(self, gameState):
		"""
		Returns the minimax action using self.depth and self.evaluationFunction
		"""
		"*** YOUR CODE HERE ***"
		#Calls the function above, returns the direction (the [1] at the end)
		return self.min_max_tree_builder(state=gameState, behavior='max', agent=0, depth=self.depth*2, max_score_A=-99999, min_score_B=99999)[1]

class ExpectimaxAgent(MultiAgentSearchAgent):
	"""
	  Your expectimax agent (question 4)
	"""

	#Takes the current state, min/max behavior (str), agent # (int), and depth (int)
	def expectimax_tree_builder(self, state, behavior, agent, depth):
		"""
		This is copied from question 2 exactly, the only part changed is the ghosts
		part (behavior = 'min') to fit the expectimax description. The changes are 
		described in that section. 
		"""

		#If we have won, lost, or reached the inputted depth, return the final score
		if (state.isWin() == True) or (state.isLose() == True) or (depth == 0):
			return self.evaluationFunction(state), Directions.STOP

		#Get the legal actions to test for the agent, 0 = Pacman, 1+ = ghosts
		legal_actions = state.getLegalActions(agent)

		#This holds the possible scores
		scores = []

		"""
		This function runs it for Pacman, building out a tree by calling the ghosts for every
		legal Pacman action and appending the scores to a list. It then takes the max score since
		it is a maximizing agent and returns that and the action. 
		"""
		if behavior == 'max':
			#Build out the tree, calling the next level with ghosts for ever Pacman action
			for actions in legal_actions:
				scores.append(self.expectimax_tree_builder(state=state.generateSuccessor(agent, actions), behavior='min', agent=1, depth=depth-1)[0])
			max_score = max(scores) #The best score in the list
			max_index = scores.index(max_score) #The index of the best action to take
			return max_score, legal_actions[max_index]

		"""
		This is pretty much identical to the behvaior == 'max' above, but instead is for a 
		minimizing agent (ghosts) and calls Pacman for the next level of the tree. We also need
		to check to see if it is the last ghost of not, becuase if it is not then we haven't gone 
		down to the next depth and we don't call Pacman yet.
		"""
		if behavior == 'min':

			random_score = 0 #Holds the sum of the scores for each move
			
			#If we are not at the last agent, run as above, calling the next agent (ghost) and not moving down, summing the possible moves
			if agent != state.getNumAgents() - 1:
				for actions in legal_actions:
					random_score += self.expectimax_tree_builder(state=state.generateSuccessor(agent, actions), behavior='min', agent=agent+1, depth=depth)[0]
			#If this is the last ghost, sum the scores of the possible moves
			if agent == state.getNumAgents() - 1:
				for actions in legal_actions:
					random_score += self.expectimax_tree_builder(state=state.generateSuccessor(agent, actions), behavior='max', agent=0, depth=depth-1)[0]
			#Return the average score for each ghost move, Directions.STOP just serves as a placeholder
			#since the code was set up to also return a direction
			return random_score/len(legal_actions), Directions.STOP

		return "Failure"

	def getAction(self, gameState):
		"""
		Returns the expectimax action using self.depth and self.evaluationFunction

		All ghosts should be modeled as choosing uniformly at random from their
		legal moves.
		"""
		"*** YOUR CODE HERE ***"
		#Calls the function above, returns the direction (the [1] at the end)
		return self.expectimax_tree_builder(state=gameState, behavior='max', agent=0, depth=self.depth*2)[1]

def betterEvaluationFunction(currentGameState):
	"""
	Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
	evaluation function (question 5).

	DESCRIPTION: Using the same logic as I did in question 1, assign some score to 
	food, power ups, and ghosts (messed around with until passing) and a decay rate
	(again, messed around with until passing). These are identical to the ones I used
	in question 1. Score every food pellet using the decay rate function from class, and 
	give that a negative value since being farther away from food (higher number) is bad. 
	Then, check the distances to the ghosts, and a assign a positive value since being farther
	away from ghosts (high number) is good. If we are right next to a ghost, assign a very very
	low number since we could lose the game. If the ghosts are scared, we don't care where they are
	at, and do not score the ghosts. Then, using the same methodology as food, check the power
	up pellets. That gives us our final score.
	"""
	"*** YOUR CODE HERE ***"
		
	#Copied from question 1 and updated to current instead of succesor
	pos = currentGameState.getPacmanPosition()
	food = currentGameState.getFood()
	ghostStates = currentGameState.getGhostStates()
	scaredTimes = [ghostState.scaredTimer for ghostState in ghostStates]
	"""
	Three things need to be considered here: where is the food at, where are the
	power up things at, and where are the ghosts at
	"""

	"""
	These scores were taken from question 1
	"""
	foodScore = 1
	powerUpScore = 2
	ghostScore = -5.5

	decayRate = 0.25 #our decay rate for n moves
	score = 0 #our score holder (higher is better)


	"""
	Here we are looking at the distance for all of the food pellets using the decay rate
	as described in class. Since being close to food is better, we want a negative score
	if food is far away.
	"""
	for food in food.asList(): #Score the rest of them using the decay rate method from lecture slides
		score -= foodScore * (1 - math.exp(-1.0 * decayRate * util.manhattanDistance(pos, food)))


	"""
	If the ghosts are not scared (max(scaredTimes == 0)), we want a positive value, since it is
	better if the ghosts are far away. If the ghosts are very close (1 spot away), we consider that 
	to be very bad and make it -99999 so it would only take that option last.
	"""
	if max(scaredTimes) == 0:
		for ghostPos in ghostStates:
			score += ghostScore * math.exp(-1.0 * decayRate * util.manhattanDistance(pos, ghostPos.getPosition()))
			if (util.manhattanDistance(pos, ghostPos.getPosition())<2): #If we are either 1 or 0 moves from a ghost
				score -= 99999


	"""
	This is the same method as the food, except with the power ups.
	"""
	for powerUp in currentGameState.data.capsules:
		score -= powerUpScore * math.exp(-1.0 * decayRate * util.manhattanDistance(pos, powerUp))

	return score

# Abbreviation
better = betterEvaluationFunction
