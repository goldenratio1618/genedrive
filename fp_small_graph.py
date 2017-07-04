import numpy as np
from numpy import linalg as la
from main import genPayoffMatrix
from simulate import cNorm
from copy import copy
from matplotlib import pyplot as plt

def genStates(graph):
    """ Generates array of all possible states
    Each state is an array of 1 or 0 for each vertex: 1 = mutant, 0 = WT
    """
    states = [[]]
    for key in graph.keys():
        newStates = []
        for state in states:
            newStates.append(state + [1])
            newStates.append(state + [0])
        states = newStates
    return states



def genTransitionMatrix(graph, states, payoffMatrix):
    """ Computes the transition matrix for a given graph, set of states, and payoff matrix. """
    matrix = np.zeros((len(states), len(states)))
    for i,state in enumerate(states):
        # supposing we're in a given state, what is the probability of getting to any other state?
        # payoffs of all nodes in this state
        payoffs = {}
        totPayoff = 0
        for loc,mutant in enumerate(state):
            # compute payoff of this node
            payoffs[loc] = 0
            for adjLoc in graph[loc]:
                payoffs[loc] += payoffMatrix[mutant][state[adjLoc]]
            totPayoff += cNorm(payoffs[loc])
        # nothing can reproduce, so we're stuck in this state forever
        if totPayoff == 0:
            matrix[i][i] = 1
            continue
        for loc,mutant in enumerate(state):
            # probability of this node reproducing
            payoffs[loc] /= totPayoff
            # a neighbor is chosen with replacement with probability equal to 1/nbs
            nbs = len(graph[loc])
            for adjLoc in graph[loc]:
                # this location might be replaced with either a mutant or wild-type
                newState1 = copy(state)
                newState2 = copy(state)
                # state assuming we created same-type
                newState1[adjLoc] = mutant
                # state assuming we created opposite-type
                newState2[adjLoc] = 1 - mutant
                j1 = states.index(newState1)
                j2 = states.index(newState2)
                matrix[i][j1] += payoffs[loc].real / nbs
                matrix[i][j2] += payoffs[loc].imag / nbs
    return matrix

def parseGraph(graph):
    adjGridDict = {}
    for line in graph:
        start,end=line.split(",")
        if int(start) in adjGridDict.keys():
            adjGridDict[int(start)].append(int(end))
        else:
            adjGridDict[int(start)] = [int(end)]
        
        if int(end) in adjGridDict.keys():
            adjGridDict[int(end)].append(int(start))
        else:
            adjGridDict[int(end)] = [int(start)]
    return adjGridDict

def initStates(states):
    """ Creates row vector of initial states (equal probability for all states with 1 mutant) """
    init = np.zeros(len(states))
    for i,state in enumerate(states):
        if sum(state) == 1:
            init[i] += 1/len(state)
    return init

def computeFP(graph, a, fb, p):
    states = genStates(graph)
    payoffMatrix = genPayoffMatrix(a, fb, p)
    init = initStates(states)
    matrix = genTransitionMatrix(graph, states, payoffMatrix)
    finalDist = init @ la.matrix_power(matrix, 200)
    err = sum(finalDist[1:-1])
    if err > 1e-6:
        raise ValueError("Error is too high: " + str(err))
    return finalDist[0]
    

def computeFP_node0(graph, a, fb, p):
    states = genStates(graph)
    payoffMatrix = genPayoffMatrix(a, fb, p)
    init = np.zeros(len(states))
    init[len(states) // 2 - 1] = 1    
    matrix = genTransitionMatrix(graph, states, payoffMatrix)
    finalDist = init @ la.matrix_power(matrix, 200)
    err = sum(finalDist[1:-1])
    if err > 1e-6:
        raise ValueError("Error is too high: " + str(err))
    return finalDist[0]
    
        

def main():
    # initialize graph - test case of triangle first
    graph_strs = ["complete_4.txt", "diamond_4.txt", "circle_4.txt", "shovel_4.txt", "line_4.txt", "star_4.txt"]
    # tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),    
    #          (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),    
    #          (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),    
    #          (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),    
    #          (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]
    # for i in range(len(tableau20)):    
    #     r, g, b = tableau20[i]    
    #     tableau20[i] = (r / 255., g / 255., b / 255.)
    graphs = [parseGraph(open("sample_graphs/" + g)) for g in graph_strs]
    a = 0
    p = 1
    fb_arr = np.arange(0.1, 3, 0.1)
    plts = []
    for i,graph in enumerate(graphs):
        fps = [computeFP_node0(graph, a, fb, p) for fb in fb_arr]
        handle, = plt.plot(fb_arr, fps, label=graph_strs[i].replace('.txt', ''))
        plts.append(handle)
    plt.legend(handles=plts)
    plt.xlabel("Wild-Type Fitness")
    plt.ylabel("Fixation Probability")
    plt.title("Fixation Probability for 4-Node Graphs")
    plt.show()

if __name__ == '__main__':
    main()