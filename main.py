from simulate import *
from cmdline import *
from gui import GUI
from gridtools import cluster, countLiveCells
from sys import stdout
from copy import deepcopy
from timeit import default_timer as timer
import numpy as np
import argparse
import datetime
import os

def calcFixProb(args, currParmValue, param, dim, origAdjGrid, dualAdjGrid, payoffMatrix, start, folder, datestr):
    if args.initMutant == -1:
        grid = genRandGrid(dim, prob=args.frac)
    else:
        grid = genRandGridNum(dim, num=args.initMutant)
    # changing small-world-ification; need to re-do smallWorldIfy
    adjGrid = np.copy(origAdjGrid)
    strParam = str(round(currParmValue, 6))
    if args.debug:
        print(param + " = " + strParam + ". Time elapsed: " + str(timer() - start))
    if param == 's':
        smallWorldIfyHeterogeneous(adjGrid, currParmValue, args.heterogeneity, args.replace)
    if param == 'h':
        smallWorldIfyHeterogeneous(adjGrid, args.swc, currParmValue, args.replace)        
    else:
        smallWorldIfyHeterogeneous(adjGrid, args.swc, args.heterogeneity, args.replace)
    if args.debug:
        print("Grid smallworldified. Time elapsed: " + str(timer() - start))
    # these will be arrays of the values for every simulation
    mutants = np.zeros(args.niters)
    fdscp = FDSCP(dim, payoffMatrix, adjGrid, grid, dualAdjGrid)
    
    # run the simulation on this many different, random grids
    for sim in range(args.niters):
        if args.debug:
            print("Running simulation: " + str(sim))
            print("Sim = " + str(sim) + ". Time elapsed: " + str(timer() - start))
        # make file to output live cell count and cluster every step
        if args.output >= 3:
            outfile_steps = open(args.outfile + folder + "data3/" + param + "=" + strParam +\
                "_sim=" + str(sim) + datestr + ".txt", "w")
            outfile_steps.writelines("Step  mutants Cluster\n")
        # reset grid to fresh state
        if args.initMutant == -1:
            grid = genRandGrid(dim, prob=args.frac)
        else:
            grid = genRandGridNum(dim, num=args.initMutant)
        fdscp.grid = grid
        fdscp.init()
        if args.debug:
            print("Grid reset. Time elapsed: " + str(timer() - start))
        steps = args.simlength
        if args.output < 3:
            if args.debug:
                run(fdscp, steps, args.delay, 0, args.visible, 1000, args.test)
            else:
                run(fdscp, steps, args.delay, 0, args.visible, -1, args.test)
                
        else:
            for step in range(steps/args.sample + 1):
                if args.debug:
                    print("Step = " + str(step) + " Time elapsed: " + str(timer() - start))
                # output data to file
                outfile_steps.writelines(str(step * args.sample) + "    " + str(countLiveCells(grid)) + "    " + str(cluster(grid, fdscp.adjGrid)) + "\n")
                # step once
                run(fdscp, args.sample, args.delay, 0, args.visible, -1)
                # make file, and output grid to that file
                if args.output >= 4:
                    outfile_grids = open(args.outfile + folder + "data4/" + "swc=" + strswc + "_sim=" + str(sim) + "_step=" + str(step * args.sample) + datestr + ".txt", "w")
                    printGrid(grid, -1, grid.shape, outfile_grids)
                    outfile_grids.close()

            outfile_steps.close()

        if args.debug:
            print("Simulation finished. Time elapsed: " + str(timer() - start))

        mutants[sim] = countLiveCells(grid)
        
        if args.debug:
            print("Finished running simulation. Time elapsed: " + str(timer() - start))
    
    zero = np.zeros(len(mutants))
    fixationEvents = mutants > 0

    # NOTE: if simulation never ends until one of the species fixates, this will equal the fixation probability
    return np.sum(fixationEvents) / len(fixationEvents)


def main(args):
    start = datetime.datetime.now()
    # all new datafiles will be stored in this folder
    folder = start.strftime("%m-%d-%Y_%H-%M-%S") + "/"
    # this folder is "guaranteed" to be unique, so no need to check if dir exists
    os.mkdir(args.outfile + folder)
    for i in range(1,args.output+1):
        os.mkdir(args.outfile + folder + "/data" + str(i) + "/")
    # add extra parameters
    datestr = "frac=" + str(args.frac) + "_rows=" + str(args.rows) + "_cols=" + \
        str(args.cols) + "_extraspace=" + str(args.extraspace) + "_niters=" + \
        str(args.niters) + "_simlength=" + str(args.simlength) + "_replace=" + \
        str(args.replace) + "_heterogeneity=" + str(args.heterogeneity)

    adjGrid = None
    dim = None
    dualAdjGrid = None
    # load the graph from a file
    if args.graphFile != '':
        graph = open(args.graphFile, 'r').readlines()
        dim,adjGrid,dualAdjGrid = parseGraph(graph)
        # dim must be a numpy array for other functions to work properly
        dim = np.array([dim])

    np.set_printoptions(threshold=np.inf)
        
    if args.output >= 1:
        # this file stores averages of final values across all simulations per swc
        outfile_avg = open(args.outfile + folder + "data1/" + datestr + ".txt", "w")
        # will be structured as a table with these 5 columns
        outfile_avg.writelines("SWC  mutants Std Cluster Std\n")

    start = timer()
    if dim is None:
        dim = np.array([args.rows,args.cols])
    
    payoffMatrix = np.array([[args.bb,args.ab*1j],[args.ab,args.aa]], dtype = np.complex128)
    if adjGrid is None:
        adjGrid = initAdjGrid(torusAdjFunc, dim, args.extraspace)

    if args.debug:
        print("payoff matrix (mutant is second row & column):")
        print(payoffMatrix)
        print("Initialized simulation. Time elapsed: " + str(timer() - start))
    # original torus adjacency grid, to be used as fresh template for small-world
    origAdjGrid = np.copy(adjGrid)
    # amount of small-world-ification to do
    swc = args.minswc

    if args.debug and args.graphFile != '':
        print("adjGrid = " + str(adjGrid))
        print("dualAdjGrid = " + str(dualAdjGrid))
        

    # "fudge factor" needed because of floating-point precision limitations
    while swc <= args.maxswc + 0.0000000001:
        fixProb = calcFixProb(args, currParmValue, param, dim, origAdjGrid, dualAdjGrid, payoffMatrix, start, folder, datestr):
        
        # fix outputting to files below
        avglc = round(np.mean(mutants), 3)
        stdlc = round(np.std(mutants), 3)
        if len(cl_new) > 0:
            avgcl = round(np.mean(cl_new), 6)
            stdcl = round(np.std(cl_new), 6)
        if args.output >= 1:
            outfile_avg.writelines(strswc + "    " + str(avglc) + "    " + str(stdlc) + "    " + str(avgcl) + "    " + str(stdcl) + "\n")
        if args.debug:
            print("Output: " + strswc + "    " + str(avglc) + "    " + str(stdlc) + "    " + str(avgcl) + "    " + str(stdcl))
        print("Mutant final population sizes: " + str(mutants))
        # make and output file of range of different final values in
        # simulations
        if args.output >= 2:
            outfile_final = open(args.outfile + folder + "data2/" + "swc=" + strswc + datestr + ".txt", "w")
            outfile_final.writelines("Run  mutants  Cluster\n")
            for i in range(len(mutants)):
                outfile_final.writelines(str(i) + "    " + str(mutants[i]) + "    " + str(round(cl[i], 6)) + "\n")
            outfile_final.close()
        swc += args.stepswc

        if args.debug:
            print("Finished outputting everything to files. Time elapsed: " + str(timer() - start))
        
    if args.output >= 1:
        outfile_avg.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Gene Drive Analysis Frontend",
                                 epilog="")

    parser.add_argument('-f', '--frac', help="Fraction of mutant cells at beginning",
                        type=float, default=0.35) #
    parser.add_argument('-m', '--initMutant', help="Number of mutant cells at beginning. Overrides -f if not -1 (defaults to 1).",
                        type=int, default=1)
    parser.add_argument('-r', '--rows', help="Number of rows of the grid",
                        type=int, default=32) #
    parser.add_argument('-c', "--cols", help="Number of columns of the grid",
                        type=int, default=64) #
    parser.add_argument('-e', "--extraspace",
                        help="Amount of extra space to add to adjacency grid (this should not be changed in most cases).",
                        type=int, default=5) #
    parser.add_argument('-v', "--visible", help="Number of steps to show grid",
                        type=int, default=-1) #
    parser.add_argument('-n', "--niters", help=("Number of times to run simulation "
                                                "per each value of small world "
                                                "coefficient"),
                        type=int, default=1) #
    parser.add_argument('-l', "--simlength", help="Length of each simulation",
                        type=int, default=-1) #
    parser.add_argument('-s', "--swc", help="Small world coefficient (for lattice only)",
                        type=float, default=0) #
    parser.add_argument('-o', "--output",
                        help=("Specify output format. Options:\n"
                            "0: do not output to file\n"
                            "1: output averages of final values across all "
                            "simulations per given small world coefficient\n"
                            "2: output final values for every simulation\n"
                            "3: output values for every step in every "
                            "simulation\n"
                            "4: output grid state at every step in every "
                            "simulation\n"
                            "Higher numbers also output everything for all lower"
                            " numbers, e.g. 3 will also output 2\n"),
                        type=int, default=0)
                        
    parser.add_argument('-p', '--replace', help="Remove edges when constructing small world",
                        action="store_false", default=True)
                            
    parser.add_argument('-g', '--heterogeneity', help="Heterogeneity of SWN",
                        type=float, default=0)

    parser.add_argument('-d', "--delay", help="Time delay between steps",
                        type=float, default=0)
                        

    parser.add_argument('-aa', "--aa", help="Fitness of A when interacting with A", type=float, default=1)

    parser.add_argument('-ab', "--ab", help="Fitness of A/B when interacting with B/A", type=float, default=2)

    parser.add_argument('-bb', "--bb", help="Fitness of B when interacting with B", type=float, default=4)


    parser.add_argument('-s', "--sample", help="When using output modes 3 or 4, how often should the grid be sampled?",
                        type=int, default=10)

    parser.add_argument('-of', "--outfile", help="Output file to store data in", default="D:/OneDrive/Documents/genedrive_data/")

    parser.add_argument('-gr', "--graphFile",
                        help=("Name of file containing graph structure. If blank, loads Cartesian graph.\n"
                        "If enabled, this setting will override the row and column arguments.\n"
                        "This graph will still be small-worldified - if this is not desired, set small world coefficient to 0.\n"
                        "File should be structured as a sequence of lines A,B. Here A and B are vertices, and this line represents an edge from A to B.\n"
                        "There is currently no way to represent weighted graphs with this simulation.\n"
                        "If this option is used, the -v flag must not be used.\n"
                        "The graph-displaying function can currently display only grid-like graphs, not custom graphs.\n"),
                        default="")

    parser.add_argument('-pl', '--plot', help=("Plot fixation probabilities of the given parameter"
                        "from 0 to its stated value.\n Fixation probabilities are computed by"
                        "running the simulation niters times for each value of the parameter.\n"
                        "Valid inputs are a, b, c, s, or h, for the parameters a, b, c,"
                        "the small-world coefficient of the graph, and the small-world heterogeneity."),
                        default="")

    parser.add_argument('-st', '--step', help="Step size of parameter in plot function", type=float, default=0.1)

    parser.add_argument('-db', "--debug", help="Enter debug mode (prints more stuff to output)",
                        action='store_true', default=False)

    parser.add_argument('-t', "--test", help="Activate unit tests (may increase computation time)", action='store_true', default=False)
    


    args = parser.parse_args()
    main(args)
