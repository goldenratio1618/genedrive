import random as rand
import argparse
import numpy as np
from matplotlib import pyplot as plt

class PairApprox:
    def __init__(self, k, n, Fb):
        self.pAA = 0
        self.pBA = 1 / (n - 1)
        self.nA = 1
        self.nB = n - 1
        self.n = n
        self.k = k
        self.Fb = Fb
        self.numSteps = 0
    
    """ These functions encode the fitnesses of each type. """
    def fa(self):
        return (1 - self.pAA) * self.k
    def fb_im(self):
        return self.pBA * self.k
    def fb_re(self):
        return (1 - self.pBA) * self.Fb * self.k
    def ft(self):
        return self.fa() * self.nA + (self.fb_im() + self.fb_re()) * (self.n - self.nA)
    
    """ Probability of mutant increasing or decreasing. """
    def pUp_1(self):
        # by multiplying by 1 - self.pAA we are already conditioning that there is a neighbor of type B we are going to replace
        # thus, we can modify the fitness slightly to take into account this information.
        return (self.fa() * (self.k-1)/self.k + self.Fb) * self.nA / self.ft() * (1 - self.pAA)
    def pUp_2(self):
        # similar trick here
        return (self.fb_im() * (self.k-1)/self.k) * self.nB / self.ft() * (1 - self.pBA)
    def pDown(self):
        # and here
        return (self.fb_re() * (self.k-1)/self.k) * self.nB / self.ft() * self.pBA

    def evolve(self):
        """ Evolves the system 1 step. Returns true if the mutant either fixated or went extinct and false otherwise. """
        val = rand.random()
        # so the simulation doesn't take forever, we estimate number of steps until something happens
        expNum = self.pUp_1() + self.pUp_2() + self.pDown()
        self.numSteps += expNum
        threshold = self.pUp_1() / expNum
        if val < threshold:
            if self.nB == 1:
                self.nA += 1
                self.nB -= 1
                return True
            self.pAA = (self.pAA * self.nA * self.k + (2 + 2 * self.pBA * (self.k - 1))) * (1 / (self.k * (self.nA + 1)))
            self.pBA = (self.pBA * self.nB * self.k - (1 + self.pBA * (self.k - 1)) + (1 - self.pBA) * (self.k - 1)) * (1 / (self.k * (self.nB - 1)))
            self.nA += 1
            self.nB -= 1
            return False
        threshold += self.pUp_2() / expNum
        if val < threshold:
            if self.nB == 1:
                self.nA += 1
                self.nB -= 1
                return True
            self.pAA = (self.pAA * self.nA * self.k + 2 * self.pBA * (self.k - 1)) * (1 / (self.k * (self.nA + 1)))
            self.pBA = (self.pBA * self.nB * self.k - self.pBA * (self.k - 1) + 1 + (1 - self.pBA) * (self.k - 1)) * (1 / (self.k * (self.nB - 1)))
            self.nA += 1
            self.nB -= 1
            return False
        threshold += self.pDown() / expNum
        if val < threshold:
            if self.nA == 1:
                self.nA -= 1
                self.nB += 1
                return True
            self.pAA = (self.pAA * self.nA * self.k - 2 * self.pAA * (self.k - 1)) * (1 / (self.k * (self.nA - 1)))
            self.pBA = (self.pBA * self.nB * self.k + self.pAA * (self.k - 1) - (1 - self.pAA) * (self.k - 1) - 1) * 1 / (self.k * (self.nB + 1))
            self.nA -= 1
            self.nB += 1
            return False
        # no evolution happened
        raise UserWarning("WARNING: No evolution occured.")
    
    def run(self):
        numSteps = 0
        maxA = 1
        while not self.evolve():
            numSteps += 1
            if self.nA > maxA:
                maxA = self.nA
        # print(maxA)
        return (self.nA == self.n, numSteps)

def main():
    parser = argparse.ArgumentParser(description="Pair Approximation for Gene Drive Model for Regular Graphs")
    parser.add_argument('-k', '--degree', help="Degree of the graph", type=int, default=2)
    parser.add_argument('-n', '--numvertices', help="Number of vertices in the graph", type=int, default=225)
    parser.add_argument('-fb', '--wtfitness', help="Fitness of the wild-type", type=float, default=1.0)
    parser.add_argument('-i', '--numiters', help="Number of times to run simulation per value of parameter", type=int, default=100)
    parser.add_argument('-p', '--plot', help="Plots fixation probability vs. Fb", action='store_true', default=False)
    parser.add_argument('-m', '--max', help="Maximum Fb value in plot", type=float, default=4)
    parser.add_argument('-np', '--numpoints', help="Number of points to plot (only used when --plot argument is used)", type=int, default=51)
    args = parser.parse_args()

    if args.plot:
        fb_vals = np.linspace(0, args.max, args.numpoints)
    else:
        fb_vals = [args.wtfitness]
    fp_vals = []
    for fb in fb_vals:
        numFix = 0
        for i in range(args.numiters):
            sim = PairApprox(args.degree, args.numvertices, fb)
            fix, _ = sim.run()
            if fix:
                numFix += 1
        fp_vals.append(numFix / args.numiters)
        print(str(fb) + ': ' + str(numFix / args.numiters))
    
    if args.plot:
        plt.plot(fb_vals, fixProbs)
        plt.xlabel('Wild-type fitness')
        plt.ylabel('Fixation probability')
        plt.show()

if __name__ == '__main__':
    main()