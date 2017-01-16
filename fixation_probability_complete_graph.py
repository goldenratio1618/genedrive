import argparse
import numpy as np
from matplotlib import pyplot as plt

def P(i, j, a, b, c, n):
    """ Computes the transition probabilities """
    if i == 0:
        return j == 0
    if i == n:
        return j == n
    if j == i + 1:
        return i * (a * (i - 1) + b * (n - i)) / TF(i, a, b, c, n) * (n - i) / (n - 1) + (n - i) * b * i / TF(i, a, b, c, n) * (n - i - 1) / (n - 1)
    if j == i - 1:
        return (n - i) * c * (n - i - 1) / TF(i, a, b, c, n) * i / (n - 1)
    if j == i:
        return 1 - P(i, i + 1, a, b, c, n) - P(i, i - 1, a, b, c, n)
    return 0
        

def TF(i, a, b, c, n):
    """ Computes the total fitness of the population"""
    return i * (a * (i - 1) + b * (n - i)) + (n - i) * (c * (n - i - 1) + b * i)


def fixProb(i, a, b, c, n):
    """ Computes the fixation probability of the mutant strain given i initial mutants """
    alpha_1 = P(1, 2, a, b, c, n) / (1 - P(1, 1, a, b, c, n))
    # this will store the value of alpha_j
    alpha = alpha_1
    # this will store the value of the product (which will equal the fixation probability)
    if i <= 1:
        prod = alpha_1
    else:
        prod = 1
    print("j = 1, alpha = " + str(alpha) + ", prod = " + str(prod))
    
    for j in range(2,n):
        # compute alpha_j
        alpha = P(j, j + 1, a, b, c, n) / (1 - alpha * P(j, j - 1, a, b, c, n) - P(j, j, a, b, c, n))
        if j >= i:
            prod *= alpha
        if j % 100 == 0 or j < 100:
            print("j = " + str(j) + ", alpha = " + str(alpha) + ", prod = " + str(prod))
    return prod

def main(args):
    print("Fixation probability = " + str(fixProb(args.i, args.a, args.b, args.c, args.n)))
    if args.plot == '':
        return
    if args.plot == 'i':
        i_range = np.arange(args.i, args.n)
        probs = [fixProb(i, args.a, args.b, args.c, args.n) for i in i_range]
        plt.plot(i_range, probs)
        plt.xlabel('Number of mutants')
        plt.ylabel('Fixation probability')
        plt.show()
    
    if args.plot == 'n':
        n_range = np.arange(args.i, args.n + 1)
        probs = [fixProb(args.i, args.a, args.b, args.c, n) for n in n_range]
        plt.plot(n_range, probs)
        plt.xlabel('Number of individuals')
        plt.ylabel('Fixation probability')
        plt.show()

    if args.plot == 'a':
        a_range = np.arange(0, args.a, args.step)
        probs = [fixProb(args.i, a, args.b, args.c, args.n) for a in a_range]
        plt.plot(a_range, probs)
        plt.xlabel('AA payoff')
        plt.ylabel('Fixation probability')
        plt.show()

    if args.plot == 'b':
        b_range = np.arange(0, args.b, args.step)
        probs = [fixProb(args.i, args.a, b, args.c, args.n) for b in b_range]
        plt.plot(b_range, probs)
        plt.xlabel('AB payoff')
        plt.ylabel('Fixation probability')
        plt.show()

    if args.plot == 'c':
        c_range = np.arange(0, args.c, args.step)
        probs = [fixProb(args.i, args.a, args.b, c, args.n) for c in c_range]
        plt.plot(c_range, probs)
        plt.xlabel('BB payoff')
        plt.ylabel('Fixation probability')
        plt.show()




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Fixation Probability Calculator", epilog="")

    parser.add_argument('-a', '--a', help="Coefficient 'a' in the payoff matrix",
                        type=float, default=0)

    parser.add_argument('-b', '--b', help="Coefficient 'b' in the payoff matrix",
                        type=float, default=1)

    parser.add_argument('-c', '--c', help="Coefficient 'c' in the payoff matrix",
                        type=float, default=2)

    parser.add_argument('-n', '--n', help="Number of cells",
                        type=int, default=10)

    parser.add_argument('-i', '--i', help="Initial number of mutant cells",
                        type=int, default=1)

    parser.add_argument('-p', '--plot', help=(
                        "Generate a plot of fixation probability against a given parameter, "
                        "holding all other parameters constant.\n"
                        "The parameter specified must be one of a, b, c, n, or i.\n"
                        "If i or n is chosen, the plot range will be from the input value of i to the input value of n.\n"
                        "If a, b, or c is chosen, the plot range will be from 0 to the input value of that parameter."),
                        default="")
    
    parser.add_argument('-s', '--step', help="If plotting a, b, or c, this will represent the step size in the plot.",
                        type=int, default=0.1)

    args = parser.parse_args()
    main(args)