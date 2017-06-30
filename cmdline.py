import argparse


def getArgs():
    parser = argparse.ArgumentParser(description="Gene Drive Analysis Frontend",
                                 epilog="")

    parser.add_argument('-of', "--outfile", help="Output directory to store data in", default="")

    parser.add_argument('-o', "--output",
                        help=("Specify output format. Options:\n"
                                "0: do not output to file\n"
                                "1: output averages of final values across all "
                                "simulations per given small world coefficient\n"
                                "2: output final values for every simulation\n"
                                "Higher numbers also output everything for all lower"
                                " numbers, e.g. 2 will also output 1\n"),
                        type=int, default=0)

    parser.add_argument('-gf', "--graphFile",
                        help=("Name of file containing graph structure.\n"
                              "File should be structured as a sequence of lines A,B."
                              "Here A and B are vertices, and this line represents an edge from A to B.\n"
                              "If this option is disabled, the program will generate and use a lattice graph.\n"
                              "If enabled, this setting will override the row and column arguments.\n"
                              "This graph will still be small-worldified - if this is not desired, set small world coefficient to 0.\n"
                              "There is currently no way to represent weighted graphs with this simulation.\n"
                              "If this option is used, the -v flag must not be used.\n"
                              "The graph-displaying function can currently display only grid-like graphs, not custom graphs.\n"),
                        default="")

    parser.add_argument('-rp', '--randomPlacement', help="Place new offspring randomly on the graph, instead of in a neighboring location.",
                        action='store_true', default=False)

    parser.add_argument('-r', '--rows',
                        help="Number of rows of the lattice if a custom graph is not being used",
                        type=int, default=15)

    parser.add_argument('-c', "--cols",
                        help="Number of columns of the lattice if a custom graph is not being used",
                        type=int, default=15)

    parser.add_argument('-fr', '--fourRegular',
                        help="Use 4-regular lattice instead of 8-regular lattice.",
                        action='store_true', default=False)

    parser.add_argument('-pl', '--plot', help=("Plot fixation probabilities of the given parameter"
                        "from 0 to its stated value.\n Fixation probabilities are computed by"
                        "running the simulation niters times for each value of the parameter.\n"
                        "Valid inputs are a, fb, s, or h, for the parameters a, F_B,"
                        "the small-world coefficient of the graph, and the small-world heterogeneity."),
                        default="")

    parser.add_argument('-min', '--min', help="Minimum value of parameter in the plot. Strongly advised to be positive if 'fb' plot argument used.", type=float, default=0)

    parser.add_argument('-np', '--numpoints', help="Number of points to plot (only used when --plot argument is used)", type=int, default=51)

    parser.add_argument('-b', '--binsearch', help=("Binary search for the value of w such that FP = 1/n.\n"
                        "Searches w values from 0 to input w value (recommended value: 4).\n"
                        "Cannot be used in conjunction with '-p w' argument.\n"
                        "Can be used in conjunction with other plot parameters, in which case a separate binary search will be run"
                        "for each value of the parameter, and the plot will use the determined w values on the y-axis."),
                        default=False, action="store_true")

    parser.add_argument('--depth', help="Depth of binary search - accuracy will be w/2^depth", type=int, default=10)

    parser.add_argument('-a', "--hzfitness", help="Probability of homozygous offspring surviving the embryo", type=float, default=1)

    parser.add_argument('-fb', "--wtfitness", help="Fitness of wild-type (fitness of gene drive is 1)", type=float, default=2)

    parser.add_argument('-p', "--genedriveprob", help="Probability of heterozygote passing mutant allele to its offspring", type=float, default=1)

    parser.add_argument('-n', "--niters", help=("Number of times to run simulation "
                                                "per value of parameter."),
                        type=int, default=100)

    parser.add_argument('-s', "--swc",
                        help=("Small world coefficient.\n"
                              "If this argument is used, randomly chosen edges from the graph"
                              "will be replaced with new random edges in accordance with"
                              "the Watts-Strogatz model."),
                        type=float, default=0)
    parser.add_argument('-kp', '--keep', help="Add more edges when constructing small world network (default is to replace existing edges)",
                        action="store_true", default=False)
    parser.add_argument('-g', '--heterogeneity', help="Heterogeneity of SWN",
                        type=float, default=0)

    parser.add_argument('-f', '--frac', help="Fraction of mutant cells at beginning. Overriden by -m.",
                        type=float, default=0.35) #
    parser.add_argument('-m', '--initMutant', help="Number of mutant cells at beginning. Overrides -f if not -1 (defaults to 1).",
                        type=int, default=1)
    parser.add_argument('-e', "--extraspace",
                        help="Amount of extra space to add to adjacency grid (this should not be changed in most cases).",
                        type=int, default=5) #
    parser.add_argument('-v', "--visible", help="Number of steps to show grid",
                        type=int, default=-1) #
    parser.add_argument('-l', "--simlength", help="Length of each simulation (leave blank to continue until fixation).",
                        type=int, default=-1) #

    parser.add_argument('-d', "--delay", help="Time delay between steps",
                        type=float, default=0)

    parser.add_argument('-sa', "--sample", help="Deprecated.",
                        type=int, default=10)
                        # When using output modes 3 or 4, how often should the grid be sampled?"


    parser.add_argument('-db', "--debug", help="Enter debug mode (prints more stuff to output)",
                        action='store_true', default=False)

    parser.add_argument('-t', "--test", help="Activate unit tests (may increase computation time)", action='store_true', default=False)

    return parser.parse_args()