import argparse

def genCircle(n, file):
    """ Generates an n-cycle """
    for i in range(n):
        file.writelines(str(i) + ',' + str((i-1) % n) + '\n')
        file.writelines(str(i) + ',' + str((i+1) % n) + '\n')

def genComplete(n, file):
    """ Generates an n-complete graph """
    for i in range(n):
        for j in range(n):
            if i != j:
                file.writelines(str(i) + ',' + str(j) + '\n')

def genLine(n, file):
    """ Generates an n-line """
    for i in range(n-1):
        file.writelines(str(i) + ',' + str(i+1) + '\n')

def genDoubleComplete(n, file):
    """ Generates n complete graphs of size n, picks one node out of each,
        and connects those nodes together in another complete graph. """
    for i in range(n):
        for j in range(i*n, (i+1)*n):
            for k in range(i*n, (i+1)*n):
                if k != j:
                    file.writelines(str(j) + ',' + str(k) + '\n')
        for j in range(n):
            if j != i:
                file.writelines(str(i*n) + ',' + str(j*n) + '\n')

            

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Graph Generator",
                                 epilog="")
    parser.add_argument('-n', '--number', help="Number of individuals", type=int, default=100)
    parser.add_argument('-f', '--file', help="Output file. Defaults to sample_graphs/TYPE_N.txt (where TYPE and N are the other arguments)",
                        default="")
    parser.add_argument('-t', '--type', help="Type of graph to create. Supported types are 'circle', 'line', 'complete', and 'doublecomplete'.",
                        default='circle')

    args = parser.parse_args()
    file = args.file
    if args.file == '':
        file = 'sample_graphs/' + args.type + '_' + str(args.number) + '.txt'
    file = open(file, 'w')
    if args.type == 'circle':
        genCircle(args.number, file)
    elif args.type == 'line':
        genLine(args.number, file)
    elif args.type == 'doublecomplete':
        genDoubleComplete(args.number, file)
    elif args.type == 'complete':
        genComplete(args.number, file)
    else:
        file.close()
        raise ValueError("Unsupported graph type.")
    file.close()
