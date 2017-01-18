import argparse

def genCircle(n, file):
    for i in range(n):
        file.writelines(str(i) + ',' + str((i-1) % n) + '\n')
        file.writelines(str(i) + ',' + str((i+1) % n) + '\n')
        

def genLine(n, file):
    for i in range(n-1):
        file.writelines(str(i) + ',' + str(i+1) + '\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Graph Generator",
                                 epilog="")
    parser.add_argument('-n', '--number', help="Number of individuals", type=int, default=100)
    parser.add_argument('-f', '--file', help="Output file. Defaults to sample_graphs/TYPE_N.txt (where TYPE and N are the other arguments)",
                        default="")
    parser.add_argument('-t', '--type', help="Type of graph to create. Supported types are 'circle' and 'line'.",
                        default='circle')

    args = parser.parse_args()
    file = args.file
    if args.file == '':
        file = 'sample_graphs/' + args.type + '_' + str(args.number) + '.txt'
        file = 'sample_graphs/' + args.type + '_' + str(args.number) + '.txt'
    file = open(file, 'w')
    if args.type == 'circle':
        genCircle(args.number, file)
    elif args.type == 'line':
        genLine(args.number, file)
    else:
        file.close()
        raise ValueError("Unsupported graph type.")
    file.close()
