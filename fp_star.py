def fix_prob(i, n, Fb):
    fps = {}
    # we know the fixation probability is 1 in this case
    fps[n] = 1
    # this is an APPROXIMATION but should be valid in the limit n -> infinity, we'll see...
    fps[n-1] = 1



def fix_prob_local(j, n, Fb, FPjp1, FPjp2):
    """ Returns the fixation probability FP(j) given FP(j+1) and FP(j+2) """
    i = j + 1
    return FPjp1 * (2 * i + Fb * (n - i)) * n / (i * (n - i) * Fb) * FPjp1 - FPjp2 / Fb  - n / (Fb * (n - i))

def main():
    n = 5
    print(fix_prob_local(n - 2, n, 2, 1, 1))

if __name__ == '__main__':
    main()