# -*- encoding: utf-8 -*-
"""
    15. MapReduce
    参考：

"""

import sys
from numpy import mat, mean, power

def read_input(file):
    for line in file:
        yield line.rstrip()



def run_main():
    input = read_input(sys.stdin)
    mapperOut = [line.split('\t') for line in input]
    cumVal = 0.0
    cumSumSq = 0.0
    cumN = 0.0
    for instance in mapperOut:
        if len(instance) != 3 : continue
        nj = float(instance[0])
        cumN += nj
        cumVal += nj * float(instance[1])
        cumSumSq += nj * float(instance[2])
    mean = cumVal / cumN
    varSum = (cumSumSq - 2*mean*cumVal + cumN*mean*mean)/cumN

    print('%d\t%f\t%f' % (cumN, mean, varSum))
    # print(sys.stderr, 'report: still alive')

    print()

if __name__ == '__main__':
    run_main()