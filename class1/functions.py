#!/usr/bin/env python3

#########################
##    IMPORT MODULES   ##
#########################
import pprint


#########################
##      VARIABLES      ##
#########################


#########################
##      FUNCTIONS      ##
#########################
def f1(n):
    i = 1
    r = 0
    counter = 0
    while i <= n:
        counter += 1
        r += i
        i += 1

    return {'r': r, 'counter': counter}


def f2(n):
    i = 1
    r = 0
    counter = 0

    while i <= n:
        j = 1
        while j <= n:
            counter += 1
            j += 1
            r += 1
        i += 1

    return {'r': r, 'counter': counter}


def f3(n):
    i = 1
    r = 0
    counter = 0

    while i <= n:
        j = i
        while j <= n:
            counter += 1
            r += 1
            j += 1
        i += 1

    return {'r': r, 'counter': counter}


def f4(n):
    i = 1
    r = 0
    counter = 0

    while i <= n:
        j = 1
        while j <= i:
            counter += 1
            r += j
            j += 1
        i += 1

    return {'r': r, 'counter': counter}


# def r1(n, counter):
#     if n == 0:
#         return 0
#     else:
#         counter += 1
#         return 1 + r1(n-1, counter), counter


#########################
##     MAIN SCRIPT     ##
#########################
def main():
    n_max = 10
    global_dict = {'f1': {'n': {}},
                   'f2': {'n': {}},
                   'f3': {'n': {}},
                   'f4': {'n': {}},
                   'r1': {'n': {}}}

    # global_dict['f1']['n'][n_max] = f1(n_max)
    for n in range(1, n_max + 1):
        global_dict['f1']['n'][n] = f1(n)
        global_dict['f2']['n'][n] = f2(n)
        global_dict['f3']['n'][n] = f3(n)
        global_dict['f4']['n'][n] = f4(n)
        # global_dict['r1']['n'][n] = {'r': r1(n)[0], 'counter': }

    pprint.pprint(global_dict)


if __name__ == "__main__":
    main()
