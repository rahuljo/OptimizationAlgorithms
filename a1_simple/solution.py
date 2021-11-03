import numpy as np
import sys
sys.path.append("..")

from optimization_algorithms.interface.mathematical_program import  MathematicalProgram
from optimization_algorithms.interface.nlp_solver import  NLPSolver
from optimization_algorithms.utils.finite_diff import finite_diff_grad


class Problem1( MathematicalProgram ):
    """
    """

    def __init__(self,c,a,n):
        self.c = c
        self.a = a
        self.n = n
        self.C = np.diag([c**((i-1)/(n-1)) for i in range(1,n+1)])

    def evaluate_sq(self, x) :

        Cx = np.dot(self.C,x)
        y = np.dot(Cx,x)
        J = np.dot(x.T,self.C) + np.dot(x.T,self.C.T)

        # and return as a tuple of arrays, namely of dim (1) and (1,n)
        return  np.array( [ y ] ) ,  J.reshape(1,-1)

    def evaluate_hole(self, x) :

        Cx = np.dot(self.C,x)
        xTCx = np.dot(Cx,x)
        K = (self.a ** 2) + xTCx
        y = xTCx / K
        J = ((self.a / K)**2) * (np.dot(x.T,self.C) + np.dot(x.T,self.C.T))

        # and return as a tuple of arrays, namely of dim (1) and (1,n)
        return  np.array( [ y ] ) ,  J.reshape(1,-1)


class Solver1(NLPSolver):

    def __init__(self,alpha):
        self.alpha = alpha

    def solve(self,sq=True): # set sq=True when running for f_sq and sq=False when running for f_hole

        func = self.problem.evaluate_sq if sq else self.problem.evaluate_hole

        theta = 1e-6

        it = 0
        x = np.array([1,1])
        counter = 1
        while counter <= 10: # continue the loop until the change in x is less than theta for 10 continuous iterations

            phi, J = func(x)
            print(f'x{it}={x}; cost{it}={phi}')

            x_upd = (x - (self.alpha * J))[0]

            dx = np.linalg.norm(x_upd-x)
            counter = counter + 1 if dx < theta else 1

            x = x_upd
            it += 1
        print(f'minima={x}; minimum={phi}')
        return x 
