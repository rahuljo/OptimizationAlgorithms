import numpy as np
import sys
sys.path.append("..")

from optimization_algorithms.interface.mathematical_program import  MathematicalProgram
from optimization_algorithms.interface.nlp_solver import  NLPSolver
from optimization_algorithms.utils.finite_diff import finite_diff_grad

import time


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


class Solver1Backtracking(NLPSolver):

    def __init__(self,rho_ls=0.01,rho_neg=0.5,rho_pos=1.2):
        self.rho_ls = rho_ls
        self.rho_neg = rho_neg
        self.rho_pos = rho_pos

    def solve(self,sq=True): # set sq=True when running for f_sq and sq=False when running for f_hole

        rho_ls, rho_neg, rho_pos = self.rho_ls, self.rho_neg, self.rho_pos

        func = self.problem.evaluate_sq if sq else self.problem.evaluate_hole

        theta = 1e-10
        alpha = 1

        it = 0
        x = np.array([1,1])
        counter = 1
        while counter <= 10: # continue the loop until the step size is less than theta for 10 continuous iterations

            phi, J = func(x)
            print(f'x{it}={x}; cost{it}={phi}')

            delta = - J / np.linalg.norm(J)

            while func(x+(alpha * delta)[0])[0] > phi + (rho_ls * (J @ (alpha * delta).T)):
                alpha *= rho_neg

            x = x + (alpha * delta)[0]
            alpha *= rho_pos

            dx = np.linalg.norm(alpha * delta)
            counter = counter + 1 if dx < theta else 1

            it += 1

        print(f'minima={x}; minimum={phi}')
        return x 
