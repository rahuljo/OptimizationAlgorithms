import numpy as np
import sys
sys.path.append("..")

from optimization_algorithms.interface.nlp_solver import  NLPSolver

class Solver0(NLPSolver):

    def __init__(self):
        """
        See also:
        ----
        NLPSolver.__init__
        """
        
        # in case you want to initialize some class members or so...


    def solve(self) :
        """

        See Also:
        ----
        NLPSolver.solve

        """
        
        # write your code here

        # use the following to get an initialization:
        x = self.problem.getInitializationSample()
        theta = 0.1
        phi, J = self.problem.evaluate(x)
        x_upd = (x - 0.1*J)[0]
        count = 1
        while np.linalg.norm(x_upd - x) >= theta or count<=10:
            if np.linalg.norm(x_upd - x) < theta:
                count += 1
            else:
                count = 1
            x = x_upd
            phi, J = self.problem.evaluate(x)
            x_upd = (x - 0.1*J)[0]
            
        # use the following to query the problem:
        
        # phi is a vector (1D np.array); use phi[0] to access the cost value (a float number). J is a Jacobian matrix (2D np.array). Use J[0] to access the gradient (1D np.array) of the cost value.

        # now code some loop that iteratively queries the problem and updates x til convergenc....

        # finally:
        return x 
