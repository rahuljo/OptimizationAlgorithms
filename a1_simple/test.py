import numpy as np
import sys
sys.path.append("..")


# import the test classes

import unittest


from optimization_algorithms.interface.nlp_solver import  NLPSolver
from optimization_algorithms.interface.mathematical_program_traced import  MathematicalProgramTraced

from optimization_algorithms.mathematical_programs.quadratic_identity_2 import QuadraticIdentity2


from solution import Problem1,Solver1

class testSolver1(unittest.TestCase):
    """
    test on problem A
    """
    Problem = Problem1
    Solver = Solver1

    def testConstructor(self):
        """
        check the constructor
        """
        problem = self.Problem(c=10,a=0.1,n=2)
        solver_sq = self.Solver(alpha=0.01)
        solver_hole = self.Solver(alpha=0.001)

    def testConvergenceSQ(self):

        problem = self.Problem(c=10,a=0.1,n=2)
        solver_sq = self.Solver(alpha=0.01)
        solver_sq.setProblem((problem))

        output =  solver_sq.solve() # Runs for 500 iterations

        self.assertTrue( np.linalg.norm( np.zeros(2) - output  ) < 0.1)

    def testConvergenceHole(self):

        problem = self.Problem(c=10,a=0.1,n=2)
        solver_hole = self.Solver(alpha=0.001)
        solver_hole.setProblem((problem))
        
        output =  solver_hole.solve(sq=False) # Runs for 184k iterations
    
        self.assertTrue( np.linalg.norm( np.zeros(2) - output  ) < 0.1)


if __name__ == "__main__":
   unittest.main()


