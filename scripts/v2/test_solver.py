import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import unittest
import numpy as np
from scripts.v2.global_optimum import solve_simplex


class SolverTests(unittest.TestCase):
    def test_interior_and_vertex(self):
        np.testing.assert_allclose(solve_simplex(np.eye(3),np.array([.2,.3,.5])),[.2,.3,.5],atol=1e-10)
        np.testing.assert_allclose(solve_simplex(np.eye(3),np.array([2.,0.,0.])),[1.,0.,0.],atol=1e-10)

    def test_solution_dominates_dense_grid(self):
        rng=np.random.default_rng(7);a=rng.normal(size=(3,3));q=a.T@a;r=rng.normal(size=3)
        w=solve_simplex(q,r);score=w@q@w-2*r@w
        self.assertTrue((w>=0).all());self.assertAlmostEqual(w.sum(),1.)
        for i in range(31):
            for j in range(31-i):
                v=np.array([i,j,30-i-j])/30
                self.assertLessEqual(score,v@q@v-2*r@v+1e-10)


if __name__=='__main__':unittest.main()
