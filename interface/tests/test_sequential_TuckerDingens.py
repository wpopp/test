import sys
import TuckerDingens
import unittest

#don't confuse python unittest
sys.argv=sys.argv[:1]

class TestTuckerDingens(unittest.TestCase):
    def test_TuckerDingens_sequential1(self):
        A = [[i if i else 0] for i in range(0, 4)]
		v = [1,2,3,4]
		#Av squares diag(v)
        Av = TuckerDingens.MatMult(A, v)

if __name__ == '__main__':
    unittest.main()
