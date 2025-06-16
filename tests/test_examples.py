import os
import sys
FILE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(os.path.dirname(FILE_DIR), "examples"))


def test_Rabi_oscillation_optimisation():
    import Rabi_oscillation_optimisation

def test_subclass():
    import subclass
    subclass.ExampleSubclass(2).propagate([1, 2], [3, 4], 5.)