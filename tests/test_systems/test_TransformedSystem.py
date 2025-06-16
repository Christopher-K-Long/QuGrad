import numpy as np

from qugrad import QuantumSystem, HilbertSpace
from qugrad.systems import TransformedSystem

from .pauli_matrices import X, Y, Z

def test_TransformedSystem_initialisation():
    H0 = Z
    Hs = [X, Y]
    hilbert_space = HilbertSpace([0, 1])
    original_system = QuantumSystem(H0, Hs, hilbert_space)
    H02 = X
    Hs2 = [Y, Z]
    hilbert_space2 = HilbertSpace([1, 2])
    quantum_system = TransformedSystem(original_system,
                                       H02,
                                       Hs2,
                                       hilbert_space2)
    assert quantum_system.original_system == original_system
    assert np.array_equal(quantum_system.H0, H02)
    assert np.array_equal(quantum_system.Hs, Hs2)
    assert quantum_system.hilbert_space == hilbert_space2
    assert quantum_system.base_system == original_system
    

def test_original_system_read_only():
    H0 = Z
    Hs = [X, Y]
    hilbert_space = HilbertSpace([0, 1])
    original_system = QuantumSystem(H0, Hs, hilbert_space)
    H02 = X
    Hs2 = [Y, Z]
    hilbert_space2 = HilbertSpace([1, 2])
    quantum_system = TransformedSystem(original_system,
                                       H02,
                                       Hs2,
                                       hilbert_space2)
    original_system2 = QuantumSystem(H0, Hs, hilbert_space)
    try:
        quantum_system.original_system = original_system2
    except AttributeError:
        pass
    else:
        raise AssertionError("TransformedSystem.original_system should be read-only")

def test_base_system_read_only():
    H0 = Z
    Hs = [X, Y]
    hilbert_space = HilbertSpace([0, 1])
    original_system = QuantumSystem(H0, Hs, hilbert_space)
    H02 = X
    Hs2 = [Y, Z]
    hilbert_space2 = HilbertSpace([1, 2])
    quantum_system = TransformedSystem(original_system,
                                       H02,
                                       Hs2,
                                       hilbert_space2)
    original_system2 = QuantumSystem(H0, Hs, hilbert_space)
    try:
        quantum_system.base_system = original_system2
    except AttributeError:
        pass
    else:
        raise AssertionError("TransformedSystem.base_system should be read-only")

def test_base_system():
    H0 = Z
    Hs = [X, Y]
    hilbert_space = HilbertSpace([0, 1])
    original_system = QuantumSystem(H0, Hs, hilbert_space)
    H02 = X
    Hs2 = [Y, Z]
    hilbert_space2 = HilbertSpace([1, 2])
    quantum_system = TransformedSystem(original_system,
                                       H02,
                                       Hs2,
                                       hilbert_space2)
    H03 = X
    Hs3 = [Y, Z]
    hilbert_space3 = HilbertSpace([1, 2])
    quantum_system2 = TransformedSystem(quantum_system,
                                        H03,
                                        Hs3,
                                        hilbert_space3)
    assert quantum_system2.base_system == original_system
    assert quantum_system2.original_system == quantum_system
    

def test_pre_processing():
    H0 = Z
    Hs = [X, Y]
    hilbert_space = HilbertSpace([0, 1])
    original_system = QuantumSystem(H0, Hs, hilbert_space)
    H02 = X
    Hs2 = [Y, Z]
    hilbert_space2 = HilbertSpace([1, 2])
    quantum_system = TransformedSystem(original_system,
                                       H02,
                                       Hs2,
                                       hilbert_space2)
    ctrl_amp_env = np.array([[1+2j, 4,     8    ],
                             [2,    7+ 5j, 9+10j],
                             [  3j,    6j,   11j],
                             [1,    1,     1    ]])
    dt = 0.1
    initial_state = np.array([1, 0])
    frequencies = np.array([0.5, 0, 2.5])
    number_channels = [1, 2]
    args = (ctrl_amp_env, initial_state, dt, frequencies, number_channels)
    output1 = quantum_system._pre_processing(*args)
    output2 = original_system._pre_processing(*args)
    assert len(output1) == len(output2)
    for out1, out2 in zip(output1, output2):
        assert np.array_equal(out1, out2)
    

def test_envolope_processing():
    H0 = Z
    Hs = [X, Y]
    hilbert_space = HilbertSpace([0, 1])
    original_system = QuantumSystem(H0, Hs, hilbert_space)
    H02 = X
    Hs2 = [Y, Z]
    hilbert_space2 = HilbertSpace([1, 2])
    quantum_system = TransformedSystem(original_system,
                                       H02,
                                       Hs2,
                                       hilbert_space2)
    ctrl_amp_env = np.array([[1+2j, 4,     8    ],
                             [2,    7+ 5j, 9+10j],
                             [  3j,    6j,   11j],
                             [1,    1,     1    ]])
    dt = 0.1
    frequencies = np.array([0.5, 0, 2.5])
    number_channels = [1, 2]
    args = (ctrl_amp_env, dt, frequencies, number_channels)
    out1 = quantum_system._envolope_processing(*args)
    out2 = original_system._envolope_processing(*args)
    assert np.array_equal(out1, out2)