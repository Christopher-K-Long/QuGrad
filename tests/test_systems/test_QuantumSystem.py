import numpy as np
import tensorflow as tf
import py_ste
from py_ste import evolvers

from qugrad import QuantumSystem, HilbertSpace
from qugrad.systems._systems import generate_channel_couplings, ExpValCustom
from .pauli_matrices import X, Y, Z

def test_generate_channel_couplings():
    number_channels = [1, 4, 2, 3]
    channel_couplings = generate_channel_couplings(number_channels)
    assert np.array_equal(channel_couplings.shape,
                          (len(number_channels), sum(number_channels)))
    assert channel_couplings.dtype == bool
    assert np.array_equal(channel_couplings,
        [[True,  False, False, False, False, False, False, False, False, False],
         [False, True,  True,  True,  True,  False, False, False, False, False],
         [False, False, False, False, False, True,  True,  False, False, False],
         [False, False, False, False, False, False, False, True,  True,  True]])
    

def test_QuantumSystem_initialisation():
    H0 = Z
    Hs = [X, Y]
    hilbert_space = HilbertSpace([0, 1])
    quantum_system = QuantumSystem(H0, Hs, hilbert_space)
    assert np.array_equal(quantum_system.H0, H0)
    assert np.array_equal(quantum_system.Hs, Hs)
    assert quantum_system.hilbert_space == hilbert_space
    assert quantum_system.using_graph == True
    assert quantum_system.dim == 2
    assert np.array_equal(quantum_system.state_shape, (2,))
    assert quantum_system.n_ctrl == 2
    assert quantum_system._evolver is None

    quantum_system = QuantumSystem(H0, Hs, hilbert_space, use_graph=False)
    assert np.array_equal(quantum_system.H0, H0)
    assert np.array_equal(quantum_system.Hs, Hs)
    assert quantum_system.hilbert_space == hilbert_space
    assert quantum_system.using_graph == False
    assert quantum_system.dim == 2
    assert np.array_equal(quantum_system.state_shape, (2,))
    assert quantum_system.n_ctrl == 2
    assert quantum_system._evolver is None

        

def test_hilbert_space_read_only():
    H0 = Z
    Hs = [X, Y]
    hilbert_space = HilbertSpace([0, 1])
    quantum_system = QuantumSystem(H0, Hs, hilbert_space)
    try:
        quantum_system.hilbert_space = HilbertSpace([0, 1, 2])
    except AttributeError:
        pass
    else:
        raise AssertionError("QuantumSystem.hilbert_space should be read-only")

def test_H0_read_only():
    H0 = Z
    Hs = [X, Y]
    hilbert_space = HilbertSpace([0, 1])
    quantum_system = QuantumSystem(H0, Hs, hilbert_space)
    try:
        quantum_system.H0 = X
    except AttributeError:
        pass
    else:
        raise AssertionError("QuantumSystem.H0 should be read-only")

    try:
        quantum_system.H0[0, 0] = 0
    except ValueError:
        pass
    else:
        raise AssertionError("QuantumSystem.H0 should be read-only")

def test_Hs_read_only():
    H0 = Z
    Hs = [X, Y]
    hilbert_space = HilbertSpace([0, 1])
    quantum_system = QuantumSystem(H0, Hs, hilbert_space)
    try:
        quantum_system.Hs = np.array([X, Z])
    except AttributeError:
        pass
    else:
        raise AssertionError("QuantumSystem.Hs should be read-only")

    try:
        quantum_system.Hs[0, 0, 0] = 0
    except ValueError:
        pass
    else:
        raise AssertionError("QuantumSystem.Hs should be read-only")

def test_dim_read_only():
    H0 = Z
    Hs = [X, Y]
    hilbert_space = HilbertSpace([0, 1])
    quantum_system = QuantumSystem(H0, Hs, hilbert_space)
    try:
        quantum_system.dim = 3
    except AttributeError:
        pass
    else:
        raise AssertionError("QuantumSystem.dim should be read-only")

def test_state_shape_read_only():
    H0 = Z
    Hs = [X, Y]
    hilbert_space = HilbertSpace([0, 1])
    quantum_system = QuantumSystem(H0, Hs, hilbert_space)
    try:
        quantum_system.state_shape = (3,)
    except AttributeError:
        pass
    else:
        raise AssertionError("QuantumSystem.state_shape should be read-only")
    try:
        quantum_system.state_shape[0] = 0
    except TypeError:
        pass
    else:
        raise AssertionError("QuantumSystem.state_shape should be read-only")

def test_n_ctrl_read_only():
    H0 = Z
    Hs = [X, Y]
    hilbert_space = HilbertSpace([0, 1])
    quantum_system = QuantumSystem(H0, Hs, hilbert_space)
    try:
        quantum_system.n_ctrl = 3
    except AttributeError:
        pass
    else:
        raise AssertionError("QuantumSystem.n_ctrl should be read-only")

def test_evolver_initialisation():
    H0 = Z
    Hs = [X, Y]
    hilbert_space = HilbertSpace([0, 1])
    quantum_system = QuantumSystem(H0, Hs, hilbert_space)
    quantum_system.evolver
    assert isinstance(quantum_system.evolver, evolvers.DenseUnitaryEvolver)

    quantum_system2 = QuantumSystem(H0, Hs, hilbert_space)
    quantum_system2.initialise_evolver()
    assert isinstance(quantum_system.evolver, evolvers.DenseUnitaryEvolver)

    quantum_system.initialise_evolver(sparse=True)
    assert isinstance(quantum_system.evolver, evolvers.SparseUnitaryEvolver)

    quantum_system.initialise_evolver(force_dynamic=True)
    assert quantum_system.evolver.__class__ ==  evolvers.DenseUnitaryEvolver

    quantum_system.initialise_evolver(sparse=True, force_dynamic=True)
    assert quantum_system.evolver.__class__ ==  evolvers.SparseUnitaryEvolver    

def test_evolver_read_only():
    H0 = Z
    Hs = [X, Y]
    hilbert_space = HilbertSpace([0, 1])
    quantum_system = QuantumSystem(H0, Hs, hilbert_space)
    try:
        quantum_system.evolver = py_ste.get_unitary_evolver(H0, Hs, sparse=True)
    except AttributeError:
        pass
    else:
        raise AssertionError("QuantumSystem.evolver should be read-only")

def test_H_static():
    H0 = Z
    Hs = [X, Y]
    hilbert_space = HilbertSpace([0, 1])
    quantum_system = QuantumSystem(H0, Hs, hilbert_space)
    np.array_equal(quantum_system.H([1, 2]), Z+X+2*Y)
    np.array_equal(quantum_system.H([[1, 2], [3, 4]]), [Z+X+2*Y, Z+3*X+4*Y])

def test_H_dynamic():
    H0 = Z
    Hs = [X, Y]
    hilbert_space = HilbertSpace([0, 1])
    quantum_system = QuantumSystem(H0, Hs, hilbert_space)
    pulse = lambda t: np.array([np.sin(t), np.cos(t)])
    H = quantum_system.H(pulse)
    assert np.array_equal(H(np.e), Z+pulse(np.e)[0]*X+pulse(np.e)[1]*Y)

    pulse = np.array([lambda t: np.sin(t), lambda t: np.cos(t)])
    H = quantum_system.H(pulse)
    assert np.array_equal(H(np.e), Z+pulse[0](np.e)*X+pulse[1](np.e)*Y)

def test_pre_processing():
    H0 = Z
    Hs = [X, Y]
    hilbert_space = HilbertSpace([0, 1])
    quantum_system = QuantumSystem(H0, Hs, hilbert_space)
    assert np.array_equal(quantum_system._pre_processing(12, "5"), (12, "5"))

def test_evnvolope_processing():
    H0 = Z
    Hs = [X, Y]
    hilbert_space = HilbertSpace([0, 1])
    quantum_system = QuantumSystem(H0, Hs, hilbert_space)
    ctrl_amp_env = np.array([[1+2j, 4,     8    ],
                             [2,    7+ 5j, 9+10j],
                             [  3j,    6j,   11j],
                             [1,    1,     1    ]])
    dt = 0.1
    frequencies = np.array([0.5, 0, 2.5])
    number_channels = [1, 2]
    ctrl_amp = quantum_system._envolope_processing(ctrl_amp_env,
                                                   dt,
                                                   frequencies,
                                                   number_channels)
    ctrl_amp2 = \
        np.array([[1,                   4+8],
                  [2*np.cos(0.5*dt),    7+9*np.cos(2.5*dt)+10*np.sin(2.5*dt)],
                  [3*np.sin(0.5*2*dt), 11*np.sin(2.5*2*dt)],
                  [np.cos(0.5*3*dt),    1+np.cos(2.5*3*dt)]])
    assert np.allclose(ctrl_amp, ctrl_amp2)

def test_processing():
    H0 = Z
    Hs = [X, Y]
    hilbert_space = HilbertSpace([0, 1])
    quantum_system = QuantumSystem(H0, Hs, hilbert_space)
    ctrl_amp_env = np.array([[1+2j, 4,     8    ],
                             [2,    7+ 5j, 9+10j],
                             [  3j,    6j,   11j],
                             [1,    1,     1    ]])
    dt = 0.1
    initial_state = np.array([1, 0])
    frequencies = np.array([0.5, 0, 2.5])
    number_channels = [1, 2]
    out = quantum_system._processing(ctrl_amp_env,
                                     initial_state,
                                     dt,
                                     frequencies,
                                     number_channels)
    assert np.array_equal(out[0],
                          quantum_system._envolope_processing(ctrl_amp_env,
                                                              dt,
                                                              frequencies,
                                                              number_channels))
    assert np.array_equal(out[1], initial_state)
    assert out[2] == dt

def test_using_graph():
    H0 = Z
    Hs = [X, Y]
    hilbert_space = HilbertSpace([0, 1])
    quantum_system = QuantumSystem(H0, Hs, hilbert_space)
    ctrl_amp_env = np.array([[1+2j, 4,     8    ],
                             [2,    7+ 5j, 9+10j],
                             [  3j,    6j,   11j],
                             [1,    1,     1    ]])
    dt = 0.1
    initial_state = np.array([1, 0])
    frequencies = np.array([0.5, 0, 2.5])
    number_channels = [1, 2]
    args = (ctrl_amp_env, initial_state, dt, frequencies, number_channels)
    out1 = quantum_system._processing(*args)
    assert quantum_system._processing == quantum_system._graph_processing
    quantum_system.using_graph = False
    assert quantum_system._processing == quantum_system._eager_processing
    out2 = quantum_system._processing(*args)
    assert np.array_equal(out1[0], out2[0])
    assert np.array_equal(out1[1], out2[1])
    assert out1[2] == out2[2]
    quantum_system2 = QuantumSystem(H0, Hs, hilbert_space, use_graph=False)
    assert quantum_system2._processing == quantum_system2._eager_processing
    out3 = quantum_system2._processing(*args)
    assert np.array_equal(out1[0], out3[0])
    assert np.array_equal(out1[1], out3[1])
    assert out1[2] == out3[2]
    

def test_get_driving_pulses():
    H0 = Z
    Hs = [X, Y]
    hilbert_space = HilbertSpace([0, 1])
    quantum_system = QuantumSystem(H0, Hs, hilbert_space)
    ctrl_amp_env = np.array([[1+2j, 4,     8    ],
                             [2,    7+ 5j, 9+10j],
                             [  3j,    6j,   11j],
                             [1,    1,     1    ]])
    dt = 0.1
    initial_state = np.array([1, 0])
    frequencies = np.array([0.5, 0, 2.5])
    number_channels = [1, 2]
    args = (ctrl_amp_env, initial_state, dt, frequencies, number_channels)
    out1 = quantum_system.get_driving_pulses(*args)
    out2 = quantum_system._processing(*args)
    assert np.array_equal(out1[0], out2[0])
    assert np.array_equal(out1[1], out2[1])
    assert out1[2] == out2[2]

def test_propagate():
    H0 = Z
    Hs = [X, Y]
    hilbert_space = HilbertSpace([0, 1])
    quantum_system = QuantumSystem(H0, Hs, hilbert_space)
    ctrl_amp_env = np.array([[1+2j, 4,     8    ],
                             [2,    7+ 5j, 9+10j],
                             [  3j,    6j,   11j],
                             [1,    1,     1    ]])
    dt = 0.1
    initial_state = np.array([1, 0], dtype=complex)
    frequencies = np.array([0.5, 0, 2.5])
    number_channels = [1, 2]
    args = (ctrl_amp_env, initial_state, dt, frequencies, number_channels)
    state = quantum_system.propagate(*args)
    py_ste_args = quantum_system.get_driving_pulses(*args)
    evolver = py_ste.get_unitary_evolver(H0, Hs)
    assert np.array_equal(state, evolver.propagate(*py_ste_args))

def test_propagate_collection():
    H0 = Z
    Hs = [X, Y]
    hilbert_space = HilbertSpace([0, 1])
    quantum_system = QuantumSystem(H0, Hs, hilbert_space)
    ctrl_amp_env = np.array([[1+2j, 4,     8    ],
                             [2,    7+ 5j, 9+10j],
                             [  3j,    6j,   11j],
                             [1,    1,     1    ]])
    dt = 0.1
    initial_state = np.array([[1, 0],
                              [0, 1]],
                             dtype=complex)
    frequencies = np.array([0.5, 0, 2.5])
    number_channels = [1, 2]
    args = (ctrl_amp_env, initial_state, dt, frequencies, number_channels)
    state = quantum_system.propagate_collection(*args)
    py_ste_args = quantum_system.get_driving_pulses(*args)
    evolver = py_ste.get_unitary_evolver(H0, Hs)
    assert np.array_equal(state, evolver.propagate_collection(*py_ste_args))

def test_propagate_all():
    H0 = Z
    Hs = [X, Y]
    hilbert_space = HilbertSpace([0, 1])
    quantum_system = QuantumSystem(H0, Hs, hilbert_space)
    ctrl_amp_env = np.array([[1+2j, 4,     8    ],
                             [2,    7+ 5j, 9+10j],
                             [  3j,    6j,   11j],
                             [1,    1,     1    ]])
    dt = 0.1
    initial_state = np.array([1, 0], dtype=complex)
    frequencies = np.array([0.5, 0, 2.5])
    number_channels = [1, 2]
    args = (ctrl_amp_env, initial_state, dt, frequencies, number_channels)
    state = quantum_system.propagate_all(*args)
    py_ste_args = quantum_system.get_driving_pulses(*args)
    evolver = py_ste.get_unitary_evolver(H0, Hs)
    assert np.array_equal(state, evolver.propagate_all(*py_ste_args))

def test_evolved_expectation_value():
    H0 = Z
    Hs = [X, Y]
    O = X+Y+Z
    hilbert_space = HilbertSpace([0, 1])
    quantum_system = QuantumSystem(H0, Hs, hilbert_space)
    ctrl_amp_env = np.array([[1+2j, 4,     8    ],
                             [2,    7+ 5j, 9+10j],
                             [  3j,    6j,   11j],
                             [1,    1,     1    ]])
    dt = 0.1
    initial_state = np.array([1, 0], dtype=complex)
    frequencies = np.array([0.5, 0, 2.5])
    number_channels = [1, 2]
    args = (ctrl_amp_env, initial_state, dt, frequencies, number_channels)
    value = quantum_system.evolved_expectation_value(*args, O)
    py_ste_args = quantum_system.get_driving_pulses(*args)
    evolver = py_ste.get_unitary_evolver(H0, Hs)
    assert np.array_equal(value,
                          evolver.evolved_expectation_value(*py_ste_args, O))

def test_evolved_expectation_value_all():
    H0 = Z
    Hs = [X, Y]
    O = X+Y+Z
    hilbert_space = HilbertSpace([0, 1])
    quantum_system = QuantumSystem(H0, Hs, hilbert_space)
    ctrl_amp_env = np.array([[1+2j, 4,     8    ],
                             [2,    7+ 5j, 9+10j],
                             [  3j,    6j,   11j],
                             [1,    1,     1    ]])
    dt = 0.1
    initial_state = np.array([1, 0], dtype=complex)
    frequencies = np.array([0.5, 0, 2.5])
    number_channels = [1, 2]
    args = (ctrl_amp_env, initial_state, dt, frequencies, number_channels)
    value = quantum_system.evolved_expectation_value_all(*args, O)
    py_ste_args = quantum_system.get_driving_pulses(*args)
    evolver = py_ste.get_unitary_evolver(H0, Hs)
    assert np.array_equal(value,
                          evolver.evolved_expectation_value_all(*py_ste_args, O))

def test_ExpValCustom_initialisation():
    H0 = Z
    Hs = [X, Y]
    O = X+Y+Z
    hilbert_space = HilbertSpace([0, 1])
    quantum_system = QuantumSystem(H0, Hs, hilbert_space)
    dt = 0.1
    initial_state = np.array([1, 0], dtype=complex)
    expval = ExpValCustom(quantum_system, initial_state, dt, O)
    
    assert expval.system == quantum_system
    assert np.array_equal(expval.initial_state, initial_state)
    assert expval.dt == dt
    assert np.array_equal(expval.observable, O)

def test_ExpValCustom_run():
    H0 = Z
    Hs = [X, Y]
    O = X+Y+Z
    hilbert_space = HilbertSpace([0, 1])
    quantum_system = QuantumSystem(H0, Hs, hilbert_space)
    dt = 0.1
    initial_state = np.array([1, 0], dtype=complex)
    expval = ExpValCustom(quantum_system, initial_state, dt, O)
    ctrl_amp = np.array([[1+2j, 4,     8    ],
                         [2,    7+ 5j, 9+10j],
                         [  3j,    6j,   11j],
                         [1,    1,     1    ]])
    value = expval.run(tf.constant(ctrl_amp))
    value2 = quantum_system.evolver.evolved_expectation_value(ctrl_amp,
                                                              initial_state,
                                                              dt,
                                                              O)
    assert float(value) == float(value2.real)

def test_ExpValCustom_run_gradient():
    H0 = Z
    Hs = [X, Y]
    O = X+Y+Z
    hilbert_space = HilbertSpace([0, 1])
    quantum_system = QuantumSystem(H0, Hs, hilbert_space)
    dt = 0.1
    initial_state = np.array([1, 0], dtype=complex)
    expval = ExpValCustom(quantum_system, initial_state, dt, O)
    ctrl_amp = np.array([[1+2j, 4,     8    ],
                         [2,    7+ 5j, 9+10j],
                         [  3j,    6j,   11j],
                         [1,    1,     1    ]])
    x = tf.constant(ctrl_amp)
    with tf.GradientTape(persistent=False) as tape:
        tape.watch(x)
        value = expval.run(x)
    grad = tf.convert_to_tensor(tape.gradient(value, x)).numpy()
    grad2 = dt*quantum_system.evolver.switching_function(ctrl_amp,
                                                         initial_state,
                                                         dt,
                                                         O)[1]
    assert np.array_equal(grad, grad2)