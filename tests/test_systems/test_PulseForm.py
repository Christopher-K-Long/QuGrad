import numpy as np

from qugrad import QuantumSystem, HilbertSpace
from qugrad.systems import PulseForm

from .pauli_matrices import X, Y, Z

def pulse_function():
    ctrl_amp_env = np.array([[1+2j, 4,     8    ],
                             [2,    7+ 5j, 9+10j],
                             [  3j,    6j,   11j],
                             [1,    1,     1    ]])
    dt = 0.1
    initial_state = np.array([1, 0])
    frequencies = np.array([0.5, 0, 2.5])
    number_channels = [1, 2]
    return ctrl_amp_env, initial_state, dt, frequencies, number_channels

def pulse_function_append(ctrl_amp_env,
                          initial_state,
                          dt,
                          frequencies,
                          number_channels):
    return 2*ctrl_amp_env, initial_state, dt, frequencies, number_channels


def test_PulseForm_initialisation():
    H0 = Z
    Hs = [X, Y]
    hilbert_space = HilbertSpace([0, 1])
    original_system = QuantumSystem(H0, Hs, hilbert_space)
    original_system.initialise_evolver()
    pulse_form = PulseForm(original_system, pulse_function)
    assert pulse_form.evolver == original_system.evolver
    assert np.array_equal(pulse_form.H0, original_system.H0)
    assert np.array_equal(pulse_form.Hs, original_system.Hs)
    assert pulse_form.hilbert_space == original_system.hilbert_space
    assert pulse_form.pulse_function == pulse_function
    assert pulse_form.appended == False

    pulse_form2 = PulseForm(original_system, pulse_function_append, append=True)
    assert pulse_form2.evolver == original_system.evolver
    assert np.array_equal(pulse_form2.H0, original_system.H0)
    assert np.array_equal(pulse_form2.Hs, original_system.Hs)
    assert pulse_form2.hilbert_space == original_system.hilbert_space
    assert pulse_form2.pulse_function == pulse_function_append
    assert pulse_form2.appended == True

def test_PulseForm_initialisation_from_QuantumSystem():
    H0 = Z
    Hs = [X, Y]
    hilbert_space = HilbertSpace([0, 1])
    original_system = QuantumSystem(H0, Hs, hilbert_space)
    original_system.initialise_evolver()
    pulse_form = original_system.pulse_form(pulse_function)
    assert pulse_form.evolver == original_system.evolver
    assert np.array_equal(pulse_form.H0, original_system.H0)
    assert np.array_equal(pulse_form.Hs, original_system.Hs)
    assert pulse_form.hilbert_space == original_system.hilbert_space
    assert pulse_form.pulse_function == pulse_function
    assert pulse_form.appended == False

    pulse_form2 = original_system.pulse_form(pulse_function_append, append=True)
    assert pulse_form2.evolver == original_system.evolver
    assert np.array_equal(pulse_form2.H0, original_system.H0)
    assert np.array_equal(pulse_form2.Hs, original_system.Hs)
    assert pulse_form2.hilbert_space == original_system.hilbert_space
    assert pulse_form2.pulse_function == pulse_function_append
    assert pulse_form2.appended == True

def test_pulse_form():
    H0 = Z
    Hs = [X, Y]
    hilbert_space = HilbertSpace([0, 1])
    original_system = QuantumSystem(H0, Hs, hilbert_space)
    original_system.initialise_evolver()
    pulse_form = PulseForm(original_system, pulse_function)
    ctrl_amp_env = np.array([[1+2j, 4,     8    ],
                             [2,    7+ 5j, 9+10j],
                             [  3j,    6j,   11j],
                             [1,    1,     1    ]])
    dt = 0.1
    initial_state = np.array([1, 0])
    frequencies = np.array([0.5, 0, 2.5])
    number_channels = [1, 2]
    args = (ctrl_amp_env, initial_state, dt, frequencies, number_channels)
    output1 = pulse_form._pre_processing()
    output2 = original_system._pre_processing(*args)
    assert len(output1) == len(output2)
    for out1, out2 in zip(output1, output2):
        assert np.array_equal(out1, out2)
    
def test_pulse_form_append():
    H0 = Z
    Hs = [X, Y]
    hilbert_space = HilbertSpace([0, 1])
    original_system = QuantumSystem(H0, Hs, hilbert_space)
    original_system.initialise_evolver()
    pulse_form = PulseForm(original_system, pulse_function_append, append=True)
    ctrl_amp_env = np.array([[1+2j, 4,     8    ],
                             [2,    7+ 5j, 9+10j],
                             [  3j,    6j,   11j],
                             [1,    1,     1    ]])
    dt = 0.1
    initial_state = np.array([1, 0])
    frequencies = np.array([0.5, 0, 2.5])
    number_channels = [1, 2]
    args1 = (ctrl_amp_env, initial_state, dt, frequencies, number_channels)
    args2 = (2*ctrl_amp_env, initial_state, dt, frequencies, number_channels)
    output1 = pulse_form._pre_processing(*args1)
    output2 = original_system._pre_processing(*args2)
    assert len(output1) == len(output2)
    for out1, out2 in zip(output1, output2):
        assert np.array_equal(out1, out2)

def test_pulse_function_read_only():
    H0 = Z
    Hs = [X, Y]
    hilbert_space = HilbertSpace([0, 1])
    original_system = QuantumSystem(H0, Hs, hilbert_space)
    original_system.initialise_evolver()
    pulse_form = PulseForm(original_system, pulse_function)
    try:
        pulse_form.pulse_function = lambda *x: x
    except AttributeError:
        pass
    else:
        raise AssertionError("PulseFrom.pulse_function should be read-only")

def test_appended_read_only():
    H0 = Z
    Hs = [X, Y]
    hilbert_space = HilbertSpace([0, 1])
    original_system = QuantumSystem(H0, Hs, hilbert_space)
    original_system.initialise_evolver()
    pulse_form = PulseForm(original_system, pulse_function)
    try:
        pulse_form.appended = True
    except AttributeError:
        pass
    else:
        raise AssertionError("PulseFrom.appended should be read-only")