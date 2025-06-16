# examples/Rabi_oscillation_optimisation.py

import numpy as np
import tensorflow as tf
from scipy.optimize import minimize

from qugrad import QuantumSystem, HilbertSpace

# Constants
X = np.array([[0,  1],  # Pauli-X
              [1,  0]],
             dtype=np.complex128)
Z = np.array([[1,  0],  # Pauli-Z
              [0, -1]],
             dtype=np.complex128)

INITIAL_STATE = np.array([1, 0], dtype=np.complex128)  # |0>
N_TIME_STEPS = 1000
T = 10
OBSERVABLE = Z

# Hamiltonian
H0 = Z
Hs = [X]

# Define quantum syste
hilbert_space = HilbertSpace([0, 1])
device = QuantumSystem(H0, Hs, hilbert_space)

# Define pulse form
def Rabi_drive(frequency, amplitude):
    amplitude = tf.cast(amplitude, tf.complex128)
    ctrl_amp = amplitude*tf.ones((N_TIME_STEPS, 1), dtype=tf.complex128)
    initial_state = INITIAL_STATE
    dt = T/N_TIME_STEPS
    frequencies = tf.reshape(tf.cast(frequency, dtype=tf.complex128), (1,))
    number_channels = [1]
    return ctrl_amp, initial_state, dt, frequencies, number_channels

# SciPy minimization expects an array of variables so we will wrap our pulse
#   form in this upack function
def unpack_variables(x):
    frequency = x[0]
    amplitude = x[1]
    return frequency, amplitude

# Adding the pulses to the device
driven_device = device.pulse_form(Rabi_drive).pulse_form(unpack_variables)

# Initial parameters
x0 = [1,   # frequency
      0.1] # amplitude

# Expectation value before optimisation
initial_expectation_value = driven_device.evolved_expectation_value(x0,
                                                                    OBSERVABLE)

# Optimisation
result = minimize(driven_device.gradient,
                  x0=x0,
                  args=(OBSERVABLE,),
                  jac=True)

# Print results
print("Initial expectation value: ", initial_expectation_value)
print("Optimisation result:")
print(result)