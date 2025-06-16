# examples/subclass.py

import numpy as np
import tensorflow as tf

from functools import reduce
from qugrad import QuantumSystem, HilbertSpace

def kron(*args) -> np.ndarray:
    """
    Kronecker product of multiple matrices.

    Parameters
    ----------
    args: list[NDArray]
        The matrices to take the Kronecker product of

    Returns
    -------
    NDArray
        The Kronecker product of the matrices
    """
    return reduce(np.kron, args)

class ExampleSubclass(QuantumSystem):
    """
    An example subclass of QuantumSystem.
    """  
    def __init__(self,
                 qubits: int,
                 use_graph: bool = True):
        """
        Initialises the ExampleSubclass.

        Parameters
        ----------
        qubits: int
            The number of qubits in the system
        use_graph: bool
            Whether to use TensorFlow graphs during computation.
        """
        self.qubits = qubits
        X = np.array([[0,  1],  # Pauli-X
                      [1,  0]])
        Z = np.array([[1,  0],  # Pauli-Z
                      [0, -1]])
        H0 = kron(*[Z]*qubits)
        Xn = lambda n: kron(np.identity(2**(n-1)),
                            X,
                            np.identity(2**(qubits-n)))
        Hs = [Xn(n) for n in range(1, qubits+1)]
        super().__init__(H0,
                         Hs,
                         HilbertSpace(np.arange(2**qubits)),
                         use_graph)

    def _pre_processing(self,
                        frequencies: np.ndarray,
                        amplitudes: np.ndarray,
                        T: float
                       ) -> tuple:
        """
        When calling any evolution method (listed in the See also section)
        `_pre_processing()` is executed on the arguements before the control
        amplitudes are modulated by the frequencies (during
        `_envolope_processing()`) and then finally the modulated control
        amplitudes are used by the evolution method.

        Parameters
        ----------
        frequencies: NDArray[Shape[`qubits`], float]
            The frequencies of the to drive X on each of the qubits
        amplitudes: NDArray[Shape[`qubits`], complex]
            The amplitude of to drive X on each of the qubits
        T: float
            The time to evolve the system for

        Returns
        -------
        tuple[tf.Tensor[Shape[n_time_steps, total_n_channels], tf.complex128], tf.Tensor[Shape[:attr:`state_shape`], tf.complex128], float, tf.Tensor[Shape[n_time_steps, total_n_channels], tf.complex128], list[int]]
            A tuple of
            1. The control amplitude envolopes
            2. The initial state
            3. The integrator time step
            4. The frequencies to modulate the control amplitude envolopes with
            5. A list of the number of channels for each control Hamiltonian

            Warning
            -------
            The number of channels for each control Hamiltonian must be stored
            as a ``list`` and not an ``NDArray`` or a
            `TensorFlow <https://www.tensorflow.org>`__ tensor.


        See Also
        --------
        * `propagate()`
        * `propagate_collection()`
        * `propagate_all()`
        * `evolved_expectation_value()`
        * `evolved_expectation_value_all()`
        * `get_driving_pulses()`
        * `gradient()`
        """
        amplitudes = tf.cast(amplitudes, tf.complex128)
        amplitudes = tf.reshape(amplitudes, (1, self.qubits))
        n_time_steps = tf.cast(tf.math.ceil(T)*100, tf.int32)
        ctrl_amp = tf.broadcast_to(amplitudes, tf.stack([n_time_steps, self.qubits], axis=0))
        initial_state = self.hilbert_space.basis_vector(0)
        dt = T/tf.cast(n_time_steps, tf.float64)
        frequencies = tf.reshape(tf.cast(frequencies, dtype=tf.complex128), (self.qubits,))
        number_channels = [1]*self.qubits
        return super()._pre_processing(ctrl_amp, initial_state, dt, frequencies, number_channels)