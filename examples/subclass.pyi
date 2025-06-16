# examples/subclass.pyi

import numpy as np

from qugrad import QuantumSystem
from typing import Callable

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
    ...

class ExampleSubclass(QuantumSystem):
    """
    An example subclass of QuantumSystem.
    """
    
    _processing: Callable[..., tuple]
    """
    Executes `_pre_processing()` followed by
    `_envolope_processing()` eagerly (i.e. without using a TensorFlow
    graph). Nonetheless, `_eager_processing()` is still auto
    differentiable.

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
    tuple[tf.Tensor[Shape[n_time_steps, `n_ctrl`], complex], tf.Tensor[Shape[`state_shape`], complex], tf.Tensor[Shape[], float]]
        A tuple of:
        1. Control amplitudes
        2. Initial state
        3. Integrator time step
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
        ...
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
        tuple[tf.Tensor[Shape[n_time_steps, total_n_channels], tf.complex128], tf.Tensor[Shape[`state_shape`], tf.complex128], float, tf.Tensor[Shape[n_time_steps, total_n_channels], tf.complex128], list[int]]
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
        ...
    def propagate(self,
                  ctrl_amp : np.ndarray[complex],
                  initial_state : np.ndarray[complex],
                  dt: float,
                  frequencies: np.ndarray[complex],
                  number_channels: list[int]
                 ) -> np.ndarray[complex]:
        """
        Evolves a state vector under the time-dependent Hamiltonian defined by
        the control amplitudes using `propagate()` from
        `PySTE <https://PySTE.readthedocs.io>`__.

        Parameters
        ----------
        frequencies: NDArray[Shape[`qubits`], float]
            The frequencies of the to drive X on each of the qubits
        amplitudes: NDArray[Shape[`qubits`], complex]
            The amplitude of to drive X on each of the qubits
        T: float
            The time to evolve the system for

        Warning
        -------
        Keyword arguments are not supported.

        Returns
        -------
        NDArray[Shape[`state_shape`], complex]
            The final state

        See Also
        --------
        * `propagate_collection()`
        * `propagate_all()`
        """
        ...
    def propagate_collection(self,
                             ctrl_amp : np.ndarray[complex],
                             initial_states : np.ndarray[complex],
                             dt: float,
                             frequencies: np.ndarray[complex],
                             number_channels: list[int]
                            ) -> np.ndarray[complex]:
        """
        Evolves a collection of state vectors under the time-dependent
        Hamiltonian defined by the control amplitudes using
        `propagate_collection()` from `PySTE <https://PySTE.readthedocs.io>`__.

        Parameters
        ----------
        frequencies: NDArray[Shape[`qubits`], float]
            The frequencies of the to drive X on each of the qubits
        amplitudes: NDArray[Shape[`qubits`], complex]
            The amplitude of to drive X on each of the qubits
        T: float
            The time to evolve the system for

        Warning
        -------
        Keyword arguments are not supported.

        Returns
        -------
        NDArray[Shape[n_states, `state_shape`], complex]
            The final state

        See Also
        --------
        * `propagate()`
        * `propagate_all()`
        """
        ...
    def propagate_all(self,
                      ctrl_amp : np.ndarray[complex],
                      initial_states : np.ndarray[complex],
                      dt: float,
                      frequencies: np.ndarray[complex],
                      number_channels: list[int]
                     ) -> np.ndarray[complex]:
        """
        Evolves a state vector under the time-dependent Hamiltonian defined by
        the control amplitudes using `propagate_all()` from
        `PySTE <https://PySTE.readthedocs.io>`__ and returns the state at each
        time-step.

        Parameters
        ----------
        frequencies: NDArray[Shape[`qubits`], float]
            The frequencies of the to drive X on each of the qubits
        amplitudes: NDArray[Shape[`qubits`], complex]
            The amplitude of to drive X on each of the qubits
        T: float
            The time to evolve the system for

        Warning
        -------
        Keyword arguments are not supported.

        Returns
        -------
        NDArray[Shape[n_time_steps+1, `state_shape`], complex]
            The state at each integrator time step (including the initial
            state).

        See Also
        --------
        * `propagate()`
        * `propagate_collection()`
        """
        ...
    def evolved_expectation_value(self,
                                  ctrl_amp : np.ndarray[complex],
                                  initial_state : np.ndarray[complex],
                                  dt: float,
                                  frequencies: np.ndarray[complex],
                                  number_channels: list[int],
                                  observable : np.ndarray[complex]
                                 ) -> complex:
        """
        Evolves a state vector under the time-dependent Hamiltonian defined by
        the control amplitudes and computes the expectation value of a specified
        observable with respect to the final state using
        `evolved_expectation_value()` from
        `PySTE <https://PySTE.readthedocs.io>`__.

        Parameters
        ----------
        frequencies: NDArray[Shape[`qubits`], float]
            The frequencies of the to drive X on each of the qubits
        amplitudes: NDArray[Shape[`qubits`], complex]
            The amplitude of to drive X on each of the qubits
        T: float
            The time to evolve the system for
        observable : NDArray[Shape[`dim`, `dim`], complex]
            The observable to take the expectation value of.

        Warning
        -------
        Keyword arguments are not supported.

        Returns
        -------
        complex
            The expectation value.

        See Also
        --------
        * `evolved_expectation_value_all()`
        * `gradient()`
        """
        ...
    def evolved_expectation_value_all(self,
                                      ctrl_amp : np.ndarray[complex],
                                      initial_state : np.ndarray[complex],
                                      dt: float,
                                      frequencies: np.ndarray[complex],
                                      number_channels: list[int],
                                      observable : np.ndarray[complex]
                                     ) -> np.ndarray[complex]:
        """
        Evolves a state vector under the time-dependent Hamiltonian defined by
        the control amplitudes and computes the expectation value of a specified
        observable with respect to the state at each time-step using
        `evolved_expectation_value_all()` from
        `PySTE <https://PySTE.readthedocs.io>`__.

        Parameters
        ----------
        frequencies: NDArray[Shape[`qubits`], float]
            The frequencies of the to drive X on each of the qubits
        amplitudes: NDArray[Shape[`qubits`], complex]
            The amplitude of to drive X on each of the qubits
        T: float
            The time to evolve the system for
        observable : NDArray[Shape[`dim`, `dim`], complex]
            The observable to take the expectation value of.

        Warning
        -------
        Keyword arguments are not supported.

        Returns
        -------
        NDArray[Shape[n_time_steps+1], complex]
            The state at each integrator time step (including the initial
            state).

         See Also
        --------
        * `evolved_expectation_value()`
        * `gradient()`
        """
        ...
    def get_driving_pulses(self,
                           ctrl_amp : np.ndarray[complex],
                           initial_states : np.ndarray[complex],
                           dt: float,
                           frequencies: np.ndarray[complex],
                           number_channels: list[int]
                          ) -> tuple[np.ndarray[complex], np.ndarray[complex], float]:
        """
        When calling any evolution method (listed in the See also section`)
        `get_driving_pulses()` is executed on the arguements before the
        evolution method.

        Parameters
        ----------
        frequencies: NDArray[Shape[`qubits`], float]
            The frequencies of the to drive X on each of the qubits
        amplitudes: NDArray[Shape[`qubits`], complex]
            The amplitude of to drive X on each of the qubits
        T: float
            The time to evolve the system for

        Warning
        -------
        Keyword arguments are not supported.

        Returns
        -------
        tuple[NDArray[Shape[n_time_steps, `n_ctrl`], complex], NDArray[Shape[`state_shape`], complex], float]
            A tuple of:
            1. Control amplitudes
            2. Initial state
            3. Integrator time step


        .. _get_driving_pulses_see_also:
        
        See Also
        --------
        * `propagate()`
        * `propagate_collection()`
        * `propagate_all()`
        * `evolved_expectation_value()`
        * `evolved_expectation_value_all()`
        * `gradient()`
        """
        ...
    def _eager_processing(self,
                          ctrl_amp : np.ndarray[complex],
                          initial_states : np.ndarray[complex],
                          dt: float,
                          frequencies: np.ndarray[complex],
                          number_channels: list[int]
                         ) -> tuple:
        """
        Executes `_pre_processing()` followed by
        `_envolope_processing()` eagerly (i.e. without using a
        `TensorFlow <https://www.tensorflow.org>`__ graph). Nonetheless,
        `_eager_processing()` is still auto differentiable.

        Parameters
        ----------
        frequencies: NDArray[Shape[`qubits`], float]
            The frequencies of the to drive X on each of the qubits
        amplitudes: NDArray[Shape[`qubits`], complex]
            The amplitude of to drive X on each of the qubits
        T: float
            The time to evolve the system for

        Warning
        -------
        Keyword arguments are not supported.

        Returns
        -------
        tuple[tf.Tensor[Shape[n_time_steps, `n_ctrl`], complex], tf.Tensor[Shape[`state_shape`], complex], tf.Tensor[Shape[], float]]
            A tuple of:
            1. Control amplitudes
            2. Initial state
            3. Integrator time step
        """
        ...
    def _traceable_eager_processing(self,
                                   ctrl_amp : np.ndarray[complex],
                                   initial_states : np.ndarray[complex],
                                   dt: float,
                                   frequencies: np.ndarray[complex],
                                   number_channels: list[int]
                                  ) -> tuple:
        """
        A function that will be traced by
        `TensorFlow <https://www.tensorflow.org>`__ to produce a graph of
        `_pre_processing()` followed by `_envolope_processing()`.

        Parameters
        ----------
        frequencies: NDArray[Shape[`qubits`], float]
            The frequencies of the to drive X on each of the qubits
        amplitudes: NDArray[Shape[`qubits`], complex]
            The amplitude of to drive X on each of the qubits
        T: float
            The time to evolve the system for

        Warning
        -------
        Keyword arguments are not supported.

        Returns
        -------
        tuple[tf.Tensor[Shape[n_time_steps, `n_ctrl`], complex], tf.Tensor[Shape[`state_shape`], complex], tf.Tensor[Shape[], float]]
            A tuple of:
            1. Control amplitudes
            2. Initial state
            3. Integrator time step
        """
        ...
    def gradient(self,
                 ctrl_amp : np.ndarray[complex],
                 initial_state : np.ndarray[complex],
                 dt: float,
                 frequencies: np.ndarray[complex],
                 number_channels: list[int],
                 observable : np.ndarray[complex]
                ) -> tuple[float, np.ndarray[float]]:
        """
        Evolves a state vector under the time-dependent Hamiltonian defined by
        the control amplitudes and computes the expectation value of a specified
        observable with respect to the final state and then computes the
        gradient of the final state with respect to the first argument
        (`args[0]`) using `switching_function()` from
        `PySTE <https://PySTE.readthedocs.io>`__.

        Parameters
        ----------
        frequencies: NDArray[Shape[`qubits`], float]
            The frequencies of the to drive X on each of the qubits
        amplitudes: NDArray[Shape[`qubits`], complex]
            The amplitude of to drive X on each of the qubits
        T: float
            The time to evolve the system for
        observable : NDArray[Shape[`dim`, `dim`], complex]
            The observable to take the expectation value of.

        Warning
        -------
        Keyword arguments are not supported.

        Returns
        -------
        tuple[complex, NDArray[Shape[n_parameters], float]]
            A tuple of the expectation value and the gradient.

        See Also
        --------
        * `evolved_expectation_value()`
        * `evolved_expectation_value_all()`
        """
        ...