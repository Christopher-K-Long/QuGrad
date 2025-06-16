"""
Defines a class representing a Hilbert space.
"""

import numpy as np
from typing import Optional, Any, Union

class HilbertSpace():
    """
    Represents a Hilbert space with a prefered basis indexed by non-negative
    integers.

    Note
    ----
    The class also can be treated as an iterator and will return :attr:`basis`
    as the iterator.


    ---
    """
    
    _basis: np.ndarray[int]
    "An array of positive integers labeling the Hilbert space basis"
    _inverse: np.ndarray[int]
    "An array satisfying the property ``_inverse[basis[i]]=i``"
    
    def __init__(self, basis: np.ndarray[int]):
        """
        Initialises a new :class:`HilbertSpace`.

        Parameters
        ----------
        basis : NDArray[Shape[:attr:`dim`], int]
            The labels for the basis vectors.
        """
        self._basis = np.array(basis)
        self._basis.flags.writeable = False
        if len(self._basis) == 0:
            self._inverse = np.empty(0, dtype=np.int64)
        else:
            self._inverse = np.zeros(self._basis.max()+1, dtype=np.int64)
        for i, x in enumerate(self._basis):
            self._inverse[x] = i
        self._inverse.flags.writeable = False
    def __iter__(self):         return iter(self._basis)
    def __getitem__(self, key): return self._basis[key]
    def __len__(self):          return len(self._basis)
    def __eq__(self, other: "HilbertSpace") -> bool:
        if isinstance(other, self.__class__):
            return np.array_equal(self._basis, other._basis)
        return False
    @property
    def dim(self):
        """
        The Hilbert space dimension
        """
        return len(self)
    @property
    def basis(self) -> np.ndarray[int]:
        """
        An array of positive integers labeling the Hilbert space basis 
        """
        return self._basis
    @property
    def inverse(self) -> np.ndarray[int]:
        "An array satisfying the property ``self.inverse[self.basis[i]]=i``"
        return self._inverse
    def basis_vector(self, basis_state_label: int) -> np.ndarray[np.complex128]:
        """Returns a column vector represnetation corresponding to the input
        basis state label.

        Parameters
        ----------
        basis_state_label : int
            A positive integer denoting the label of the basis state to generate 
            the basis vector for.

        Returns
        -------
        NDArray[Shape[:attr:`dim`], np.complex128]
            The basis vector corresponding to the input basis state label.
        """
        return np.eye(1,
                      len(self._basis),
                      self._inverse[basis_state_label]
                     ).flatten().astype(np.complex128)
    def get_subspace(self, filter: np.ndarray[bool]) -> "HilbertSpace":
        """Generates a new subspace by filtering the basis state labels.

        Parameters
        ----------
        filter : NDArray[Shape[:attr:`dim`], bool]
            ``True`` entries are retained in the new subspace.

        Returns
        -------
        HilbertSpace
            The filtered subspace.
        """
        return HilbertSpace(self._basis[filter])
    @staticmethod
    def _labels(state: Any) -> str:
        """
        Generates a strings that represent the state.

        Parameters
        ----------
        state : Any
            The state to label.

        Returns
        -------
        str
            The label for the specified state.
        """
        return f"|{repr(state)}⟩"
    def labels(self,
               states: Optional[Union[int, list[int]]] = None
              ) -> Union[str, list[str]]:
        """
        Generates a string (list of strings) that represent the state(s).

        Parameters
        ----------
        states : int | list[int], optional
            The state(s) to label. If ``None`` then the labels for all states in
            :attr:`basis` are returned. By default ``None``.

        Returns
        -------
        str | list[str]
            The label(s) for the specified states.
        """
        if isinstance(states, int):
            return self._labels(states)
        elif states is None: states = self.basis
        return [self._labels(int(s)) for s in states]