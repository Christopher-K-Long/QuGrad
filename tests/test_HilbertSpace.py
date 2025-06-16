from qugrad import HilbertSpace
import numpy as np

def test_initialisation():
    """
    Test the initialisation of the HilbertSpace class.
    """
    basis = [0, 1, 5, 3]
    hilbert_space = HilbertSpace(basis)
    
    # Check that the basis is correctly set
    assert np.array_equal(hilbert_space.basis, basis)
    
    # Check that the inverse is correctly set
    x = hilbert_space.basis[hilbert_space.inverse[hilbert_space.basis]]
    assert np.array_equal(x, basis)
    
    # Check that the dimension is correct
    assert hilbert_space.dim == len(basis)

def test_basis_read_only():
    hilbert_space = HilbertSpace([0])
    try:
        hilbert_space.basis = np.array([0])
    except AttributeError:
        pass
    else:
        raise AssertionError("HilbertSpace.basis should be read-only")
    
    try:
        hilbert_space.basis[0] = 0
    except ValueError:
        pass
    else:
        raise AssertionError("HilbertSpace.basis should be read-only")

def test_inverse_read_only():
    hilbert_space = HilbertSpace([0])

    try:
        hilbert_space.inverse = np.array([0])
    except AttributeError:
        pass
    else:
        raise AssertionError("HilbertSpace.inverse should be read-only")
    
    try:
        hilbert_space.inverse[0] = 0
    except ValueError:
        pass
    else:
        raise AssertionError("HilbertSpace.inverse should be read-only")

def test_dim_read_only():
    hilbert_space = HilbertSpace([0])
    try:
        hilbert_space.dim = 1
    except AttributeError:
        pass
    else:
        raise AssertionError("HilbertSpace.dim should be read-only")

def test_iter():
    hilbert_space = HilbertSpace([0, 1, 5, 3])
    assert len(hilbert_space) == len(hilbert_space.basis)
    for n, label in enumerate(hilbert_space):
        assert label == hilbert_space.basis[n]
        assert label == hilbert_space[n]

def test_basis_vector(): 
    hilbert_space = HilbertSpace([0, 1, 5, 3])

    for n, label in enumerate(hilbert_space.basis):
        assert np.array_equal(hilbert_space.basis_vector(label),
                              np.array([0]*n+[1]+[0]*(len(hilbert_space)-n-1)))

def test_get_subspace():
    hilbert_space = HilbertSpace([0, 1, 5, 3])
    filter = np.array([True, False, True, False])
    subspace = hilbert_space.get_subspace(filter)

    assert isinstance(subspace, HilbertSpace)
    
    assert np.array_equal(subspace.basis, hilbert_space.basis[filter])
    
    x = subspace.basis[subspace.inverse[subspace.basis]]
    assert np.array_equal(x, subspace.basis)
    
    assert subspace.dim == len(subspace.basis)

def test_labels():
    hilbert_space = HilbertSpace([0, 1, 5, 3])
    labels = hilbert_space.labels()
    for n, label in enumerate(labels):
        assert label == f"|{str(hilbert_space[n])}⟩"

    some_labels = hilbert_space.labels([1, 5])

    assert np.array_equal(some_labels, ["|1⟩", "|5⟩"])

    single_label = hilbert_space.labels(1)
    assert single_label == "|1⟩"

def test_empty_HilbertSpace():
    hilbert_space = HilbertSpace([])
    assert len(hilbert_space) == 0
    assert len(hilbert_space.basis) == 0
    assert len(hilbert_space.inverse) == 0
    try:
        hilbert_space.basis_vector(0)
    except IndexError:
        pass
    else:
        raise AssertionError("HilbertSpace.basis_vector should raise IndexError for empty HilbertSpace")
    subspace = hilbert_space.get_subspace([])
    assert isinstance(subspace, HilbertSpace)
    assert len(subspace) == 0
    assert len(subspace.basis) == 0
    assert len(subspace.inverse) == 0
    assert len(hilbert_space.labels()) == 0

def test_equality():
    hilbert_space1 = HilbertSpace([0, 1, 5, 3])
    hilbert_space2 = HilbertSpace([0, 1, 5, 3])
    hilbert_space3 = HilbertSpace([0, 1, 2])
    class SubclassHilbertSpace(HilbertSpace):
        pass
    hilbert_space4 = SubclassHilbertSpace([0, 1, 5, 3])

    assert hilbert_space1 == hilbert_space2
    assert not (hilbert_space1 == hilbert_space3)
    assert not (hilbert_space1 == "not a HilbertSpace")
    assert not (hilbert_space1 == hilbert_space4)

    assert not(hilbert_space1 != hilbert_space2)
    assert hilbert_space1 != hilbert_space3
    assert hilbert_space1 != "not a HilbertSpace"
    assert hilbert_space1 != hilbert_space4