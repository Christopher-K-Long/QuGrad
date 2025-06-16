from qugrad.pulses import compose, compose_unpack

def test_compose():
    def g(x):
        return x + 3

    def f(x):
        return x ** 2

    h = compose(f, g)
    assert h(3) != g(f(3)) # check the test functions don't commute
    assert h(3) == f(g(3)) # check the composition is correct

def test_compose_additional_args():
    def g(x):
        return x + 3

    def f(x, a, power):
        return x ** power + a

    h = compose(f, g, 5, power=3)
    assert h(3) == f(g(3), 5, power=3)

def test_compose_uppack():
    def g(x, y):
        return x + y, x*y

    def f(x, y):
        return x ** y

    h = compose_unpack(f, g)
    try:
        assert h(2, 3) != g(*f(2, 3)) # check the test functions don't commute
    except TypeError:
        pass
    assert h(2, 3) == f(*g(2, 3)) # check the composition is correct

def test_compose_uppack_additional_args():
    def g(x, y):
        return x + y, x*y

    def f(x, y, a, b):
        return x ** y + a*b

    h = compose_unpack(f, g, 5, b=3)
    assert h(2, 3) == f(*g(2, 3), 5, b=3)