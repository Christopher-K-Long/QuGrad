"Utility functions for pulse generation and manipulation."

import typing

from typing import Callable

g_out = typing.TypeVar("g_out")
g_args = typing.TypeVar("g_args")
g_kwargs = typing.TypeVar("g_kwargs")
f_out = typing.TypeVar("f_out")
f_args = typing.TypeVar("f_args")
f_kwargs = typing.TypeVar("f_kwargs")

def compose(f: Callable[[g_out, f_args, f_kwargs], f_out],
            g: Callable[[g_args, g_kwargs], g_out],
            *args: f_args,
            **kwargs: f_kwargs
           ) -> Callable[[g_args, g_kwargs], f_out]:
    """
    Creates a function ``h`` given by the composition of `f` and `g`::

        h(...)=f(g(...))
        
    That is the output of `g` is piped into the first argument of `f`.

    Additionally, additionaly arguments and keyword arguments can be passed to
    `f` using `args` and `kwargs`::

        h(...)=f(g(...), *args, **kwargs)
        
    Explicitly ``h`` is defined as::

        h = lambda *a, **kw: f(g(*a, **kw), *args, **kwargs)
        
    Parameters
    ----------
    f : Callable[[g_out, f_args, f_kwargs], f_out]
        The second function to be called in the composition
    g : Callable[[g_args, g_kwargs], g_out]
        The first function to be called in the composition
    *args : f_args
        Additional arguments to pass to ``f``
    **kwargs : f_kwargs
        Additional keyword arguments to pass to ``f``

    Returns
    -------
    Callable[[g_args, g_kwargs], f_out]
        The composite function ``h``

    See Also
    --------
    :func:`compose_unpack()`
    """
    return lambda *a, **kw: f(g(*a, **kw), *args, **kwargs)

def compose_unpack(f: Callable[[g_out, f_args, f_kwargs], f_out],
                   g: Callable[[g_args, g_kwargs], g_out],
                   *args: f_args,
                   **kwargs: f_kwargs
                  ) -> Callable[[g_args, g_kwargs], f_out]:
    """
    Creates a function ``h`` given by the composition of `f` and ``*g``::
        
        h(...) = f(*g(...))

    That is the output of `g` is unpacked and piped into the arguments of `f`.

    Additionally, additionaly arguments and keyword arguments can be passed to
    `f` using `args` and `kwargs`::

        h(...) = f(*g(...), *args, **kwargs)

    Explicitly ``h`` is defined as::

        h = lambda *a, **kw: f(*g(*a, **kw), *args, **kwargs)

    Parameters
    ----------
    f : Callable[[g_out, f_args, f_kwargs], f_out]
        The second function to be called in the composition
    g : Callable[[g_args, g_kwargs], g_out]
        The first function to be called in the composition
    *args : f_args
        Additional arguments to pass to ``f``
    **kwargs : f_kwargs
        Additional keyword arguments to pass to ``f``

    Returns
    -------
    Callable[[g_args, g_kwargs], f_out]
        The composite function ``h``

    See Also
    --------
    :func:`compose()`
    """
    return lambda *a, **kw: f(*g(*a, **kw), *args, **kwargs)