# Pulse Forms

We saw in the [quick start section](getting_started.md#quick-start) that we can use [``pulse_form()``](../reference/_autosummary/qugrad.QuantumSystem.rst#qugrad.QuantumSystem.pulse_form) to re-parameterise the control amplitudes. Alternatively, we can subclass [``QuantumSystem``](../reference/_autosummary/qugrad.QuantumSystem.rst). If we subclass [``QuantumSystem``](../reference/_autosummary/qugrad.QuantumSystem.rst) and override [``_pre_processing()``](../reference/_autosummary/qugrad.QuantumSystem.rst#qugrad.QuantumSystem._pre_processing) we should also create a [``.pyi`` stub file](https://mypy.readthedocs.io/en/stable/stubs.html) file to document the new interfaces for:

- [``propagate()``](../reference/_autosummary/qugrad.QuantumSystem.rst#qugrad.QuantumSystem.propagate)
- [``propagate_collection()``](../reference/_autosummary/qugrad.QuantumSystem.rst#qugrad.QuantumSystem.propagate_collection)
- [``propagate_all()``](../reference/_autosummary/qugrad.QuantumSystem.rst#qugrad.QuantumSystem.propagate_all)
- [``evolved_expectation_value()``](../reference/_autosummary/qugrad.QuantumSystem.rst#qugrad.QuantumSystem.evolved_expectation_value)
- [``evolved_expectation_value_all()``](../reference/_autosummary/qugrad.QuantumSystem.rst#qugrad.QuantumSystem.evolved_expectation_value_all)
- [``get_driving_pulses()``](../reference/_autosummary/qugrad.QuantumSystem.rst#qugrad.QuantumSystem.get_driving_pulses)
- [``gradient()``](../reference/_autosummary/qugrad.QuantumSystem.rst#qugrad.QuantumSystem.gradient)
- ``_eager_processing()``
- ``_traceable_eager_processing()``

That is we should update the arguments in the docstrings to correspond to those of the new [``_pre_processing()``](../reference/_autosummary/qugrad.QuantumSystem.rst#qugrad.QuantumSystem._pre_processing) function.

Below is an example subclass followed the the stub file:

```{literalinclude} ../../examples/subclass.py
```

The corresponding stub file:

```{literalinclude} ../../examples/subclass.pyi
```

---
[Previous](getting_started.md) | [Next](examples.md)