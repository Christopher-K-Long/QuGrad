# Getting Started

## What is QuGrad
A python library for quantum optimal control using [PySTE](https://PySTE.readthedocs.io) for the quantum evolution and gradient calculations and [TensorFlow](https://www.tensorflow.org) for backpropagating gradients through the pulse shaping functions.

## Installation

```{include} ../../README.md
:start-after: "## Installation"
:end-before: "## Documentation"
```

## Quick Start

A simple quantum optimal control problem on one qubit is optimising the drive frequency $\omega$ and amplitude $A$ such that a Rabi oscillation transfers the $\left|0\right\rangle$ state to the $\left|1\right\rangle$ state. To study this we will consider the Hamiltonian

$$
H(t)=Z+A\cos(\omega t)X,
$$

where $X$ and $Z$ are the x and z Pauli matrices. We will evolve the system for $T=10$ units of time.

To perform the optimisation we need an appropriate cost function. Here we use the expectation value of the $Z$ operator as this is minimised by the $\left|1\right\rangle$ state. The following program solves this task:

```{literalinclude} ../../examples/Rabi_oscillation_optimisation.py
```

The output should look like:

```txt
Initial expectation value:  (0.9747897138091817+0j)
Optimisation result:
  message: Optimization terminated successfully.
  success: True
   status: 0
      fun: -0.9999999999995379
        x: [ 1.557e+00  1.473e+00]
      nit: 12
      jac: [-3.482e-07  1.729e-07]
 hess_inv: [[ 7.625e-02 -3.556e-02]
            [-3.556e-02  2.850e-02]]
     nfev: 37
     njev: 37
```
(You may see some output from [TensorFlow](https://www.tensorflow.org) too.)

---
[Previous](overview.md) | [Next](pulse_forms.md)