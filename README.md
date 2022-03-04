# Magneto-Coriolis modes and torsional oscillations in a full sphere

## Equation 
Momentum: ...

Induction: ...

Background/imposed fields are assumed to be axisymmetric, and matrices produced at a given m.

## Benchmarks
Some benchmark tests can be found in /test
### Kinematic dynamo

### Inertial modes

### Magneto-Coriolis modes

## Get the code
`git clone https://github.com/guagualuo/MCmodes.git`

This code would rely on some of the sparse linear operators implemented in [QuICC](https://github.com/QuICC)

One should also do

`git clone https://github.com/QuICC/QuICC.git`

to get QuICC.

Since we only need contents in QuICC/Python/, and due to its internal implementation, currently one has to append dir_contains_QuICC/QuICC/Python in the sys.path variable, or add it as a content root if you are using PyCharm for example. 
