Optimization Algorithms
=======================

Optking performs Newton-Raphson (NR) and Qausi-NR optimizations for small to medium molecules.

Algorithms include
    * Steepest descent
    * NR
    * Rational function optimization (RFO)
    * Restricted Step RFO
    * Partioned RFO
    * Intrinsic Reaction Coordinate (IRC) optimizations 

API Documentation
-----------------

For more information on the OptHelper classes see `optimizations`_. To simply run an optimization
the OptHelper classes or the interfaces through QCEngine and Psi4 are recommended.

For more information on interacting directly with the algorithms see the API documentation below. In short,
the OptimizationAlgorithm class provides a basic interface for running optimizations namely the `take_step()`
method. The OptimizationManager class extends this interface to encompass the addition of linesearching
to any of the basic algorithms and IRCFollowing.

.. automodapi:: optking.stepAlgorithms
.. automodapi:: optking.linesearch
.. automodapi:: optking.IRCfollowing
