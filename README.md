# A Computational Fluid Dynamics Solver for Simulating a Karman Vortex Street

This is a simple CFD simulation application, implemented according to the computational scheme by Griebel et al. [1]. It is set up to solve a problem often studied using CFD, known as a Kármán vortex street.

By default, the problem is set up with a 4.0m x 1.0m domain, discretised with a 512 x 128 grid, simulating 5 seconds of time of fluid flowing passed a circular object.

At the end of execution, a VTK file is produced for visualisation purposes. This can be loaded into VisIt for analysis.

## Building

The application can be built with the provided Makefile. e.g.

```
$ make
```

This will build a `vortex` binary.

## Running

The application can be ran in its default configuration with:

```
$ ./vortex
```

This will output its status every 100 iterations. At the end of execution a VTK file will be produced. 

There are numerous other options available. These can be queried with:

```
$ ./vortex --help
```

To write out the simulation state every 100 iterations, you could use:

```
$ mkdir out
$ ./vortex -c -o out/my_sim
```

## References

[1]: Michael Griebel, Thomas Dornseifer, Tilman Neunhoeffer, “Numerical Simulation in Fluid Dynamics”, SIAM, 1998. [https://people.math.sc.edu/Burkardt/cpp_src/nast2d/nast2d.html](https://people.math.sc.edu/Burkardt/cpp_src/nast2d/nast2d.html)
