# bipartiteSBM-MCMC  [![Build Status](https://travis-ci.org/junipertcy/bipartiteSBM-MCMC.svg?branch=master)](https://travis-ci.org/junipertcy/bipartiteSBM-MCMC)

**bipartiteSBM-MCMC** is a MCMC sampler for the degree-corrected bipartite Stochastic Block Model. Three sampling procedures are provided. In `marginalize` and `anneal` mode, the group memberships of each node are found assuming we know the number of communities, (Ka, Kb), of the system. In `estimate` mode, one samples the posterior distribution directly assuming we do not know the number of groups.

It is also used as a submodule for the [det_k_bisbm](https://github.com/junipertcy/det_k_bisbm) library.

## Table of content

1. [Usage](#usage)
  1. [Compilation](#compilation)
  2. [Example marginalization](#example-marginalization)
  3. [Example maximization](#example-maximization)
  3. [Example estimation](#example-estimation)
2. [Companion article](#companion-article)


## Usage

### Compilation

This code requires [compilers](http://en.cppreference.com/w/cpp/compiler_support) that support C++11 features. 
It also depends on `boost::program_options` and `cmake`.

Compilation:
```
cmake .
make
```

The binaries are built in `bin/`.

### Example marginalization

Example call:
```commandline
bin/mcmc -e dataset/southernWomen.edgelist -n 4 4 4 3 3 3 3 3 3 2 -t 1000 -x 100 -y 18 14 -z 5 5 -E 0.001
```

### Example maximization

In the maximization mode, we guess the planted partition by maximizing the likelihood of the partition (with simulated 
annealing).

The call is similar to that of the marginalization mode:
```commandline
bin/mcmc -e dataset/southernWomen.edgelist -n 4 4 4 3 3 3 3 3 3 2 -t 1000 -x 100 --maximize -c exponential -a 10 0.1 -y 18 14 -z 5 5 -E 0.001 --randomize
```

Both the burn-in and sampling frequency are ignored in the maximization mode. 

4 cooling schedules are implemented: `exponential`, `linear`, `logarithmic` and `constant`. 
It is advised to test these annealing scheme in order to decide which best finds the optimum.

There inverse temperature is given as
```
beta(t) = 1/T_0 * alpha^(-t)                (Exponential)
beta(t) = 1/T_0 * [1 - eta * t / T_0]^(-1)  (Linear)
beta(t) = log(t + d) / c                    (Logarithmic)
beta(t) = 1 / T_0                           (Constant)
```

where `t` is the MCMC step. The parameters of these cooling schedule are passed like so:
```
-a T_0 alpha    (Exponential)
-a T_0 eta      (Linear)
-a c d          (Logarithmic)
-a T_0          (Constant)
```

### Example estimation
Example call:
```commandline
bin/mcmc_history -e dataset/southernWomen.edgelist -n 4 4 4 3 3 3 3 3 3 2 -t 1000 -y 18 14 -z 5 5 --estimate -f 10 --randomize
```

## References

Please cite:

**Estimating the number of communities in a bipartite network**

Tzu-Chi Yen and Daniel Larremore, *in preparation*.
