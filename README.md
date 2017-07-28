# bisbm_block_estimation_mcmc

C++ implementation of a MCMC sampler for the bipartite degree-corrected Stochastic Block Model. Three sampling 
procedures are provided. In `marginalize` and `anneal` mode, the group memberships of each node are found assuming we 
know the number of communities, (Ka, Kb), of the system. In `estimate` mode, one samples the posterior distribution as 
we do not know the group sizes, and (Ka, Kb) are also updated and sampled.


## Table of content

1. [Usage](#usage)
  1. [Compilation](#compilation)
  2. [Example marginalization](#example-marginalization)
  3. [Example maximization](#example-maximization)
  3. [Example estimation](#example-estimation)
2. [Companion article](#companion-article)


## Usage

### Compilation

Depends on `boost::program_options` and `cmake`.

Compilation:

	cmake .
	make

The binaries are built in `bin/`.

### Example marginalization

Example call:

	bin/mcmc -e dataset/southernWomen.edgelist0 -n 4 4 4 3 3 3 3 3 3 2 --membership_path southernWomen_membership.txt -t 1000 -y 18 14 -z 5 5


### Example maximization

In the maximization mode, we guess the planted partition by maximizing the likeihood of the partition (with simulated 
annealing).

The call is similar to that of the marginalization mode:

	bin/mcmc -e dataset/southernWomen.edgelist0 -n 4 4 4 3 3 3 3 3 3 2 --membership_path southernWomen_membership.txt -t 1000 -y 18 14 -z 5 5 --maximize -c exponential

Both the burn-in and sampling frequency are ignored in the maximization mode.

4 cooling schedules are implemented: `exponential`, `linear`, `logarithmic` and `constant`.

There inverse temperature is given as

    beta(t) = 1/T_0 * alpha^(-t)                (Exponential)
    beta(t) = 1/T_0 * [1 - eta * t / T_0]^(-1)  (Linear)
    beta(t) = log(t + d) / c                    (Logarithmic)
    beta(t) = 1 / T_0                           (Constant)

where $t$ is the MCMC step. The paramters of these cooling schedule are passed like so:

	-a T_0 alpha    (Exponential)
	-a T_0 eta      (Linear)
	-a c d          (Logarithmic)
	-a T_0          (Constant)

### Example estimation

    bin/mcmc_history -e dataset/southernWomen.edgelist0 -n 4 4 4 3 3 3 3 3 3 2 -t 1000 -y 18 14 -z 5 5 --estimate -f 10 > blocks.txt

## Companion article

Please cite

**Estimating the number of communities in a bipartite network**

Tzu-Chi Yen and Daniel Larremore, *in preparation*.