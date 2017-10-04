# bipartiteSBM-MCMC [![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](http://makeapullrequest.com) [![Build Status](https://travis-ci.org/junipertcy/bipartiteSBM-MCMC.svg?branch=master)](https://travis-ci.org/junipertcy/bipartiteSBM-MCMC)

**bipartiteSBM-MCMC** is a MCMC sampler for the degree-corrected bipartite Stochastic Block Model. Three sampling procedures are provided.
In `marginalize` and `maximize` mode, the group memberships of each node are found assuming we know the number of communities, (Ka, Kb), of the system.
In `estimate` mode, one samples the posterior distribution directly assuming we do not know the number of groups.

It is also used as a submodule for the [det_k_bisbm](https://github.com/junipertcy/det_k_bisbm) library.

## Table of content

- [Usage](#usage)
    - [Compilation](#compilation)
    - [Options](#options)
- [Examples](#examples)
    - [Example marginalization](#example-marginalization)
    - [Example maximization](#example-maximization)
    - [Example estimation](#example-estimation)
- [Further specs](#further-specs)
    - [Cooling schedule](#cooling-schedule)
    - [Optional membership file](#optional-membership-file)
- [Companion article](#companion-article)


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

### Options:
```commandline
bin/mcmc  
```
with no options will print usage. 

All useful outputs are directed to `stdout`.

## Examples

### <a id="example-marginalization"></a>Example marginalization

In the `marginalization` mode, one samples a pool of equilibrated Markov chain configurations,
compute the marginal distribution (over possible community labels) of each node,
and finally returns the configuration by assigning each node to its maximal possible community label.

```commandline
bin/mcmc -e <edge_list_path> -n <block_sizes> -b <burn_in_steps> -t <sampling_steps> -x <steps_await> -y <block_types> -z <ka> <kb> -f <sampling_frequency> -E <epsilon> --randomize --membership_path <optional_membership_file>
```

* REQUIRED:

    `-e <edge_list_path>` – path to the edgelist file; formatted as [explained](https://github.com/junipertcy/det_k_bisbm#dataset).

    `-n <block_sizes>` – block sizes vector (optional if `--membership_path` is specified).

    `-b <burn_in_steps>` – number of burn-in steps.

    `-t <sampling_steps>` – number of sweeps in the simulated annealing process.

    `-x <steps_await>` – number of accumulated steps before the stop of algorithm when the max/min likelihood shows no change.

    `-y <block_types>` – block types vector. Note that the node indexes should be ordered that `type-a` starts first, and then `type-b` the second.

    `-z <ka> <kb>` – number of type-a and type-b communities (optional if `--membership_path` is specified).

* OPTIONAL:

    `-f <sampling_frequency>` – number of steps between each sample
    
    `-E <epsilon>` – the epsilon parameter for more efficient MCMC sampling on SBM.
    
    `--randomize` – shuffle the community labels of each node.
    
    `--membership_path <optional_membership_file>` – initial node membership configuration for seeding the Markov chain.
 
#### Example call (marginalization):
```commandline
bin/mcmc -e dataset/bisbm-n_1000-ka_4-kb_6-r-1.0-Ka_30-Ir_1.75.gt.edgelist -n 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 -t 20000 -x 100 -b 1000 -y 500 500 -z 10 10 -f 10 > marginalization_result.txt
```
The output is sent to `stdout` thus via passing a pipe `>` to a file path, one could work further on the result.

* OUTPUT:
    > 0 0 1 1 0 0 1 0 1 1 1 0 1 0 0 1 0 1 1 0 1 0 1 1 0 0 0 ...
    
The output is an instance of Monte Carlo samples. Each column represents the community label of each node.

### <a id="example-maximization"></a>Example maximization

In the `maximization` mode, we guess the planted partition by maximizing the likelihood of the partition (with simulated 
annealing). 
There are no `burn-in` process and no `<sampling_frequency>` if our goal is to find the maximal likelihood configuration.

```commandline
bin/mcmc -e <edge_list_path> -n <block_sizes> -t <sampling_steps> -x <steps_await> -y <block_types> -z <ka> <kb> --maximize -c <cooling_schedule> -a <param_1> <param_2> --randomize  -E <epsilon> --membership_path <optional_membership_file>
```

* REQUIRED:

    `-e <edge_list_path>` – path to the edgelist file; formatted as [explained](https://github.com/junipertcy/det_k_bisbm#dataset).
    
    `-n <block_sizes>` – block sizes vector (optional if `--membership_path` is specified).
    
    `-t <sampling_steps>` – number of sweeps in the simulated annealing process.
    
    `-x <steps_await>` – number of accumulated steps before the stop of algorithm when the max/min likelihood shows no change.
    
    `-y <block_types>` – block types vector. Note that the node indexes should be ordered that `type-a` starts first, and then `type-b` the second.
    
    `-z <ka> <kb>` – number of type-a and type-b communities (optional if `--membership_path` is specified).
    
    `--maximize` – maximization mode.
    
    `-c <cooling_schedule>`, and `-a <param_1> <param_2>` – see [cooling schedule specification](#cooling-schedule).

* OPTIONAL:

    `--randomize` – shuffle the community labels of each node.
    
    `-E <epsilon>` – the epsilon parameter for more efficient MCMC sampling on SBM.
    
    `--membership_path <optional_membership_file>` – initial node membership configuration for seeding the Markov chain.


#### Example call (maximization):
The call is similar to that of the marginalization mode:
```commandline
bin/mcmc -e dataset/southernWomen.edgelist -n 4 4 4 3 3 3 3 3 3 2 -t 1000 -x 100 --maximize -c exponential -a 10 0.1 -y 18 14 -z 5 5 -E 0.001 --randomize
```

* OUTPUT:
    > 4 4 4 4 4 4 2 0 2 1 3 3 1 1 1 0 2 2 6 6 6 6 6 8 5 8 8 9 5 7 7 7
    
Same as `marginalization`, the output is an instance of Monte Carlo samples.
Each column represents the community label of each node.


### <a id="example-estimation"></a>Example estimation
In the `estimation` mode, one could use the Markov chain Monte Carlo algorithm to sample the posterior distribution directly.
There are two complementary options for the sampling, one is via passing `--uni --estimate`, the other is via passing `--estimate`.
The prior mode implements the algorithm outlined in the [paper](https://journals.aps.org/pre/abstract/10.1103/PhysRevE.96.032310) by Maria A. Riolo _et al_; the latter model implements the algorithm proposed in our paper.

```commandline
bin/mcmc_history -e <edge_list_path> -n <block_sizes> -t <sampling_steps> -y <block_types> -z <ka> <kb> -f <sampling_frequency> --randomize
```

* REQUIRED:

    `-e <edge_list_path>` – path to the edgelist file; formatted as [explained](https://github.com/junipertcy/det_k_bisbm#dataset).
    
    `-n <block_sizes>` – block sizes vector (optional if `--membership_path` is specified).
    
    `-b <burn_in_steps>` – number of burn-in steps.
    
    `-t <sampling_steps>` – number of sweeps in the simulated annealing process.
    
    `-y <block_types>` – block types vector. Note that the node indexes should be ordered that `type-a` starts first, and then `type-b` the second.
    
    `-z <ka> <kb>` – number of type-a and type-b communities (optional if `--membership_path` is specified).
    
    `--estimate` – mode for estimating the number of communities.
    
* OPTIONAL:

    `--randomize` – shuffle the community labels of each node.
    
    `-f <sampling_frequency>` – number of steps between each sample
    
    `--uni` – whether bipartite structure is strictly adhered.
    
    `--membership_path <optional_membership_file>` – initial node membership configuration for seeding the Markov chain.

#### Example call (estimating the 2D posterior):
```commandline
bin/mcmc_history -e dataset/bisbm-n_1000-ka_4-kb_6-r-1.0-Ka_30-Ir_1.75.gt.edgelist -n 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 -t 100000 -x 10000 -y 500 500 -z 10 10 --randomize --estimate
```

#### Example call (estimating the 1D posterior):
```commandline
bin/mcmc_history -e dataset/bisbm-n_1000-ka_4-kb_6-r-1.0-Ka_30-Ir_1.75.gt.edgelist -n 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 -t 100000 -x 10000 -y 500 500 -z 10 10 --randomize --estimate --uni
```

#### Example call (estimating the 2D posterior with user-defined initial condition):
One could pass a file containing the communities of the nodes (e.g. `<optional_membership_file.txt>`) and initialize the MCMC chain via this starting configuration. 
When this is the case, `-n` and `-z` flag are not needed.
```commandline
time bin/mcmc_history -e ../../dataset/bisbm-n_1000-ka_4-kb_6-r-1.0-Ka_30-Ir_1.75.gt.edgelist -t 10000 -x 10000 -y 500 500 --randomize --estimate --membership_path dataset/optional_membership_file.txt
```
#### Example call (estimating the 1D posterior with user-defined initial condition):
Or, if one does not specifically target a bipartite structure:
 ```commandline
time bin/mcmc_history -e ../../dataset/bisbm-n_1000-ka_4-kb_6-r-1.0-Ka_30-Ir_1.75.gt.edgelist -t 10000 -x 10000 -y 500 500 --randomize --estimate --membership_path dataset/optional_membership_file.txt --uni
 ```
 
* OUTPUT (bipartite):
    > ...<br/>
    99970,4,6,-70531.4,0,0,0,0,0,1,0,0,0,0,0,0,0,0, ...<br/>
    99980,4,6,-70529.3,0,0,1,0,0,0,0,0,0,0,0,0,0,0, ...<br/>
    99990,4,6,-70529.3,0,0,1,0,0,0,0,0,0,0,0,0,0,0, ...
    
    
The output consists of a series of Monte Carlo samples. By default, the only the last 1000 samples are sent to stdout.
The first four columns are Monte Carlo sweep number, number of groups `Ka`, number of groups `Kb` and the log-likelihood,
the latter accurate to within overall additive and multiplicative constants.
The successive columns represent the community labels of each node.

* OUTPUT (unipartite; `--uni` mode):
    > ...<br/>
    99970,10,-61119.1,0,0,0,0,0,1,0,0,0,0,0,0,0,0, ...<br/>
    99980,10,-61118.3,0,0,1,0,0,0,0,0,0,0,0,0,0,0, ...<br/>
    99990,10,-61120,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0, ...
    
Similarly, the output consists of a series of Monte Carlo samples and only the last 1000 samples are printed.  
The first four columns are Monte Carlo sweep number, number of groups `K` and the log-likelihood,
the latter accurate to within overall additive and multiplicative constants.
The successive columns represent the community labels of each node.


## Further specs

### <a id="cooling-schedule"></a>Cooling schedule

Four cooling schedules are implemented: `exponential`, `linear`, `logarithmic` and `constant`. 
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

Note that `<param_2>` is not required when the cooling schedule is `constant`.

### <a id="optional-membership-file"></a>Optional membership file

When one wants to initiate a customized configuration, one should prepare a file, say `optional_membership_file.txt`, which contains one community label per line.
```ini
<community_id_of_node_id_1>
<community_id_of_node_id_2>
...
<community_id_of_node_id_n>
```

## <a id="companion-article"></a>Companion article

Please cite:

**Estimating the number of communities in a bipartite network**

Tzu-Chi Yen and Daniel Larremore, *in preparation*.
