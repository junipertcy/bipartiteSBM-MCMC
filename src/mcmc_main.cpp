/* ~~~~~~~~~~~~~~~~ Notes ~~~~~~~~~~~~~~~~ */
// A few base assumption go into this program:
//
// - We assume that node identifiers are zero indexed contiguous integers.
// - We assume that block memberships are zero indexed contiguous integers.
// - We assume that the SBM is of the undirected and simple variant.

/* ~~~~~~~~~~~~~~~~ Includes ~~~~~~~~~~~~~~~~ */
// STL
#include <iostream>
#include <chrono>
#include <vector>
#include <utility>
#include <random>
#include <string>
// Boost
#include <boost/program_options.hpp>
// Program headers
#include "types.h"
#include "blockmodel.h"
#include "output_functions.h"
#include "metropolis_hasting.h"
#include "graph_utilities.h"
#include "config.h"

namespace po = boost::program_options;


int main(int argc, char const *argv[]) {
    /* ~~~~~ Program options ~~~~~~~*/
    std::string edge_list_path;
    std::string membership_path;
    uint_vec_t n;
    uint_vec_t y;
    uint_vec_t z;
    unsigned int burn_in;
    unsigned int sampling_steps;
    unsigned int sampling_frequency;
    unsigned int steps_await;
    bool randomize = false;
    bool maximize = false;
    bool estimate = false;
    bool uni1 = false;  // used for experimental comparison
    bool uni2 = false;  // used for experimental comparison
    std::string cooling_schedule;
    float_vec_t cooling_schedule_kwargs(2, 0);
    unsigned int seed = 0;
    unsigned int num_edges = 0;
    double epsilon;

    bool is_bipartite = true;
    bool use_mh_naive = false;
    bool use_mh_tiago = false;

    po::options_description description("Options");
    description.add_options()
            ("edge_list_path,e", po::value<std::string>(&edge_list_path), "Path to edge list file.")
            ("membership_path", po::value<std::string>(&membership_path), "Path to membership file.")
            ("n,n", po::value<uint_vec_t>(&n)->multitoken(), "Block sizes vector.\n")
            ("types,y", po::value<uint_vec_t>(&y)->multitoken(), "Block types vector. (when -v is on)\n")
            ("burn_in,b", po::value < unsigned int > (&burn_in)->default_value(1000), "Burn-in time.")
    ("sampling_steps,t", po::value < unsigned int > (&sampling_steps)->default_value(1000),
            "Number of sampling steps in marginalize mode. Length of the simulated annealing process.")
    ("sampling_frequency,f", po::value < unsigned int > (&sampling_frequency)->default_value(10),
            "Number of step between each sample in marginalize mode. Unused in likelihood maximization mode.")
    ("bisbm_partition,z", po::value<uint_vec_t>(&z)->multitoken(), "BISBM number of blocks to be inferred.")
            ("maximize,m", "Maximize likelihood instead of marginalizing.")
            ("estimate,l", "Estimate KA and KB during marginalizing.")
            ("uni1", "Experimental use; Estimate K during marginalizing – Riolo's approach.")
            ("uni2",
             "Experimental use; Estimate KA + KB during marginalizing – Riolo's approach, but jumps that violate the bipartite structure are simply rejected.")
            ("cooling_schedule,c", po::value<std::string>(&cooling_schedule)->default_value("exponential"),
             "Cooling schedule for the simulated annealing algorithm. Options are exponential, "\
             "linear, logarithmic and constant.")
            ("cooling_schedule_kwargs,a", po::value<float_vec_t>(&cooling_schedule_kwargs)->multitoken(),
             "Additional arguments for the cooling schedule provided as a list of floats. "\
             "Depends on the choice of schedule:\n"\
             "Exponential: T_0 (init. temperature > 0)\n"\
             "             alpha (in ]0,1[).\n"\
             "Linear: T_0 (init. temperature > 0)\n"\
             "        eta (rate of decline).\n"\
             "Logarithmic: c (rate of decline)\n"\
             "             d (delay > 1)\n"\
             "Constant: T (temperature > 0)")
            ("steps_await,x", po::value<unsigned int>(&steps_await)->default_value(1000),
            "Stop the algorithm after x successive sweeps occurred and both the max/min entropy values did not change.")
    ("epsilon,E", po::value<double>(&epsilon)->default_value(1.),
            "The parameter epsilon for faster vertex proposal moves (in Tiago Peixoto's prescription).")
            ("randomize,r",
             "Randomize initial block state.")
            ("seed,d", po::value < unsigned
    int > (&seed), "Seed of the pseudo random number generator (Mersenne-twister 19937). "\
            "A random seed is used if seed is not specified.")
    ("help,h", "Produce this help message.");

    po::variables_map var_map;
    po::store(po::parse_command_line(argc, argv, description), var_map);
    po::notify(var_map);

    if (var_map.count("help") > 0 || argc == 1) {
#if OUTPUT_HISTORY == 0
        std::cout << "MCMC algorithms for the SBM (final output only)\n";
#else
        std::cout << "MCMC algorithms for the SBM (output intermediate states)\n";
#endif
        std::cout << "Usage:\n"
                  << "  " + std::string(argv[0]) + " [--option_1=value] [--option_s2=value] ...\n";
        std::cout << description;
        return 0;
    }
    if (var_map.count("edge_list_path") == 0) {
        std::cout << "edge_list_path is required (-e flag)\n";
        return 1;
    }

    if (var_map.count("types") == 0 && var_map.count("use_bisbm") > 0) {
        std::cout << "types is required for bisbm mode (-y flag)\n";
        return 1;
    }
    if (var_map.count("bisbm_partition") == 0) {
        std::cout << "number of partitions is required (-z flag)\n";
        return 1;
    }

    if (var_map.count("estimate") > 0) {
        estimate = true;
    }

    if (var_map.count("uni1") > 0) {
        uni1 = true;
        is_bipartite = false;
    } else if (var_map.count("uni2") > 0) {
        uni2 = true;
        is_bipartite = false;
    }

    if (var_map.count("maximize") > 0) {
        maximize = true;
        if (var_map.count("cooling_schedule_kwargs") == 0) {
            // defaults
            if (cooling_schedule == "exponential") {
                cooling_schedule_kwargs[0] = 1;
                cooling_schedule_kwargs[1] = 0.99;
            }
            if (cooling_schedule == "linear") {
                cooling_schedule_kwargs[0] = sampling_steps + 1;
                cooling_schedule_kwargs[1] = 1;
            }
            if (cooling_schedule == "logarithmic") {
                cooling_schedule_kwargs[0] = 1;
                cooling_schedule_kwargs[1] = 1;
            }
            if (cooling_schedule == "constant") {
                cooling_schedule_kwargs[0] = 1;
            }
        } else {
            // kwards not defaulted, must check.
            if (cooling_schedule == "exponential") {
                if (cooling_schedule_kwargs[0] <= 0) {
                    std::cerr << "Invalid cooling schedule argument for linear schedule: T_0 must be grater than 0.\n";
                    std::cerr << "Passed value: T_0=" << cooling_schedule_kwargs[0] << "\n";
                    return 1;
                }
                if (cooling_schedule_kwargs[1] <= 0 || cooling_schedule_kwargs[1] >= 1) {
                    std::cerr
                            << "Invalid cooling schedule argument for exponential schedule: alpha must be in ]0,1[.\n";
                    std::cerr << "Passed value: alpha=" << cooling_schedule_kwargs[1] << "\n";
                    return 1;
                }
            } else if (cooling_schedule == "linear") {
                if (cooling_schedule_kwargs[0] <= 0) {
                    std::cerr << "Invalid cooling schedule argument for linear schedule: T_0 must be grater than 0.\n";
                    std::cerr << "Passed value: T_0=" << cooling_schedule_kwargs[0] << "\n";
                    return 1;
                }
                if (cooling_schedule_kwargs[1] <= 0 || cooling_schedule_kwargs[1] > cooling_schedule_kwargs[0]) {
                    std::cerr << "Invalid cooling schedule argument for linear schedule: eta must be in ]0, T_0].\n";
                    std::cerr << "Passed value: T_0=" << cooling_schedule_kwargs[0]
                              << ", eta=" << cooling_schedule_kwargs[1] << "\n";
                    return 1;
                }
                if (cooling_schedule_kwargs[1] * sampling_steps > cooling_schedule_kwargs[0]) {
                    std::cerr
                            << "Invalid cooling schedule argument for linear schedule: eta * sampling_steps must be smaller or equal to T_0.\n";
                    std::cerr << "Passed value: eta*sampling_steps=" << cooling_schedule_kwargs[1] * sampling_steps
                              << ", T_0=" << cooling_schedule_kwargs[0] << "\n";
                    return 1;
                }
            } else if (cooling_schedule == "logarithmic") {
                if (cooling_schedule_kwargs[0] <= 0) {
                    std::cerr
                            << "Invalid cooling schedule argument for logarithmic schedule: c must be greater than 0.\n";
                    std::cerr << "Passed value: c=" << cooling_schedule_kwargs[0] << "\n";
                    return 1;
                }
                if (cooling_schedule_kwargs[1] <= 0) {
                    std::cerr
                            << "Invalid cooling schedule argument for logarithmic schedule: d must be greater than 0.\n";
                    std::cerr << "Passed value: d=" << cooling_schedule_kwargs[1] << "\n";
                    return 1;
                }
            } else if (cooling_schedule == "constant") {
                if (cooling_schedule_kwargs[0] <= 0) {
                    std::cerr
                            << "Invalid cooling schedule argument for constant schedule: temperature must be greater than 0.\n";
                    std::cerr << "Passed value: T=" << cooling_schedule_kwargs[0] << "\n";
                    return 1;
                }
            } else {
                std::cerr << "Invalid cooling schedule. Options are exponential, linear, logarithmic.\n";
                return 1;
            }
        }
    }
    if (var_map.count("epsilon") > 0) {
        std::clog << "An epsilon param is assigned; we will use Tiago's smart MCMC moves. \n";
        use_mh_tiago = true;
    } else {
        std::clog << "An epsilon param is NOT assigned; we will use naive MCMC moves. \n";
        use_mh_naive = true;
    }
    if (var_map.count("randomize") > 0) {
        randomize = true;
    }
    if (var_map.count("seed") == 0) {
        // seeding based on the clock
        seed = (unsigned int) std::chrono::high_resolution_clock::now().time_since_epoch().count();
    }

    /* ~~~~~ Setup objects ~~~~~~~*/
    std::mt19937 engine(seed);
    // number of blocks
    unsigned int g = unsigned(int(n.size()));

    uint_vec_t memberships_init;
    if (var_map.count("membership_path") != 0) {
        std::clog << "Now trying to read membership from membership_path.\n";

        if (!load_memberships(memberships_init, membership_path)) {
            std::clog << "WARNING: error in loading memberships, read memberships from block sizes\n";
            if (var_map.count("n") == 0) {
                std::cout << "n is required (-n flag)\n";
                return 1;
            }
            // memberships from block sizes
            {
                unsigned int accu = 0;
                for (auto it = n.begin(); it != n.end(); ++it) {
                    accu += *it;
                }
                memberships_init.resize(accu, 0);
                unsigned shift = 0;
                for (unsigned int r = 0; r < n.size(); ++r) {
                    for (unsigned int i = 0; i < n[r]; ++i) {
                        memberships_init[shift + i] = r;
                    }
                    shift += n[r];
                }
            }
        } else {
            randomize = false;
            std::clog << " ---- read membership from file! ---- \n";
        }
    } else {
        // memberships from block sizes
        if (var_map.count("n") == 0) {
            std::cout << "n is required (-n flag)\n";
            return 1;
        }
        {
            unsigned int accu = 0;
            for (auto it = n.begin(); it != n.end(); ++it) {
                accu += *it;
            }
            memberships_init.resize(accu, 0);
            unsigned shift = 0;
            for (unsigned int r = 0; r < n.size(); ++r) {
                for (unsigned int i = 0; i < n[r]; ++i) {
                    memberships_init[shift + i] = r;
                }
                shift += n[r];
            }
        }
    }
    uint_vec_t types_init;
    unsigned int KA;
    unsigned int KB;
    unsigned int NA;
    unsigned int NB;
    // number of types (must be equal to 2; TODO: support multipartite version!)
    unsigned int num_types = unsigned(int(y.size()));
    if (num_types != 2) {
        std::cerr << "Number of types must be equal to 2!" << "\n";
        return 1;
    }

    // number of type-a and type-b vertices
    NA = y[0];
    NB = y[1];
    KA = z[0];
    KB = z[1];

    {
        unsigned int accu = 0;
        for (auto it = n.begin(); it != n.end(); ++it) {
            accu += *it;
        }
        types_init.resize(accu, 0);
        unsigned shift = 0;
        for (unsigned int r = 0; r < y.size(); ++r) {
            for (unsigned int i = 0; i < y[r]; ++i) {
                types_init[shift + i] = r;
            }
            shift += y[r];
        }
    }
    // sanity check for the "types"-vector
    if (memberships_init.size() != types_init.size()) {
        std::cerr << memberships_init.size() << ", " << types_init.size() << '\n';
        std::cerr << "Types do not sum to the number of vertices!" << "\n";
        return 1;
    }
    // number of vertices
    unsigned int N = 0;
    for (unsigned int i = 0; i < g; ++i) {
        N += n[i];
    }
    // Graph structure
    edge_list_t edge_list;
    load_edge_list(edge_list, edge_list_path);
    adj_list_t adj_list = edge_to_adj(edge_list, N);
    num_edges = unsigned(int(edge_list.size()));
    edge_list.clear();

    // blockmodel
    blockmodel_t blockmodel(memberships_init, types_init, g, KA, KB, epsilon, unsigned(int(adj_list.size())),
                            &adj_list, is_bipartite);
    memberships_init.clear();
    types_init.clear();
    if (randomize) {
        blockmodel.shuffle_bisbm(engine, NA, NB);
    }
    // probabilities
    float_mat_t p(g, float_vec_t(g, 0));
    uint_mat_t m = blockmodel.get_m();
    for (unsigned int r = 0; r < g; ++r) {
        for (unsigned int s = 0; s < g; ++s) {
            p[r][s] = m[r][s];
        }
    }

    // Bind proper Metropolis-Hasting algorithm
    // We have three modes: marginalizing, estimating, and annealing
    //
    // marginalizing: naive_mcmc, tiago_mcmc, and heat_bath are allowed
    // estimating: only heat_bath is allowed
    // annealing: naive_mcmc and tiago_mcmc is allowed

    std::unique_ptr<metropolis_hasting> algorithm;
    if (maximize && !estimate) {
        if (use_mh_naive) {
            std::clog << "We use naive jumps to maximize the posterior...\n";
            algorithm = std::make_unique<mh_naive>();
        } else {
            std::clog << "We use smart jumps to maximize the posterior...\n";
            algorithm = std::make_unique<mh_tiago>();
        }
    } else if (estimate && !maximize && uni1) {  // Riolo's implementation
        std::clog << "We use Riolo's jumps to sample the posterior...\n";
        algorithm = std::make_unique<mh_riolo_uni1>();
    } else if (estimate && !maximize && uni2) {  // Riolo's implementation, but reject jumps that violate bipartite constraint
        std::clog << "We use Riolo's jumps to sample the posterior...\n";
        algorithm = std::make_unique<mh_riolo_uni2>();
    } else if (estimate && !maximize) {
        std::clog << "We use Riolo's jumps to sample the posterior...\n";
        algorithm = std::make_unique<mh_riolo>();
    } else {
        // marginalize
        if (use_mh_naive) {
            std::clog << "We use naive jumps to marginalize the posterior...\n";
            algorithm = std::make_unique<mh_naive>();
        } else if (use_mh_tiago) {
            std::clog << "We use smart jumps to marginalize the posterior...\n";
            algorithm = std::make_unique<mh_tiago>();
        } else { // heat-bath
            std::clog << "We use heat-bath jumps to marginalize the posterior...\n";
            algorithm = std::make_unique<mh_heat_bath>();
        }
    }
    /* ~~~~~ Logging ~~~~~~~*/
#if LOGGING == 1
    std::clog << "edge_list_path: " << edge_list_path << "\n";
    std::clog << "probabilities:\n";
    output_mat<float_mat_t>(p, std::clog);
    std::clog << "sizes (g=" << n.size() << "): ";
    for (auto it = n.begin(); it != n.end(); ++it)
        std::clog << *it << " ";
    std::clog << "\n";
    std::clog << "burn_in: " << burn_in << "\n";
    std::clog << "sampling_steps: " << sampling_steps << "\n";
    std::clog << "sampling_frequency: " << sampling_frequency << "\n";
    std::clog << "steps_await: " << steps_await << "\n";
    std::clog << "epsilon: " << epsilon << "\n";
    if (maximize) { std::clog << "maximize: true\n"; }
    else { std::clog << "maximize: false\n"; }
    if (randomize) { std::clog << "randomize: true\n"; }
    else { std::clog << "randomize: false\n"; }
    std::clog << "num_vertice_types: (y=" << y.size() << "): ";
    for (auto it = y.begin(); it != y.end(); ++it)
        std::clog << *it << " ";
    std::clog << "\n";

    std::clog << "multipartite_blocks: (z=" << z.size() << "): ";
    for (auto it = z.begin(); it != z.end(); ++it)
        std::clog << *it << " ";
    std::clog << "\n";
    if (maximize) {
        std::clog << "cooling_schedule: " << cooling_schedule << "\n";
        std::clog << "cooling_schedule_kwargs: ";
        output_vec<float_vec_t>(cooling_schedule_kwargs);
    }
    std::clog << "seed: " << seed << "\n";
#endif
    /* ~~~~~ Actual algorithm ~~~~~~~*/
    double rate = 0;
    uint_mat_t marginal(adj_list.size(), uint_vec_t(g, 0));
    if (maximize && !estimate) {
        if (cooling_schedule == "exponential") {
            algorithm->anneal(blockmodel, p, &exponential_schedule, cooling_schedule_kwargs, sampling_steps,
                              steps_await, engine);
        }
        if (cooling_schedule == "linear") {
            algorithm->anneal(blockmodel, p, &linear_schedule, cooling_schedule_kwargs, sampling_steps, steps_await,
                              engine);
        }
        if (cooling_schedule == "logarithmic") {
            algorithm->anneal(blockmodel, p, &logarithmic_schedule, cooling_schedule_kwargs, sampling_steps,
                              steps_await, engine);
        }
        if (cooling_schedule == "constant") {
            algorithm->anneal(blockmodel, p, &constant_schedule, cooling_schedule_kwargs, sampling_steps, steps_await,
                              engine);
        }
        output_vec<uint_vec_t>(blockmodel.get_memberships(), std::cout);
    } else if (estimate && !maximize) {  // estimate
        algorithm->estimate(blockmodel, marginal, p, burn_in, sampling_frequency, sampling_steps, engine);
    } else  // marginalize
    {

        rate = algorithm->marginalize(blockmodel, marginal, p, burn_in, sampling_frequency, sampling_steps, engine);
        uint_vec_t memberships(blockmodel.get_N(), 0);
        for (unsigned int i = 0; i < blockmodel.get_N(); ++i) {
            unsigned int max = 0;
            for (unsigned int r = 0; r < g; ++r) {
                if (marginal[i][r] > max) {
                    memberships[i] = r;
                    max = marginal[i][r];
                }
            }
        }
        output_vec<uint_vec_t>(memberships, std::cout);
        std::clog << "acceptance ratio " << rate << "\n";
    }
    return 0;
}
