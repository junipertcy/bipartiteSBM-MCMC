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
#include "types.hh"
#include "blockmodel.hh"
#include "output_functions.hh"
#include "metropolis_hasting.hh"
#include "graph_utilities.hh"
#include "support/util.hh"
#include "config.hh"

namespace po = boost::program_options;


int main(int argc, char const *argv[]) {
    /* ~~~~~ Program options ~~~~~~~*/
    size_t KA{0};
    size_t KB{0};
    size_t NA{0};
    size_t NB{0};
    std::string edge_list_path;
    std::string membership_path;
    uint_vec_t n;
    uint_vec_t mb;
    uint_vec_t y;
    uint_vec_t z;
    size_t burn_in;
    size_t sampling_steps;
    size_t sampling_frequency;
    size_t steps_await;
    bool randomize = false;
    bool merge = false;
    std::string cooling_schedule;
    float_vec_t cooling_schedule_kwargs(2, 0);
    size_t seed = 0;
    double epsilon;
    uint_vec_t types_init;

    po::options_description description("Options");
    description.add_options()
            ("edge_list_path,e", po::value<std::string>(&edge_list_path), "Path to edge list file.")
            ("membership_path", po::value<std::string>(&membership_path), "Path to membership file.")
            ("mb", po::value<uint_vec_t>(&mb)->multitoken(), "Path to membership file.")
            ("n,n", po::value<uint_vec_t>(&n)->multitoken(), "Block sizes vector.\n")
            ("types,y", po::value<uint_vec_t>(&y)->multitoken(), "Block types vector. (when -v is on)\n")
            ("burn_in,b", po::value<size_t>(&burn_in)->default_value(1000), "Burn-in time.")
            ("sampling_steps,t", po::value<size_t>(&sampling_steps)->default_value(1000),
             "Number of sampling steps in marginalize mode. Length of the simulated annealing process.")
            ("sampling_frequency,f", po::value<size_t>(&sampling_frequency)->default_value(10),
             "Number of step between each sample in marginalize mode. Unused in likelihood maximization mode.")
            ("bisbm_partition,z", po::value<uint_vec_t>(&z)->multitoken(), "bipartite number of blocks to be inferred.")
            ("uni", "Experimental use; Estimate K during marginalizing – Riolo's approach.")
            ("cooling_schedule,c", po::value<std::string>(&cooling_schedule)->default_value("abrupt_cool"),
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
            ("steps_await,x", po::value<size_t>(&steps_await)->default_value(1000),
             "Stop the algorithm after x successive sweeps occurred and both the max/min entropy values did not change.")
            ("epsilon,E", po::value<double>(&epsilon)->default_value(1.),
             "The parameter epsilon for faster vertex proposal moves (in Tiago Peixoto's prescription).")
            ("randomize,r",
             "Randomize initial block state.")
            ("merge,g",
             "Perform agglomerative merges to the initial block state.")
            ("seed,d", po::value<size_t>(&seed),
             "Seed of the pseudo random number generator (Mersenne-twister 19937). A random seed is used if seed is not specified.")
            ("help,h", "Produce this help message.");

    po::variables_map var_map;
    po::store(po::parse_command_line(argc, argv, description), var_map);
    po::notify(var_map);

    if (var_map.count("help") > 0 || argc == 1) {
        std::cout << "MCMC algorithms for the bipartiteSBM (final output only)\n";
        std::cout << "Usage:\n"
                  << "  " + std::string(argv[0]) + " [--option_1=value] [--option_s2=value] ...\n";
        std::cout << description;
        return 0;
    }
    if (var_map.count("edge_list_path") == 0) {
        std::cerr << "edge_list_path is required (-e flag)\n";
        return 1;
    }

    if (var_map.count("types") == 0) {
        std::cerr << "types is required for bisbm mode (-y flag)\n";
        return 1;
    } else {
        // number of types (must be equal to 2)
        auto num_types = unsigned(int(y.size()));
        if (num_types != 2) {
            std::cerr << "Number of types must be equal to 2!" << "\n";
            return 1;
        } else {
            NA = y[0];
            NB = y[1];
            types_init.resize(NA + NB, 0);
            unsigned shift = 0;
            for (size_t r = 0; r < y.size(); ++r) {
                for (size_t i = 0; i < y[r]; ++i) {
                    types_init[shift + i] = unsigned(int(r));
                }
                shift += y[r];
            }
        }
    }

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
        if (cooling_schedule == "abrupt_cool") {
            cooling_schedule_kwargs[0] = steps_await;
        }
    } else {
        // kwargs not defaulted, must check.
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
        } else if (cooling_schedule == "abrupt_cool") {
            if (cooling_schedule_kwargs[0] <= 0) {
                std::cerr
                        << "Invalid cooling schedule argument for abrupt_cool schedule: tau must be larger than 0. \n";
                std::cerr << "Passed value: tau=" << cooling_schedule_kwargs[0] << "\n";
                return 1;
            }
        } else {
            std::cerr << "Invalid cooling schedule. Options are exponential, linear, logarithmic, abrupt_cool.\n";
            return 1;
        }
    }
    if (var_map.count("epsilon") > 0) {
        std::clog << "An epsilon param is assigned; we will use Tiago Peixoto's smart MCMC moves. \n";
    } else {
        std::cerr << "[ERROR] An epsilon param is NOT assigned; \n";
        std::cerr
                << "to perform MCMC with naive proposals, assign a large value for the epsilon parameter (eq. -E 10000.). \n";
        return 1;
    }
    if (var_map.count("randomize") > 0) {
        randomize = true;
    }
    if (var_map.count("merge") > 0) {
        merge = true;
    }
    if (var_map.count("seed") == 0) {
        // seeding based on the clock
        seed = (size_t) std::chrono::high_resolution_clock::now().time_since_epoch().count();
    }

    /* ~~~~~ Setup objects ~~~~~~~*/
    std::mt19937 engine(seed);
    uint_vec_t memberships_init;
    size_t N = 0;

    bool prepared = false;
    if (var_map.count("membership_path") != 0) {
        std::clog << "Loading nodes' membership from membership_path.\n";
        if (!load_memberships(memberships_init, membership_path)) {
            std::clog << "WARNING: error in loading memberships, read memberships from block sizes\n";
        } else {
            randomize = false;
            // -n and -z are not needed
            n.assign(memberships_init.size(), 0);
            for (auto const &b: memberships_init) ++n[b];
            // initiate z
            z.resize(2, 0);
            unsigned int max_n_ka = 0;
            unsigned int max_n_kb = 0;
            for (size_t mb_ = 0; mb_ < memberships_init.size(); ++mb_) {
                if (mb_ < y[0]) {
                    if (memberships_init[mb_] > max_n_ka) {
                        max_n_ka = memberships_init[mb_];
                    }
                }
                if (memberships_init[mb_] > max_n_kb) {
                    max_n_kb = memberships_init[mb_];
                }
            }
            z[0] = max_n_ka + 1;
            z[1] = max_n_kb - max_n_ka;
            n.resize(z[0] + z[1], 0);
            prepared = !prepared;
            N = memberships_init.size();
            std::clog << " ---- read membership from file! ---- \n";
        }
    } else if (var_map.count("mb") > 0) {
        // memberships from commandline input
        size_t accu = 0;
        for (auto const &it: n) accu += it;
        if (mb.size() != accu) {
            std::cerr << "[error] input vector size of memberships is different from the number of nodes \n";
            output_vec(mb, std::cerr);
            std::cerr << "#mb = " << mb.size() << "; while #nodes = " << accu << ". \n";
            return 1;
        }
        memberships_init = mb;
        mb.clear();
        if (var_map.count("bisbm_partition") == 0) {
            std::cerr << "number of partitions is required (-z flag)\n";
            return 1;
        } else {
            KA = z[0];
            KB = z[1];
        }
        N = memberships_init.size();
        prepared = !prepared;
    }

    if (!prepared) {
        // memberships from block sizes (the -n flag)
        if (var_map.count("n") == 0) {
            std::cerr << "n is required (-n flag) if one does not specify the membership of nodes\n";
            return 1;
        }
        size_t accu = 0;
        for (auto const &it: n) accu += it;
        memberships_init.resize(accu, 0);
        unsigned shift = 0;
        for (size_t r = 0; r < n.size(); ++r) {
            for (size_t i = 0; i < n[r]; ++i) {
                memberships_init[shift + i] = r;
            }
            shift += n[r];
        }
        if (var_map.count("bisbm_partition") == 0) {
            std::cerr << "number of partitions is required (-z flag)\n";
            return 1;
        } else {
            KA = z[0];
            KB = z[1];
        }
        N = memberships_init.size();
    }

    // sanity check for the "types"-vector
    if (memberships_init.size() != types_init.size()) {
        std::cerr << memberships_init.size() << ", " << types_init.size() << '\n';
        std::cerr << "Types do not sum to the number of vertices!" << "\n";
        return 1;
    }

    // Graph structure
    edge_list_t edge_list;
    load_edge_list(edge_list, edge_list_path);
    const adj_list_t adj_list = edge_to_adj(edge_list, N);
    edge_list.clear();

    // Bind proper Metropolis-Hasting algorithm
    std::unique_ptr<metropolis_hasting> algorithm;
    algorithm = std::make_unique<mh_tiago>();

    //blockmodel for the blocks
    if (merge) {
        std::iota(memberships_init.begin(), memberships_init.end(), 0);
        blockmodel_t blockmodel(memberships_init, types_init, NA + NB, NA, NB, epsilon, &adj_list);
        memberships_init.clear();
        types_init.clear();

        blockmodel.init_bisbm();

        int_vec_t ka_s;
        int_vec_t kb_s;
        std::tie(ka_s, kb_s) = geospace(NA, KA, NB, KB, 1.01);

        for (size_t i = 0; i < ka_s.size() - 1; ++i) {
            size_t diff_a = -(ka_s[i + 1] - ka_s[i]);
            size_t diff_b = -(kb_s[i + 1] - kb_s[i]);
            blockmodel.agg_merge(engine, diff_a, diff_b, 10);
            if (i != ka_s.size() - 2) {
                if (cooling_schedule == "abrupt_cool") {
                    algorithm->anneal(blockmodel, &abrupt_cool_schedule, cooling_schedule_kwargs, (NA + NB) * 1,
                                      steps_await, engine);
                } else {
                    std::cerr << "Only abrupt cooling annealing is supported.";
                    return 1;
                }
            }
        }
        algorithm->anneal(blockmodel, &abrupt_cool_schedule, cooling_schedule_kwargs, sampling_steps,
                          steps_await, engine);
        output_vec<uint_vec_t>(*blockmodel.get_memberships(), std::cout);
    } else {
        size_t ka{0};
        size_t kb{0};
        for (size_t t = 0; t < NA + NB; ++t) {
            if (types_init[t] == 0 && memberships_init[t] > ka) {
                ka = memberships_init[t];
            } else if (types_init[t] == 1 && memberships_init[t] > kb) {
                kb = memberships_init[t];
            }
        }
        kb -= ka;
        ka += 1;
        int diff_a = ka - KA;
        int diff_b = kb - KB;
        if (diff_a != 0 || diff_b != 0) {
            blockmodel_t blockmodel(memberships_init, types_init, ka + kb, ka, kb, epsilon, &adj_list);
            memberships_init.clear();
            types_init.clear();
            blockmodel.init_bisbm();
            if (diff_a >= 0 && diff_b >= 0) {
                int_vec_t ka_s;
                int_vec_t kb_s;
                std::tie(ka_s, kb_s) = geospace(KA + diff_a, KA, KB + diff_b, KB, 1.01);
                if (ka_s.size() == 1) {
                    blockmodel.agg_merge(engine, diff_a, diff_b, 10);
                }
                for (size_t i = 0; i < ka_s.size() - 1; ++i) {
                    diff_a = -(ka_s[i + 1] - ka_s[i]);
                    diff_b = -(kb_s[i + 1] - kb_s[i]);
                    blockmodel.agg_merge(engine, diff_a, diff_b, 10);
                    if (i != ka_s.size() - 2) {
                        if (cooling_schedule == "abrupt_cool") {
                            algorithm->anneal(blockmodel, &abrupt_cool_schedule, cooling_schedule_kwargs, (NA + NB) * 1,
                                              steps_await, engine);
                        } else {
                            std::cerr << "Only abrupt cooling annealing is supported.";
                            return 1;
                        }
                    }
                }
            } else {
                blockmodel.agg_merge(engine, diff_a, diff_b, 100);
            }
            algorithm->anneal(blockmodel, &abrupt_cool_schedule, cooling_schedule_kwargs, sampling_steps,
                              steps_await, engine);
            output_vec<uint_vec_t>(*blockmodel.get_memberships(), std::cout);
        } else {
            blockmodel_t blockmodel(memberships_init, types_init, KA + KB, KA, KB, epsilon, &adj_list);

            memberships_init.clear();
            types_init.clear();
            if (randomize) {
                blockmodel.shuffle_bisbm(engine, NA, NB);
            } else {
                blockmodel.init_bisbm();
            }
            double rate = 0;
            if (cooling_schedule == "exponential") {
                rate = algorithm->anneal(blockmodel, &exponential_schedule, cooling_schedule_kwargs, sampling_steps,
                                         steps_await, engine);
            }
            if (cooling_schedule == "linear") {
                rate = algorithm->anneal(blockmodel, &linear_schedule, cooling_schedule_kwargs, sampling_steps, steps_await,
                                         engine);
            }
            if (cooling_schedule == "logarithmic") {
                rate = algorithm->anneal(blockmodel, &logarithmic_schedule, cooling_schedule_kwargs, sampling_steps,
                                         steps_await, engine);
            }
            if (cooling_schedule == "constant") {
                rate = algorithm->anneal(blockmodel, &constant_schedule, cooling_schedule_kwargs, sampling_steps,
                                         steps_await, engine);
            }
            if (cooling_schedule == "abrupt_cool") {
                rate = algorithm->anneal(blockmodel, &abrupt_cool_schedule, cooling_schedule_kwargs, sampling_steps,
                                         steps_await, engine);
            }
            std::clog << "acceptance ratio " << rate << "\n";
            output_vec<uint_vec_t>(*blockmodel.get_memberships(), std::cout);
        }

    }

    return 0;
}
