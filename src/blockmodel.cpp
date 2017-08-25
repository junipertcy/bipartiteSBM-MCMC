#include <iostream>
#include "blockmodel.h"
#include "output_functions.h"  // TODO: debug use
#include "graph_utilities.h"  // for the is_disjoint function

using namespace std;


blockmodel_t::blockmodel_t(const uint_vec_t &memberships, const uint_vec_t &types, unsigned int g, unsigned int KA,
                           unsigned int KB, double epsilon, unsigned int N, adj_list_t *adj_list_ptr, bool is_bipartite) :
        random_block_(0, g - 1), random_node_(0, N - 1) {
    is_bipartite_ = is_bipartite;
    K_ = KA + KB;
    KA_ = KA;
    KB_ = KB;

    epsilon_ = epsilon;
    memberships_ = memberships;
    types_ = types;
    adj_list_ptr_ = adj_list_ptr;
    n_.resize(g, 0);
    deg_.resize(memberships.size(), 0);
    num_edges_ = 0;
    entropy_from_degree_correction_ = 0.;

    for (unsigned int j = 0; j < memberships.size(); ++j) {
        if (types_[j] == 0) {
            nsize_A_ += 1;
        } else if (types_[j] == 1) {
            nsize_B_ += 1;
        }

        ++n_[memberships[j]];

        for (auto nb = adj_list_ptr_->at(j).begin(); nb != adj_list_ptr_->at(j).end(); ++nb) {
            ++deg_[j];
            ++num_edges_;
        }
    }
    num_edges_ /= 2;

    double deg_factorial = 0;
    for (unsigned int node = 0; node < memberships.size(); ++node) {
        deg_factorial = 0;

        for (unsigned int deg = 1; deg <= deg_[node]; ++deg) {
            deg_factorial += std::log(deg);
        }
        entropy_from_degree_correction_ += deg_factorial;
    }

    compute_k();

    // Note that Tiago's MCMC proposal jumps has to randomly access elements in an adjacency list
    // Here, we define an vectorized data structure to make such data access O(1) [else it'll be O(n)].

    adj_list_.resize(adj_list_ptr_->size());
    for (auto i = 0; i < adj_list_ptr_->size(); ++i) {
        adj_list_[i].resize(adj_list_ptr_->at(i).size(), 0);
    }

    for (unsigned int node = 0; node < memberships_.size(); ++node) {
        unsigned int idx = 0;
        for (auto nb = adj_list_ptr_->at(node).begin(); nb != adj_list_ptr_->at(node).end(); ++nb) {
            adj_list_[node][idx] = *nb;
            ++idx;
        }
    }

}

bool blockmodel_t::change_KA(std::mt19937 &engine) noexcept {
    unsigned int r;
    // With probability 0.5, decrease KA, otherwise increase it
    if (random_real(engine) < 0.5) {
        // Count the number of empty groups
        unsigned int empty = 0;
        for (r = 0; r < KA_; ++r) {
            if (n_[r] == 0) {
                empty += 1;
            }
        }
        // If there are any empty groups, remove one of them, or otherwise do
        // nothing
        if (empty > 0) {
            // If there is more than one empty group, choose at random which one
            // to remove
            r = unsigned(int(random_real(engine) * KA_));
            while (n_[r] > 0) {
                r = unsigned(int(random_real(engine) * KA_));
            }
            if (r == KA_ - 1) {
                return false;
            }

            // Decrease KA by 1
            KA_ = KA_ - 1;

            // Update the group labels (both type-A and type-B nodes)
            for (auto node = 0; node < memberships_.size(); ++node) {
                if (types_[node] == 0) {
                    if (memberships_[node] == KA_) {
                        memberships_[node] = r;
                    }
                } else if (types_[node] == 1) {
                    memberships_[node] -= 1;
                }
            }

            bookkeeping_n_.clear();
            bookkeeping_n_.resize(KA_ + KB_, 0);

            for (unsigned int _g = 0; _g < KA_ + KB_; ++_g) {
                if (_g == r) {
                    bookkeeping_n_[_g] = n_[KA_];
                } else if (_g >= KA_) {  // type-B
                    bookkeeping_n_[_g] = n_[_g + 1];
                } else {
                    bookkeeping_n_[_g] = n_[_g];
                }
            }

            n_.clear();
            n_.resize(KA_ + KB_);
            n_ = bookkeeping_n_;
            compute_k();

            return true;
        }
    } else {

        // With probability k/(n+k) increase k by 1, adding an empty group
        if ((nsize_A_ + KA_) * random_real(engine) < KA_) {
            if (KA_ < GLOBAL_KA) {
                // Update the group labels (both type-A and type-B nodes)
                for (auto node = 0; node < memberships_.size(); ++node) {
                    if (types_[node] == 0) {
                        // Do nothing
                    } else if (types_[node] == 1) {
                        memberships_[node] += 1;
                    }
                }

                KA_ = KA_ + 1;

                // Update n_r
                bookkeeping_n_.clear();
                bookkeeping_n_.resize(KA_ + KB_, 0);
                for (unsigned int _g = 0; _g < KA_ + KB_; ++_g) {
                    if (_g == KA_ - 1) {
                        bookkeeping_n_[_g] = 0;
                    } else if (_g >= KA_) {
                        bookkeeping_n_[_g] = n_[_g - 1];
                    } else {
                        bookkeeping_n_[_g] = n_[_g];
                    }
                }

                n_.clear();
                n_.resize(KA_ + KB_);
                n_ = bookkeeping_n_;

                compute_k();
                return true;
            }
        }
    }
    return false;
}

bool blockmodel_t::change_KB(std::mt19937 &engine) noexcept {
    unsigned int r;
    // With probability 0.5, decrease KB, otherwise increase it
    if (random_real(engine) < 0.5) {

        // Count the number of empty groups
        unsigned int empty = 0;
        for (r = 0; r < KB_; ++r) {
            if (n_[r + KA_] == 0) {
                empty += 1;
            }
        }

        // If there are any empty groups, remove one of them, or otherwise do
        // nothing
        if (empty > 0) {
            // If there is more than one empty group, choose at random which one
            // to remove
            r = unsigned(int(random_real(engine) * KB_));
            while (n_[r + KA_] > 0) {
                r = unsigned(int(random_real(engine) * KB_));
            }
            if (r == KB_ - 1) {
                return false;
            }

            // Decrease KB by 1
            KB_ = KB_ - 1;

            // Update the group labels (both type-A and type-B nodes)
            for (auto node = 0; node < memberships_.size(); ++node) {
                if (types_[node] == 0) {
                    // KB_ to decrease; do nothing for type-A nodes
                } else if (types_[node] == 1) {
                    if (memberships_[node] == KB_ + KA_) {
                        memberships_[node] = r + KA_;
                    }

                }
            }

            // Update n_r
            bookkeeping_n_.clear();
            bookkeeping_n_.resize(KA_ + KB_, 0);
            for (unsigned int _g = 0; _g < KA_ + KB_; ++_g) {
                if (_g == r + KA_) {
                    bookkeeping_n_[_g] = n_[KB_ + KA_];
                } else if (_g >= KA_ + KB_) {
                    bookkeeping_n_[_g] = n_[_g + 1];
                } else {
                    bookkeeping_n_[_g] = n_[_g];
                }
            }

            n_.clear();
            n_.resize(KA_ + KB_);
            n_ = bookkeeping_n_;

            compute_k();

            // Update m_rs
            // No need; we calculate it on the fly during MCMC
            return true;
        }

    } else {
        // With probability k/(n+k) increase k by 1, adding an empty group
        if ((nsize_B_ + KB_) * random_real(engine) < KB_) {
            if (KB_ < GLOBAL_KB) {
                // Update the group labels (both type-A and type-B nodes)
                for (auto node = 0; node < memberships_.size(); ++node) {
                    if (types_[node] == 0) {
                        // Do nothing
                    } else if (types_[node] == 1) {
//                        memberships_[node] += 1;
                    }
                }

                KB_ = KB_ + 1;

                // Update n_r
                bookkeeping_n_.clear();
                bookkeeping_n_.resize(KA_ + KB_, 0);
                for (unsigned int _g = 0; _g < KA_ + KB_; ++_g) {
                    if (_g == KA_ + KB_ - 1) {
                        bookkeeping_n_[_g] = 0;
                    } else {
                        bookkeeping_n_[_g] = n_[_g];
                    }
                }
                n_.clear();
                n_.resize(KA_ + KB_);
                n_ = bookkeeping_n_;

                compute_k();
                return true;
            }
        }
    }
    return false;
}

double blockmodel_t::compute_entropy_from_m_mr(uint_mat_t m, uint_vec_t m_r) const noexcept {
    int_vec_t n = get_size_vector();

    double entropy = -(double) num_edges_ - entropy_from_degree_correction_;

    for (unsigned r_ = 0; r_ < n.size(); ++r_) {
        for (unsigned s_ = 0; s_ < n.size(); ++s_) {
            if (m_r[r_] * m_r[s_] * m[r_][s_] != 0) {
                entropy -= 1. / 2. * (double) m[r_][s_] *
                           std::log((double) m[r_][s_] / (double) m_r[r_] / (double) m_r[s_]);
            }
        }
    }
    return entropy;
}

std::vector<mcmc_state_t> blockmodel_t::mcmc_state_change_riolo_uni1(std::mt19937 &engine) {
    std::vector<mcmc_state_t> states(1);
    bool cond = true;

    while (cond) {
        states[0].memberships.resize(memberships_.size(), 0);
        for (auto i = 0; i < memberships_.size(); ++i) {
            states[0].memberships[i] = memberships_[i];
        }

        // decide whether to update type-a nodes or type-b nodes
        double num_nodes = (double) states[0].memberships.size();
        if (random_real(engine) < 1. / (num_nodes - 1)) {  // type-II move
            //        std::clog << "1\n";
            unsigned int r = unsigned(int(random_real(engine) * (K_ + 1)));
            unsigned int s = r;
            while (s == r) {
                s = unsigned(int(random_real(engine) * (K_ + 1)));
            }

            if (r != K_) {  // re-labeling, else no re-labeling is necessary!
                for (auto node = 0; node < states[0].memberships.size(); ++node) {
                    if (states[0].memberships[node] == r) {
                        states[0].memberships[node] = K_;
                    }
                }
            }

            unsigned int counter = 0;
            unsigned int rnd_node_in_label_s;
            bool last_node_in_s = false;
            if (s != K_) {
                rnd_node_in_label_s = (unsigned) (int) (random_real(engine) * n_[s]);
                if (n_[s] == 1) {
                    last_node_in_s = true;
                }
            } else {
                rnd_node_in_label_s = (unsigned) (int) (random_real(engine) * n_[r]);
                if (n_[r] == 1) {
                    states[0].memberships = memberships_;
                    return states;
                }
            }

            for (unsigned int i = 0; i < memberships_.size(); ++i) {
                if (states[0].memberships[i] == s) {
                    if (counter == rnd_node_in_label_s) {
                        states[0].memberships[i] = r;
                    }
                    counter++;
                }
            }
            // check if we have removed the last remaining node of label s;
            if (last_node_in_s) {
                for (auto node = 0; node < states[0].memberships.size(); ++node) {
                    if (states[0].memberships[node] == K_) {
                        states[0].memberships[node] = s;
                    }
                }
            }
            if (!debugger(states[0].memberships)) {

                throw 1;
            }
            //        std::clog << "2\n";
        } else {  // type-I move
            if (K_ == 1) {
                // we do nothing
                states[0].memberships = memberships_;
                return states;
            } else {
                //            std::clog << "3\n";
                unsigned int r = unsigned(int(random_real(engine) * K_));
                unsigned int s = r;
                while (s == r) {
                    s = unsigned(int(random_real(engine) * K_));
                }

                unsigned int counter = 0;
                unsigned int rnd_node_in_label_r = (unsigned) (int) (random_real(engine) * n_[r]);

                for (unsigned int i = 0; i < memberships_.size(); ++i) {
                    if (memberships_[i] == r) {
                        if (counter == rnd_node_in_label_r) {
                            states[0].memberships[i] = s;
                        }
                        counter++;
                    }
                }

                // check if we have removed the last remaining node of label r;
                if (n_[r] == 1) {
                    for (unsigned int i = 0; i < memberships_.size(); ++i) {
                        if (states[0].memberships[i] == K_ - 1) {
                            states[0].memberships[i] = r;
                        }
                    }
                }
            }
            if (!debugger(states[0].memberships)) {
                throw 1;
            }
        }

        cond = !cond;

    }

    return states;
}

std::vector<mcmc_state_t> blockmodel_t::mcmc_state_change_riolo_uni2(std::mt19937 &engine) {
    std::vector<mcmc_state_t> states(1);
    bool cond = true;

    while (cond) {
        states[0].memberships.resize(memberships_.size(), 0);
        for (auto i = 0; i < memberships_.size(); ++i) {
            states[0].memberships[i] = memberships_[i];
        }

        // decide whether to update type-a nodes or type-b nodes
        double num_nodes = (double) states[0].memberships.size();
        if (random_real(engine) < 1. / (num_nodes - 1)) {  // type-II move
            //        std::clog << "1\n";
            unsigned int r = unsigned(int(random_real(engine) * (K_ + 1)));
            unsigned int s = r;
            while (s == r) {
                s = unsigned(int(random_real(engine) * (K_ + 1)));
            }

            if (r != K_) {  // re-labeling, else no re-labeling is necessary!
                for (auto node = 0; node < states[0].memberships.size(); ++node) {
                    if (states[0].memberships[node] == r) {
                        states[0].memberships[node] = K_;
                    }
                }
            }

            unsigned int counter = 0;
            unsigned int rnd_node_in_label_s;
            bool last_node_in_s = false;
            if (s != K_) {
                rnd_node_in_label_s = (unsigned) (int) (random_real(engine) * n_[s]);
                if (n_[s] == 1) {
                    last_node_in_s = true;
                }
            } else {
                rnd_node_in_label_s = (unsigned) (int) (random_real(engine) * n_[r]);
                if (n_[r] == 1) {
                    states[0].memberships = memberships_;
                    return states;
                }
            }

            for (unsigned int i = 0; i < memberships_.size(); ++i) {
                if (states[0].memberships[i] == s) {
                    if (counter == rnd_node_in_label_s) {
                        states[0].memberships[i] = r;
                    }
                    counter++;
                }
            }
            // check if we have removed the last remaining node of label s;
            if (last_node_in_s) {
                for (auto node = 0; node < states[0].memberships.size(); ++node) {
                    if (states[0].memberships[node] == K_) {
                        states[0].memberships[node] = s;
                    }
                }
            }
            if (!debugger(states[0].memberships)) {

                throw 1;
            }
            //        std::clog << "2\n";
        } else {  // type-I move
            if (K_ == 1) {
                // we do nothing
                states[0].memberships = memberships_;
                return states;
            } else {
                //            std::clog << "3\n";
                unsigned int r = unsigned(int(random_real(engine) * K_));
                unsigned int s = r;
                while (s == r) {
                    s = unsigned(int(random_real(engine) * K_));
                }

                unsigned int counter = 0;
                unsigned int rnd_node_in_label_r = (unsigned) (int) (random_real(engine) * n_[r]);

                for (unsigned int i = 0; i < memberships_.size(); ++i) {
                    if (memberships_[i] == r) {
                        if (counter == rnd_node_in_label_r) {
                            states[0].memberships[i] = s;
                        }
                        counter++;
                    }
                }

                // check if we have removed the last remaining node of label r;
                if (n_[r] == 1) {
                    for (unsigned int i = 0; i < memberships_.size(); ++i) {
                        if (states[0].memberships[i] == K_ - 1) {
                            states[0].memberships[i] = r;
                        }
                    }
                }
            }
            if (!debugger(states[0].memberships)) {
                throw 1;
            }
        }

        set<unsigned int> type_a_labels;
        set<unsigned int> type_b_labels;

        for (auto i = 0; i < states[0].memberships.size(); ++i) {
            if (types_[i] == 0) {
                type_a_labels.insert(states[0].memberships[i]);
            } else {
                type_b_labels.insert(states[0].memberships[i]);
            }
        }
        cond = !is_disjoint(type_a_labels, type_b_labels);

    }

    return states;
}

std::vector<mcmc_state_t> blockmodel_t::mcmc_state_change_riolo(std::mt19937 &engine) {
    // This function returns a move, which will be accepted or rejected by Hasting's rule
    // if it is updated AND it increases K_a or K_b, then we re-label the nodes outside of this function

    std::vector<mcmc_state_t> states(1);

    states[0].memberships.resize(memberships_.size(), 0);
    for (auto i = 0; i < memberships_.size(); ++i) {
        states[0].memberships[i] = memberships_[i];
    }

    // states[0].memberships = memberships_;

    // decide whether to update type-a nodes or type-b nodes
    auto num_nodes = (double) states[0].memberships.size();
    auto num_nodes_a = (double) nsize_A_;
    auto num_nodes_b = (double) nsize_B_;

    if (random_real(engine) < num_nodes_a / num_nodes) {
//        if (random_real(engine) < 1.) {

            if (random_real(engine) < 1. / (num_nodes - 1)) {
            auto r = unsigned(int(random_real(engine) * (KA_ + 1)));
            unsigned int s = r;
            while (s == r) {
                s = unsigned(int(random_real(engine) * (KA_ + 1)));
            }

            if (r != KA_) {  // re-labeling, else no re-labeling is necessary!
                for (auto node = 0; node < states[0].memberships.size(); ++node) {
                    if (types_[node] == 0 && states[0].memberships[node] == r) {
                        states[0].memberships[node] = KA_;
                    } else if (types_[node] == 1) {
                        ++states[0].memberships[node];
                    }
                }
            } else {
                for (auto node = 0; node < states[0].memberships.size(); ++node) {
                    if (types_[node] == 1) {
                        ++states[0].memberships[node];
                    }
                }
            }

            unsigned int counter = 0;
            unsigned int rnd_node_in_label_s;
            bool last_node_in_s = false;
            if (s != KA_) {
                rnd_node_in_label_s = (unsigned) (int) (random_real(engine) * n_[s]);
                if (n_[s] == 1) {
                    last_node_in_s = true;
                }
            } else {
                rnd_node_in_label_s = (unsigned) (int) (random_real(engine) * n_[r]);
                if (n_[r] == 1) {
                    states[0].memberships = memberships_;
                    return states;
                }
            }

            for (unsigned int i = 0; i < memberships_.size(); ++i) {
                if (types_[i] == 0 && states[0].memberships[i] == s) {
                    if (counter == rnd_node_in_label_s) {
                        states[0].memberships[i] = r;
                    }
                    counter++;
                }
            }

            // check if we have removed the last remaining node of label s;
            if (last_node_in_s) {
                for (auto node = 0; node < states[0].memberships.size(); ++node) {
                    if (types_[node] == 0 && states[0].memberships[node] == KA_) {
                        states[0].memberships[node] = s;
                    } else if (types_[node] == 1) {
                        --states[0].memberships[node];
                    }
                }
            }
        } else {  // type-I move for type-a nodes
            if (KA_ == 1) {
                // we do nothing
                states[0].memberships = memberships_;
                return states;
            } else {
                unsigned int r = unsigned(int(random_real(engine) * KA_));
                unsigned int s = r;
                while (s == r) {
                    s = unsigned(int(random_real(engine) * KA_));
                }

                unsigned int counter = 0;
                auto rnd_node_in_label_r = (unsigned) (int) (random_real(engine) * n_[r]);

                for (unsigned int i = 0; i < memberships_.size(); ++i) {
                    if (types_[i] == 0 && memberships_[i] == r) {
                        if (counter == rnd_node_in_label_r) {
                            states[0].memberships[i] = s;
                        }
                        counter++;
                    }
                }

                // check if we have removed the last remaining node of label r;
                if (n_[r] == 1) {
                    for (unsigned int i = 0; i < memberships_.size(); ++i) {
                        if (types_[i] == 0 && states[0].memberships[i] == KA_ - 1) {
                            states[0].memberships[i] = r;
                        } else if (types_[i] == 1) {
                            --states[0].memberships[i];
                        }
                    }
                }
            }
        }
    } else {
        if (random_real(engine) < 1. / (num_nodes_b - 1)) {  // type-II move for type-b nodes
            auto r = unsigned(int(random_real(engine) * (KB_ + 1)));
            unsigned int s = r;
            while (s == r) {
                s = unsigned(int(random_real(engine) * (KB_ + 1)));
            }

            if (r != KB_) {  // re-labeling, else no re-labeling is necessary!
                for (auto node = 0; node < states[0].memberships.size(); ++node) {
                    if (types_[node] == 1 && states[0].memberships[node] == r + KA_) {
                        states[0].memberships[node] = KB_ + KA_;
                    } else if (types_[node] == 0) {
                        // do nothing;
                    }
                }
            }

            // Now, move a node in s to empty group r;
            unsigned int counter = 0;
            unsigned int rnd_node_in_label_s;
            bool last_node_in_s = false;

            if (s != KB_) {
                rnd_node_in_label_s = (unsigned) (int) (random_real(engine) * n_[s + KA_]);
                if (n_[s + KA_] == 1) {
                    last_node_in_s = true;
                }
            } else {
                rnd_node_in_label_s = (unsigned) (int) (random_real(engine) * n_[r + KA_]);
                if (n_[r + KA_] == 1) { // if s == KB_ + 1, and n[s] = 1, we move the node back to r: zero-sum game.
                    states[0].memberships = memberships_;
                    return states;
                }
            }

            for (unsigned int i = 0; i < states[0].memberships.size(); ++i) {
                if (types_[i] == 1 && states[0].memberships[i] == s + KA_) {
                    if (counter == rnd_node_in_label_s) {
                        states[0].memberships[i] = r + KA_;
                    }
                    counter++;
                }
            }

            // check if we have removed the last remaining node of label s;
            if (last_node_in_s) {
                for (auto node = 0; node < states[0].memberships.size(); ++node) {
                    if (types_[node] == 1 && states[0].memberships[node] == KB_ + KA_) {
                        states[0].memberships[node] = s + KA_;
                    } else if (types_[node] == 0) {
                        // do nothing;
                    }
                }
            }
        } else {  // type-I move for type-b nodes
            if (KB_ == 1) {
                states[0].memberships = memberships_;  // we do nothing
                return states;
            } else {
                auto r = unsigned(int(random_real(engine) * KB_));
                unsigned int s = r;
                while (s == r) {
                    s = unsigned(int(random_real(engine) * KB_));
                }

                unsigned int counter = 0;
                auto rnd_node_in_label_r = (unsigned) (int) (random_real(engine) * n_[r + KA_]);
                for (unsigned int i = 0; i < states[0].memberships.size(); ++i) {
                    if (types_[i] == 1 && memberships_[i] == r + KA_) {
                        if (counter == rnd_node_in_label_r) {
                            states[0].memberships[i] = s + KA_;
                        }
                        counter++;
                    }
                }

                // check if we have removed the last remaining node of label r;
                if (n_[r + KA_] == 1) {
                    for (unsigned int i = 0; i < memberships_.size(); ++i) {
                        if (types_[i] == 1 && states[0].memberships[i] == KA_ + KB_ - 1) {
                            states[0].memberships[i] = r + KA_;
                        } else if (types_[i] == 0) {
                            // do nothing;
                        }
                    }
                }
            }
        }
    }
    return states;
}

bool blockmodel_t::debugger(uint_vec_t mb) noexcept {
    // TODO: delete -- SANITY check
    // check the proposed memberships_ is correct, and it produces correct n_

    unsigned int max = 0;
    for (unsigned int i = 0; i < mb.size(); ++i) {
        if (mb[i] > max) {
            max = mb[i];
        }
    }

    uint_vec_t nn_;
    nn_.resize(max + 1, 0);

    for (unsigned int i = 0; i < mb.size(); ++i) {
        ++nn_[mb[i]];
    }

    for (auto i = 0; i < nn_.size(); ++i) {
        if (nn_[i] == 0) {
            output_vec<uint_vec_t>(mb, std::clog);
            return false;
        }
    }
    return true;
}

std::vector<mcmc_state_t> blockmodel_t::single_vertex_change_heat_bath(std::mt19937 &engine) noexcept {
    std::vector<mcmc_state_t> moves(1);

    moves[0].vertex = unsigned(random_node_(engine));
    moves[0].source = memberships_[moves[0].vertex];
    moves[0].target = moves[0].source;

    // heat bath
    uint_mat_t m = get_m();
    uint_mat_t m_part = m;
    int_vec_t n = get_size_vector();
    unsigned int KA = get_KA();
    unsigned int KB = get_KB();
    double prob_ = 2. * get_num_edges() / (double) get_nsize_A() / (double) get_nsize_B();
    float_vec_t log_prob;
    float_vec_t prob;
    float_vec_t cum_prob;

    // for the degree-correction
    int_vec_t deg = get_degree();
    uint_vec_t sum_of_deg_in_block;
    sum_of_deg_in_block.resize(KA + KB, 0);
    for (unsigned int node = 0; node < memberships_.size(); ++node) {
        sum_of_deg_in_block[memberships_[node]] += deg[node];
    }

    double log_factorial = 0.;
    if (types_[moves[0].vertex] == 0) {
        log_prob.resize(KA);
        prob.resize(KA);
        cum_prob.resize(KA);

        for (unsigned int i = 0; i < log_prob.size(); ++i) {
            log_prob[i] = (float) 0.;
        }

        for (unsigned g = 0; g < KA; ++g) {
            n = get_size_vector();
            n[moves[0].source] -= 1;
            n[g] += 1;

            int_vec_t ki = get_k(moves[0].vertex);
            m_part = get_m();
            for (unsigned int l = 0; l < n.size(); ++l) {
                if (l != moves[0].source && l != g) {
                    m_part[moves[0].source][l] -= ki[l];
                    m_part[l][moves[0].source] = m_part[moves[0].source][l];

                    m_part[g][l] += ki[l];
                    m_part[l][g] = m_part[g][l];
                }
            }

            // degree-corrected model
            uint_vec_t new_sum_of_deg_in_block = sum_of_deg_in_block;
            new_sum_of_deg_in_block[memberships_[moves[0].vertex]] -= deg[moves[0].vertex];
            new_sum_of_deg_in_block[g] += deg[moves[0].vertex];
            for (unsigned int q = 0; q < KA + KB; ++q) {
                if (n[q] != 0) {  // degree-corrected model
                    log_prob[g] += new_sum_of_deg_in_block[q] * std::log(n[q]);
                    log_factorial = 0.;
                    for (int m = n[q]; m < n[q] + new_sum_of_deg_in_block[q]; ++m) {
                        log_factorial += std::log(1. / (double) m);
                    }
                    log_prob[g] += log_factorial;
                }
            }
            new_sum_of_deg_in_block.clear();  // TODO: this may not be needed.
            for (unsigned int _g = 0; _g < KA; ++_g) {
                log_factorial = 0.;
                for (unsigned int i = 1; i <= n[_g]; ++i) {
                    log_factorial += std::log(i);
                }
                log_prob[g] += log_factorial;
                log_prob[g] -= std::log(0.5 * prob_ * n[_g] * n[_g] + 1.);

                for (unsigned int _gg = 0; _gg < KB; ++_gg) {
                    log_factorial = 0.;
                    for (unsigned int i = 1; i <= m_part[_g][_gg + KA]; ++i) {
                        log_factorial += std::log(i);
                    }
                    log_prob[g] += log_factorial;
                    log_prob[g] -= (m_part[_g][_gg + KA] + 1) * std::log(prob_ * n[_g] * n[_gg + KA] + 1.);
                }
            }
            n.clear();
            m_part.clear();
        }

        float max = -1000000;
        for (unsigned int d = 0; d < log_prob.size(); ++d) {
            if (log_prob[d] > max) {
                max = log_prob[d];
            }
        }

        float total = 0;
        for (unsigned int d = 0; d < log_prob.size(); ++d) {
            log_prob[d] = log_prob[d] - max;
            total += std::exp(log_prob[d]);
        }

        for (unsigned int d = 0; d < log_prob.size(); ++d) {
            prob[d] = std::exp(log_prob[d]) / total;
            if (d != 0) {
                cum_prob[d] = prob[d] + cum_prob[d - 1];
            } else {
                cum_prob[d] = prob[d];
            }

        }
    } else if (types_[moves[0].vertex] == 1) {
        log_prob.resize(KB);
        prob.resize(KB);
        cum_prob.resize(KB);

        for (unsigned int i = 0; i < log_prob.size(); ++i) {
            log_prob[i] = (float) 0.;
        }

        for (unsigned g = 0; g < KB; ++g) {
            n = get_size_vector();
            n[moves[0].source] -= 1;
            n[g + KA] += 1;

            int_vec_t ki = get_k(moves[0].vertex);
            m_part = get_m();
            for (unsigned int l = 0; l < n.size(); ++l) {
                if (l != moves[0].source && l != g + KA) {
                    m_part[moves[0].source][l] -= ki[l];
                    m_part[l][moves[0].source] = m_part[moves[0].source][l];

                    m_part[g + KA][l] += ki[l];
                    m_part[l][g + KA] = m_part[g + KA][l];
                }
            }

            // degree-corrected model
            uint_vec_t new_sum_of_deg_in_block = sum_of_deg_in_block;
            new_sum_of_deg_in_block[memberships_[moves[0].vertex]] -= deg[moves[0].vertex];
            new_sum_of_deg_in_block[g + KA] += deg[moves[0].vertex];

            for (unsigned int q = 0; q < KA + KB; ++q) {
                if (n[q] != 0) {  // degree-corrected model
                    log_prob[g] += new_sum_of_deg_in_block[q] * std::log(n[q]);
                    log_factorial = 0.;
                    for (int m = n[q]; m < n[q] + new_sum_of_deg_in_block[q]; ++m) {
                        log_factorial += std::log(1. / (double) m);
                    }
                    log_prob[g] += log_factorial;
                }
            }
            new_sum_of_deg_in_block.clear(); // TODO: this may not be needed

            for (unsigned int _g = 0; _g < KB; ++_g) {
                log_factorial = 0.;
                for (unsigned int i = 1; i <= n[_g + KA]; ++i) {
                    log_factorial += std::log(i);
                }
                log_prob[g] += log_factorial;
                log_prob[g] -= std::log(0.5 * prob_ * n[_g + KA] * n[_g + KA] + 1.);

                for (unsigned int _gg = 0; _gg < KA; ++_gg) {
                    log_factorial = 0.;
                    for (unsigned int i = 1; i <= m_part[_g + KA][_gg]; ++i) {
                        log_factorial += std::log(i);
                    }
                    log_prob[g] += log_factorial;
                    log_prob[g] -= (m_part[_g + KA][_gg] + 1) * std::log(prob_ * n[_g + KA] * n[_gg] + 1.);
                }
            }
            n.clear();
            m_part.clear();
        }

        float max = -1000000;
        for (unsigned int d = 0; d < log_prob.size(); ++d) {
            if (log_prob[d] > max) {
                max = log_prob[d];
            }
        }

        float total = 0;
        for (unsigned int d = 0; d < log_prob.size(); ++d) {
            log_prob[d] = log_prob[d] - max;
            total += std::exp(log_prob[d]);
        }

        for (unsigned int d = 0; d < log_prob.size(); ++d) {
            prob[d] = std::exp(log_prob[d]) / total;

            if (d != 0) {
                cum_prob[d] = prob[d] + cum_prob[d - 1];
            } else {
                cum_prob[d] = prob[d];
            }

        }
    }

    double rnd = random_real(engine);
    unsigned int ind = 0;
    if (types_[moves[0].vertex] == 0) {
        while (rnd > cum_prob[ind]) {
            ind += 1;
        }
        moves[0].target = ind;
    } else if (types_[moves[0].vertex] == 1) {
        while (rnd > cum_prob[ind]) {
            ind += 1;
        }
        moves[0].target = ind + KA;
    }

    log_prob.clear();
    prob.clear();
    cum_prob.clear();
    return moves;
}

std::vector<mcmc_state_t> blockmodel_t::single_vertex_change_naive(std::mt19937 &engine) noexcept {
    std::vector<mcmc_state_t> moves(1);

    moves[0].vertex = unsigned(random_node_(engine));
    moves[0].source = memberships_[moves[0].vertex];
    moves[0].target = moves[0].source;

    // naive method
    if (types_[moves[0].vertex] == 0) {
        moves[0].target = unsigned(int(random_real(engine) * KA_));
    } else if (types_[moves[0].vertex] == 1) {
        moves[0].target = unsigned(int(random_real(engine) * KB_)) + KA_;
    }
    return moves;
}

std::vector<mcmc_state_t> blockmodel_t::single_vertex_change_tiago(std::mt19937 &engine) noexcept {

    double epsilon = epsilon_;
    double R_t = 0.;
    unsigned int vertex_j;
    unsigned int proposal_t;
    uint_mat_t m = get_m();
    uint_vec_t m_r = get_m_r();
    std::vector<mcmc_state_t> moves(1);
    int proposal_membership = 0;

    moves[0].vertex = unsigned(random_node_(engine));
    moves[0].source = memberships_[moves[0].vertex];
    moves[0].target = moves[0].source;

    while (moves[0].source == moves[0].target) {

        // Here, instead of naively move to adjacent blocks, we follow Tiago Peixoto's approach (PRE 89, 012804 [2014])
        auto which_to_move = (int) (random_real(engine) * adj_list_[moves[0].vertex].size());
        vertex_j = adj_list_[moves[0].vertex][which_to_move];
        proposal_t = memberships_[vertex_j];
        if (types_[moves[0].vertex] == 0) {
            proposal_membership = int(random_real(engine) * KA_);
            R_t = epsilon * (KA_) / (m_r[proposal_t] + epsilon * (KA_));
        } else if (types_[moves[0].vertex] == 1) {
            proposal_membership = int(random_real(engine) * KB_) + KA_;
            R_t = epsilon * (KB_) / (m_r[proposal_t] + epsilon * (KB_));
        }
        if (random_real(engine) < R_t) {
            moves[0].target = unsigned(proposal_membership);
        } else {
            // TODO: May exist a better way of writing...
            which_to_move = (int) (random_real(engine) * m_r[proposal_t]);
            unsigned int counter = 0;
            for (auto s = 0; s < m_r.size(); ++s) {
                counter += m[proposal_t][s];
                if (counter > which_to_move) {
                    moves[0].target = unsigned(s);
                    break;
                }
            }
        }
    }

//    moves[0].vertex = 0;
//    moves[0].source = 0;
//    moves[0].target = 1;


    return moves;
}

int_vec_t blockmodel_t::get_k(unsigned int vertex) const noexcept { return k_[vertex]; }

int_vec_t blockmodel_t::get_size_vector() const noexcept { return n_; }

int_vec_t blockmodel_t::get_degree() const noexcept { return deg_; }

uint_vec_t blockmodel_t::get_memberships() const noexcept { return memberships_; }

uint_vec_t blockmodel_t::get_types() const noexcept { return types_; }


double blockmodel_t::get_epsilon() const noexcept { return epsilon_; }

uint_mat_t blockmodel_t::get_m() const noexcept {
    uint_mat_t m(get_g(), uint_vec_t(get_g(), 0));
    for (unsigned int vertex = 0; vertex < adj_list_ptr_->size(); ++vertex) {
        for (auto nb = adj_list_ptr_->at(vertex).begin(); nb != adj_list_ptr_->at(vertex).end(); ++nb) {
            ++m[memberships_[vertex]][memberships_[*nb]];
        }
    }
    for (unsigned int r = 0; r < get_g(); ++r) {
        for (unsigned int s = 0; s < get_g(); ++s) {
            m[r][s] /= 2;  // edges are counted twice (the adj_list is symmetric)
            m[r][s] = m[s][r];  // symmetrize m matrix.
        }
    }
    return m;
}

uint_mat_t blockmodel_t::get_m_from_membership(uint_vec_t mb) const noexcept {

    // Note that in Riolo's setting, we have to compare two jump choices of different sizes;
    // For the newly proposed system with matrix m, we have to calculate its size every time here;
    unsigned int cand_n_ = get_n_from_mb(mb);

    uint_mat_t m(cand_n_, uint_vec_t(cand_n_, 0));

    for (unsigned int vertex = 0; vertex < adj_list_ptr_->size(); ++vertex) {
        for (auto nb = adj_list_ptr_->at(vertex).begin(); nb != adj_list_ptr_->at(vertex).end(); ++nb) {
            ++m[mb[vertex]][mb[*nb]];
        }
    }
    for (unsigned int r = 0; r < cand_n_; ++r) {
        for (unsigned int s = 0; s < cand_n_; ++s) {
            m[r][s] /= 2;  // edges are counted twice (the adj_list is symmetric)
            m[r][s] = m[s][r];  // symmetrize m matrix.
        }
    }
    return m;
}

uint_vec_t blockmodel_t::get_m_r() const noexcept {
    uint_mat_t m = get_m();
    uint_vec_t m_r(get_g(), 0);
    unsigned int m_r_ = 0;
    for (unsigned int r = 0; r < get_g(); ++r) {
        m_r_ = 0;
        for (unsigned int s = 0; s < get_g(); ++s) {
            m_r_ += m[r][s];
        }
        m_r[r] = m_r_;
    }
    return m_r;
}

uint_vec_t blockmodel_t::get_m_r_from_m(uint_mat_t m) const noexcept {
    uint_vec_t m_r(m.size(), 0);
    unsigned int m_r_ = 0;
    for (unsigned int r = 0; r < m.size(); ++r) {
        m_r_ = 0;
        for (unsigned int s = 0; s < m.size(); ++s) {
            m_r_ += m[r][s];
        }
        m_r[r] = m_r_;
    }
    return m_r;
}


unsigned int blockmodel_t::get_n_from_mb(uint_vec_t mb) const noexcept {
    if (is_bipartite_) {
        unsigned int cand_KA_ = 0;
        unsigned int cand_KB_ = 0;
        for (auto i = 0; i < mb.size(); ++i) {
            if (types_[i] == 0) {
                if (mb[i] > cand_KA_) {
                    cand_KA_ = mb[i];
                }
            } else {
                if (mb[i] > cand_KB_) {
                    cand_KB_ = mb[i];
                }
            }
        }
        unsigned int KA_ = cand_KA_ + 1;
        unsigned int KB_ = cand_KB_ + 1 - KA_;
        unsigned int cand_n_ = KA_ + KB_;

        return cand_n_;
    } else {
        unsigned int cand_K_ = 0;
        for (auto i = 0; i < mb.size(); ++i) {
            if (mb[i] > cand_K_) {
                cand_K_ = mb[i];
            }
        }
        unsigned int K_ = cand_K_ + 1;

        return K_;
    }

}

int_vec_t blockmodel_t::get_n_r_from_mb(uint_vec_t mb) const noexcept {
    unsigned int cand_n_ = get_n_from_mb(mb);

    int_vec_t n_r_;
    n_r_.clear();
    n_r_.resize(cand_n_, 0);

    for (unsigned int j = 0; j < mb.size(); ++j) {
        ++n_r_[mb[j]];
    }
    return n_r_;
}

bool blockmodel_t::get_is_bipartite() const noexcept { return is_bipartite_; }

unsigned int blockmodel_t::get_N() const noexcept { return unsigned(int(memberships_.size())); }

unsigned int blockmodel_t::get_g() const noexcept { return unsigned(int(n_.size())); }

unsigned int blockmodel_t::get_num_edges() const noexcept { return num_edges_; }

double blockmodel_t::get_entropy_from_degree_correction() const noexcept { return entropy_from_degree_correction_; }

unsigned int blockmodel_t::get_K() const noexcept { return K_; }

unsigned int blockmodel_t::get_KA() const noexcept { return KA_; }

unsigned int blockmodel_t::get_nsize_A() const noexcept { return nsize_A_; }

unsigned int blockmodel_t::get_KB() const noexcept { return KB_; }

unsigned int blockmodel_t::get_nsize_B() const noexcept { return nsize_B_; }

void blockmodel_t::apply_mcmc_moves(std::vector<mcmc_state_t> moves) noexcept {
    for (unsigned int i = 0; i < moves.size(); ++i) {

        // Change block degrees and block sizes
        for (auto neighbour = adj_list_ptr_->at(moves[i].vertex).begin();
             neighbour != adj_list_ptr_->at(moves[i].vertex).end();
             ++neighbour) {
            --k_[*neighbour][moves[i].source];
            ++k_[*neighbour][moves[i].target];
        }

        if (n_.size() != KA_ + KB_) {
            std::cerr << "sanity check failed: n_.size() != KA_ + KB_\n";
            std::cerr << "n_.size() = " << n_.size() << ",KA_ = " << KA_ << ",KB = " << KB_ << "\n";

        }

        --n_[moves[i].source];
        ++n_[moves[i].target];

        // Set new memberships
        memberships_[moves[i].vertex] = moves[i].target;
    }
}

void blockmodel_t::apply_mcmc_states_u(std::vector<mcmc_state_t> states) noexcept {
    // Key things to do here:
    // 1. update memberships_, n_;
    // 2. update KA_, KB_;

    for (unsigned int i = 0; i < states.size(); ++i) {
        for (unsigned int i = 0; i < memberships_.size(); ++i) {
            memberships_[i] = states[0].memberships[i];
        }
        unsigned int max_K_ = 0;
        for (auto i = 0; i < memberships_.size(); ++i) {
            if (memberships_[i] > max_K_) {
                max_K_ = memberships_[i]; // key point; keep membership compact;
            }
        }
        K_ = max_K_ + 1;
        n_.resize(K_, 0);
        for (unsigned int j = 0; j < n_.size(); ++j) {
            n_[j] = 0;
        }

        for (unsigned int j = 0; j < memberships_.size(); ++j) {
            ++n_[memberships_[j]];
        }

        if (n_.size() != K_) {
            std::cerr << "sanity check failed: n_.size() != K_\n";
            std::cerr << "n_.size() = " << n_.size() << ", K_ = " << K_ << "\n";
        }
    }

    // TODO: delete -- SANITY check
    unsigned int total_num = 0;
    for (unsigned int i = 0; i < n_.size(); ++i) {
        if (n_[i] == 0) {
            std::clog << "n_[i] cannot be zero!\n";
        }
        total_num += n_[i];
    }
    if (total_num != memberships_.size()) {
        std::clog << "total_num != memberships_.size(); n_ is already wrong here!! \n";
    }
}


void blockmodel_t::apply_mcmc_states(std::vector<mcmc_state_t> states) noexcept {
    // Key things to do here:
    // 1. update memberships_, n_;
    // 2. update KA_, KB_;

    for (unsigned int i = 0; i < states.size(); ++i) {
        for (unsigned int i = 0; i < memberships_.size(); ++i) {
            memberships_[i] = states[0].memberships[i];
        }
        unsigned int max_KA_ = 0;
        unsigned int max_KB_ = 0;
        for (auto i = 0; i < memberships_.size(); ++i) {
            if (types_[i] == 0) {
                if (memberships_[i] > max_KA_) {
                    max_KA_ = memberships_[i]; // key point; keep membership compact;
                }
            } else {
                if (memberships_[i] > max_KB_) {
                    max_KB_ = memberships_[i];
                }
            }
        }
        KA_ = max_KA_ + 1;
        KB_ = max_KB_ + 1 - KA_;
        K_ = KA_ + KB_;

        n_.resize(KA_ + KB_, 0);
        for (unsigned int j = 0; j < n_.size(); ++j) {
            n_[j] = 0;
        }
        for (unsigned int j = 0; j < memberships_.size(); ++j) {
            ++n_[memberships_[j]];
        }

        if (n_.size() != KA_ + KB_) {
            std::cerr << "sanity check failed: n_.size() != KA_ + KB_\n";
            std::cerr << "n_.size() = " << n_.size() << ",KA_ = " << KA_ << ",KB = " << KB_ << "\n";
        }
    }

    // TODO: delete -- SANITY check
    unsigned int total_num = 0;
    for (unsigned int i = 0; i < n_.size(); ++i) {
        if (n_[i] == 0) {
            std::clog << "n_[i] cannot be zero!\n";
        }
        total_num += n_[i];
    }
    if (total_num != nsize_A_ + nsize_B_) {
        std::clog << "total_num != nsize_A_ + nsize_B_; n_ is already wrong here!! \n";
    }
}

void blockmodel_t::shuffle_bisbm(std::mt19937 &engine, unsigned int NA, unsigned int NB) noexcept {
    std::shuffle(&memberships_[0], &memberships_[NA], engine);
    std::shuffle(&memberships_[NA], &memberships_[NA + NB], engine);
    compute_k();
}


void blockmodel_t::compute_k() noexcept {
    k_.clear();
    k_.resize(adj_list_ptr_->size());
    for (unsigned int i = 0; i < adj_list_ptr_->size(); ++i) {
        k_[i].resize(this->n_.size(), 0);
        for (auto nb = adj_list_ptr_->at(i).begin(); nb != adj_list_ptr_->at(i).end(); ++nb) {
            ++k_[i][memberships_[*nb]];
        }
    }
}

int_vec_t blockmodel_t::get_k_r_from_mb(uint_vec_t mb) const noexcept {
    int_vec_t k_r_;
    k_r_.clear();
    k_r_.resize(mb.size(), 0);

    int_vec_t n_r_ = get_n_r_from_mb(mb);
    for (auto i = 0; i < mb.size(); ++i) {
        k_r_[mb[i]] += deg_[i];
    }
    return k_r_;
}

double blockmodel_t::get_log_factorial(int number) const noexcept {
    double log_factorial = 0.;

    for (int i = 1; i <= number; ++i) {
        log_factorial += std::log(i);
    }
    return log_factorial;

}

double blockmodel_t::get_int_data_likelihood_from_mb_uni(uint_vec_t mb) const noexcept {
    uint_mat_t m_rs_ = get_m_from_membership(mb);
    int_vec_t n_r_ = get_n_r_from_mb(mb);
    int_vec_t k_r_ = get_k_r_from_mb(mb);
    double p_ = 2. * num_edges_ / (double) mb.size() / (double) mb.size();
    double log_idl = 0.;

    for (auto r = 0; r < n_r_.size(); ++r) {
        log_idl +=
                k_r_[r] * std::log(n_r_[r]) + get_log_factorial(n_r_[r] - 1) -
                get_log_factorial(n_r_[r] + k_r_[r] - 1);
        log_idl += get_log_factorial(m_rs_[r][r]) - (m_rs_[r][r] + 1.) * std::log(0.5 * p_ * n_r_[r] * n_r_[r] + 1);
        for (auto s = 0; s < r; ++s) {
            log_idl += get_log_factorial(m_rs_[r][s]) - (m_rs_[r][s] + 1.) * std::log(p_ * n_r_[r] * n_r_[s] + 1);
        }
    }
    return log_idl;
}

double blockmodel_t::get_int_data_likelihood_from_mb_bi(uint_vec_t mb) const noexcept {
    uint_mat_t m_rs_ = get_m_from_membership(mb);
    int_vec_t n_r_ = get_n_r_from_mb(mb);
    int_vec_t k_r_ = get_k_r_from_mb(mb);
    double p_ = 1. * num_edges_ / (double) get_nsize_A() / (double) get_nsize_B();
    double log_idl = 0.;

    for (auto r = 0; r < n_r_.size(); ++r) {
        log_idl +=
                k_r_[r] * std::log(n_r_[r]) + get_log_factorial(n_r_[r] - 1) -
                get_log_factorial(n_r_[r] + k_r_[r] - 1);
        //log_idl += get_log_factorial(m_rs_[r][r]) - (m_rs_[r][r] + 1.) * std::log(0.5 * p_ * n_r_[r] * n_r_[r] + 1);
        for (auto s = 0; s < r; ++s) {
            log_idl += get_log_factorial(m_rs_[r][s]) - (m_rs_[r][s] + 1.) * std::log(p_ * n_r_[r] * n_r_[s] + 1);
        }
    }
    return log_idl;
}


double blockmodel_t::get_log_posterior_from_mb_uni(uint_vec_t mb) const noexcept {
    uint_mat_t m_rs_ = get_m_from_membership(mb);
    int_vec_t n_r_ = get_n_r_from_mb(mb);
    int_vec_t k_r_ = get_k_r_from_mb(mb);
    double p_ = 2. * num_edges_ / (double) mb.size() / (double) mb.size();
    double log_posterior = 0.;

    for (auto r = 0; r < n_r_.size(); ++r) {
        log_posterior += get_log_factorial(n_r_[r]);
        log_posterior +=
                k_r_[r] * std::log(n_r_[r]) + get_log_factorial(n_r_[r] - 1) -
                get_log_factorial(n_r_[r] + k_r_[r] - 1);
        log_posterior +=
                get_log_factorial(m_rs_[r][r]) - (m_rs_[r][r] + 1.) * std::log(0.5 * p_ * n_r_[r] * n_r_[r] + 1);
        for (auto s = 0; s < r; ++s) {
            log_posterior +=
                    get_log_factorial(m_rs_[r][s]) - (m_rs_[r][s] + 1.) * std::log(p_ * n_r_[r] * n_r_[s] + 1);
        }
    }
    log_posterior -= KA_ * (nsize_A_ - 2.) + KB_ * (nsize_B_ - 2.);


    return log_posterior;
}

double blockmodel_t::get_log_posterior_from_mb_bi(uint_vec_t mb) const noexcept {
    uint_mat_t m_rs_ = get_m_from_membership(mb);
    int_vec_t n_r_ = get_n_r_from_mb(mb);
    int_vec_t k_r_ = get_k_r_from_mb(mb);
    double p_ = 1. * num_edges_ / (double) get_nsize_A() / (double) get_nsize_B();
    double log_posterior = 0.;

    for (auto r = 0; r < n_r_.size(); ++r) {
        log_posterior += get_log_factorial(n_r_[r]);
        log_posterior +=
                k_r_[r] * std::log(n_r_[r]) + get_log_factorial(n_r_[r] - 1) -
                get_log_factorial(n_r_[r] + k_r_[r] - 1);
        for (auto s = 0; s < r; ++s) {
            log_posterior +=
                    get_log_factorial(m_rs_[r][s]) - (m_rs_[r][s] + 1.) * std::log(p_ * n_r_[r] * n_r_[s] + 1);
        }
    }
    log_posterior -= KA_ * (nsize_A_ - 2.) + KB_ * (nsize_B_ - 2.);


    return log_posterior;
}