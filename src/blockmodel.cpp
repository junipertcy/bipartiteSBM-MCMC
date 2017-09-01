#include <iostream>
#include "blockmodel.h"
#include "output_functions.h"  // TODO: debug use
#include "graph_utilities.h"  // for the is_disjoint function
#include <boost/math/special_functions/gamma.hpp>

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
        auto num_nodes = (double) states[0].memberships.size();
        if (random_real(engine) < 1. / (num_nodes - 1)) {  // type-II move
            //        std::clog << "1\n";
            auto r = unsigned(int(random_real(engine) * (K_ + 1)));
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

    // decide whether to update type-a nodes or type-b nodes
    auto num_nodes = (double) states[0].memberships.size();
    auto num_nodes_a = (double) nsize_A_;
    auto num_nodes_b = (double) nsize_B_;

    if (random_real(engine) < num_nodes_a / num_nodes) {  // move type-a nodes
        if (random_real(engine) < 1. / (num_nodes_a - 1.)) {  // type-II move for type-a nodes
            auto r = unsigned(int(random_real(engine) * (KA_ + 1)));  // 0 or 1; if r = 1, s = 0
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
                for (auto node = 0; node < states[0].memberships.size(); ++node) {  // if r = 1, s = 0;
                    if (types_[node] == 1) {
                        ++states[0].memberships[node];
                    }
                }
            }
            // The code here deals with special cases: i.e. when nodes to be moved in group-s is the last node
            // in the group
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
            } else {
                auto r = unsigned(int(random_real(engine) * KA_));
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

    if (KA_ == 1 && KB_ == 1) {
        // return trivial move
        moves[0].vertex = unsigned(random_node_(engine));
        moves[0].source = memberships_[moves[0].vertex];
        moves[0].target = moves[0].source;
        return moves;
    }

    //TODO: improve this block
    unsigned int K = 1;
    while (K == 1) {
        moves[0].vertex = unsigned(random_node_(engine));
        if (types_[moves[0].vertex] == 0) {
            K = KA_;
        } else if (types_[moves[0].vertex] == 1) {
            K = KB_;
        }
    }
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

// TODO: move it to the template?
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

        for (auto _mb: mb) {
            if (_mb > cand_K_) {
                cand_K_ = _mb;
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

    for (auto _mb: mb) {
        ++n_r_[_mb];
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

    for (unsigned int j = 0; j < states.size(); ++j) {
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

    for (unsigned int j = 0; j < states.size(); ++j) {
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
        for (auto k: memberships_) {
            ++n_[k];
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
    k_r_.resize(mb.size(), 0);

    int_vec_t n_r_ = get_n_r_from_mb(mb);

    for (auto i = 0; i < mb.size(); ++i) {
        k_r_[mb[i]] += deg_[i];
    }
    return k_r_;
}

double blockmodel_t::get_log_factorial(int number) const noexcept {
    double log_factorial = lgamma(number + 1.);

    // original, slower, implementation;
//    double log_factorial = 0.;
//
//    for (int i = 1; i <= number; ++i) {
//        log_factorial += std::log(i);
//    }

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
                k_r_[r] * std::log(n_r_[r]) + get_log_factorial(n_r_[r] - 1) - get_log_factorial(n_r_[r] + k_r_[r] - 1);
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
    log_posterior -= K_ * (memberships_.size() - 2.);


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