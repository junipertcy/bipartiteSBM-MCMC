#ifndef BLOCKMODEL_H
#define BLOCKMODEL_H


#include <random>
#include <utility>
#include <algorithm> // std::shuffle
#include <vector>
#include "types.h"

class blockmodel_t {
protected:
    std::uniform_real_distribution<> random_real;
    std::random_device rd;
    std::mt19937 gen{rd()};

public:
    // Ctor
    blockmodel_t() : random_real(0, 1) { ; }

    blockmodel_t(const uint_vec_t &memberships, const uint_vec_t &types, unsigned int g, unsigned int KA,
                 unsigned int KB, double epsilon, unsigned int N, adj_list_t *adj_list_ptr, bool is_bipartite);

    std::vector<mcmc_state_t> mcmc_state_change_riolo_uni1(std::mt19937 &engine);
    std::vector<mcmc_state_t> mcmc_state_change_riolo_uni2(std::mt19937 &engine);
    std::vector<mcmc_state_t> mcmc_state_change_riolo(std::mt19937 &engine);

    std::vector<mcmc_state_t> single_vertex_change_naive(std::mt19937 &engine) noexcept;
    std::vector<mcmc_state_t> single_vertex_change_tiago(std::mt19937 &engine) noexcept;

    int_vec_t get_k(unsigned int vertex) const noexcept;

    int_vec_t get_size_vector() const noexcept;

    int_vec_t get_degree() const noexcept;

    uint_vec_t get_memberships() const noexcept;

    uint_vec_t get_types() const noexcept;

    uint_mat_t get_m() const noexcept;

    uint_mat_t get_m_from_membership(uint_vec_t mb) const noexcept;

    uint_vec_t get_m_r() const noexcept;

    uint_vec_t get_m_r_from_m(uint_mat_t m) const noexcept;

    int_vec_t get_n_r_from_mb(uint_vec_t mb) const noexcept;

    bool get_is_bipartite() const noexcept;

    unsigned int get_N() const noexcept;

    unsigned int get_g() const noexcept;

    unsigned int get_num_edges() const noexcept;

    double get_entropy_from_degree_correction() const noexcept;

    double get_epsilon() const noexcept;

    unsigned int get_K() const noexcept;

    unsigned int get_KA() const noexcept;

    unsigned int get_KB() const noexcept;

    double compute_entropy_from_m_mr(uint_mat_t m, uint_vec_t m_r) const noexcept;

    double get_log_factorial(int number) const noexcept;

    unsigned int get_nsize_A() const noexcept;

    unsigned int get_nsize_B() const noexcept;

    unsigned int get_n_from_mb(uint_vec_t mb) const noexcept;

    void apply_mcmc_moves(std::vector<mcmc_state_t> moves) noexcept;

    void apply_mcmc_states_u(std::vector<mcmc_state_t> states) noexcept;  // for uni-partite SBM

    void apply_mcmc_states(std::vector<mcmc_state_t> states) noexcept;

    void shuffle_bisbm(std::mt19937 &engine, unsigned int NA, unsigned int NB) noexcept;

    bool debugger(uint_vec_t mb) noexcept;

    int_vec_t get_k_r_from_mb(uint_vec_t mb) const noexcept;

    double get_int_data_likelihood_from_mb_uni(uint_vec_t mb) const noexcept;

    double get_int_data_likelihood_from_mb_bi(uint_vec_t mb) const noexcept;

    double get_log_posterior_from_mb_uni(uint_vec_t mb) const noexcept;

    double get_log_posterior_from_mb_bi(uint_vec_t mb) const noexcept;

    double get_log_single_type_prior(uint_vec_t mb, unsigned int type) const noexcept;

private:
    /// State variable
    bool is_bipartite_ = true;
    unsigned int K_ = 0;
    unsigned int KA_ = 0;
    unsigned int nsize_A_ = 0;
    unsigned int KB_ = 0;
    unsigned int nsize_B_ = 0;
    double epsilon_ = 0.;
    adj_list_t *adj_list_ptr_;
    int_mat_t k_;
    int_vec_t n_;
    int_vec_t deg_;
    uint_vec_t memberships_;
    uint_vec_t types_;
    unsigned int num_edges_ = 0;
    double entropy_from_degree_correction_ = 0.;
    uint_mat_t adj_list_;
    uint_mat_t m_;
    uint_vec_t m_r_;

    /// Internal distribution. Generator must be passed as a service
    std::uniform_int_distribution<> random_block_;
    std::uniform_int_distribution<> random_node_;
    /// Private methods
    /* Compute the degree matrix from scratch. */
    void compute_k() noexcept;
    void compute_m() noexcept;  // Note: get_m and compute_m are different.
    void compute_m_r() noexcept;

};

#endif // BLOCKMODEL_H