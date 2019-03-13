#ifndef BLOCKMODEL_HH
#define BLOCKMODEL_HH


#include <random>
#include <utility>
#include <algorithm> // std::shuffle
#include <vector>
#include "types.hh"
#include "output_functions.hh"

class blockmodel_t {
protected:
    std::uniform_real_distribution<> random_real;
    std::random_device rd;
    std::mt19937 gen{rd()};

public:
    /** Default constructor */
    blockmodel_t(const uint_vec_t& memberships, uint_vec_t types, size_t g, size_t KA,
                 size_t KB, double epsilon, const adj_list_t* adj_list_ptr);

    const int_vec_t* get_k(size_t vertex) const noexcept;

    const int get_degree(size_t vertex) const noexcept;

    const int get_num_edges() const noexcept;

    const int get_na() const noexcept;

    const int get_nb() const noexcept;

    const uint_vec_t* get_memberships() const noexcept;

    const int_mat_t* get_m() const noexcept;  /* Optimizing this function (pass by ref) is extremely important!! (why?) */

    const int_vec_t* get_m_r() const noexcept;

    const uint_mat_t* get_eta_rk_() const noexcept;

    const int_vec_t* get_n_r() const noexcept;

    size_t get_g() const noexcept;

    double get_epsilon() const noexcept;

    double get_entropy() const noexcept;

    size_t get_KA() const noexcept;

    size_t get_KB() const noexcept;

    uint_vec_t& get_vlist() noexcept;

    std::vector< std::vector<size_t> >& get_adj_list() noexcept;

    void shuffle_bisbm(std::mt19937& engine, size_t NA, size_t NB) noexcept;

    void init_bisbm() noexcept;

    bool apply_mcmc_moves(std::vector<mcmc_state_t>& moves, double dS) noexcept;

    std::vector<mcmc_state_t> single_vertex_change(std::mt19937& engine, size_t vtx) noexcept;

    void summary() noexcept;

    double entropy() noexcept;

private:
    /// State variable
    size_t KA_{0};
    size_t na_{0};
    size_t KB_{0};
    size_t nb_{0};
    size_t K_{0};
    unsigned int max_degree_{0};
    double epsilon_{0.};
    double entropy_{0.};  // not true entropy
    const adj_list_t * const adj_list_ptr_;

    int_mat_t k_;
    int_vec_t n_r_;

    int_vec_t deg_;
    std::vector< std::vector<size_t> > adj_list_;
    size_t num_edges_ = 0;

    uint_vec_t memberships_;
    uint_vec_t vlist_;
    const uint_vec_t types_;

    double entropy_from_degree_correction_{0.};

    int_mat_t m_;
    int_vec_t m_r_;
    uint_mat_t eta_rk_;  // number of nodes of degree k that belong to group r.

    /// used for `estimate` mode
    size_t which_to_move_{0};

    /// in apply_mcmc_moves
    const int_vec_t* ki_;

    /// for single_vertex_change
    double R_t_{0.};
    size_t vertex_j_{0};
    size_t proposal_t_{0};

    /// for single_vertex_change and apply_mcmc_moves
    size_t __vertex__{0};
    size_t __source__{0};
    size_t __target__{0};

    std::vector<mcmc_state_t> moves_ = std::vector<mcmc_state_t>(1);

    /// Internal distribution. Generator must be passed as a service
    std::uniform_int_distribution<size_t> random_block_;

    /// Private methods
    /* Compute stuff from scratch. */
    void compute_k() noexcept;
    void compute_m() noexcept;  // Note: get_m and compute_m are different.
    void compute_m_r() noexcept;
    void compute_eta_rk() noexcept;
    void compute_n_r() noexcept;
};

#endif // BLOCKMODEL_H