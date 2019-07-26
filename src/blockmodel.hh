#ifndef BLOCKMODEL_HH
#define BLOCKMODEL_HH


#include <random>
#include <utility>
#include <algorithm> // std::shuffle
#include <vector>
#include <map>
#include <queue>
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

    int get_degree(size_t vertex) const noexcept;

    int get_num_edges() const noexcept;

    int get_na() const noexcept;

    int get_nb() const noexcept;

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

    void agg_merge(std::mt19937 &engine, int diff_a, int diff_b, int nm) noexcept;

    void agg_merge(std::mt19937 &engine, int diff, int nm) noexcept;

    void agg_split(std::mt19937 &engine, bool type, int nm) noexcept;

    double compute_dS(mcmc_move_t& move) noexcept;

    double compute_dS(const block_move_t& move) noexcept;

    double compute_dS(size_t mb, std::vector<bool>& split_move) noexcept;

    std::vector< std::vector<size_t> >& get_adj_list() noexcept;

    void shuffle_bisbm(std::mt19937& engine, size_t NA, size_t NB) noexcept;

    void init_bisbm() noexcept;

    void apply_split_moves(const std::vector<mcmc_move_t>& moves) noexcept;

    bool apply_mcmc_moves(const std::vector<mcmc_move_t>& moves, double dS) noexcept;

    void apply_block_moves(const std::set<size_t>& impacted, const std::vector<std::set<size_t>>& accepted) noexcept;

    std::vector<mcmc_move_t> single_vertex_change(std::mt19937& engine, size_t vtx) noexcept;

    block_move_t& single_block_change(std::mt19937& engine, size_t src) noexcept;

    void summary() noexcept;

    double entropy() noexcept;

    double null_entropy() noexcept;

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
    std::vector< std::vector<size_t> > b_adj_list_;
    int_map_vec_t adj_map_;  // for entropy() only
    size_t num_edges_ = 0;

    uint_vec_t memberships_;
    uint_vec_t vlist_;
    uint_vec_t blist_;
    std::vector<bool> splitter_;
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

    std::vector<mcmc_move_t> moves_ = std::vector<mcmc_move_t>(1);
    std::vector<block_move_t> bmoves_;
    block_move_t bmove_;
    std::vector<std::set<size_t>> accepted_set_vec_;

    /// Private methods
    /* Compute stuff from scratch. */
    void compute_b_adj_list() noexcept;
    void compute_k() noexcept;
    void compute_m() noexcept;  // Note: get_m and compute_m are different.
    void compute_m_r() noexcept;
    void compute_eta_rk() noexcept;
    void compute_n_r() noexcept;
};


#endif // BLOCKMODEL_H