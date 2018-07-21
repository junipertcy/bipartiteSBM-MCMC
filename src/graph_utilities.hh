#ifndef GRAPH_UTILITIES_H
#define GRAPH_UTILITIES_H

#include <string>
#include <fstream>
#include <sstream>
#include <math.h>
#include "types.hh"

/* Load beliefs of memberships of each node. Returns true on sucess. */
bool load_memberships(uint_vec_t & memberships, const std::string membership_path);

/* Load an edge list. Result passed by reference. Returns true on success. */
bool load_edge_list(edge_list_t & edge_list, const std::string edge_list_path);

/* Convert adjacency list to edge list. Result passed by reference. */
adj_list_t edge_to_adj(const edge_list_t & edge_list, unsigned int num_vertices=0);

/* Check if two sets are disjoint. Returns true if disjoint. */
// Check: https://stackoverflow.com/questions/1964150/c-test-if-2-sets-are-disjoint
template<class Set1, class Set2>
bool is_disjoint(const Set1 &set1, const Set2 &set2)
{
    if(set1.empty() || set2.empty()) return true;

    typename Set1::const_iterator
            it1 = set1.begin(),
            it1End = set1.end();
    typename Set2::const_iterator
            it2 = set2.begin(),
            it2End = set2.end();

    if(*it1 > *set2.rbegin() || *it2 > *set1.rbegin()) return true;

    while(it1 != it1End && it2 != it2End)
    {
        if(*it1 == *it2) return false;
        if(*it1 < *it2) { it1++; }
        else { it2++; }
    }
    return true;
}


#endif // GRAPH_UTILITIES_H