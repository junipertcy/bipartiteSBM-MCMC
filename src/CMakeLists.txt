add_executable(
        mcmc
        mcmc_main.cc metropolis_hasting.cc output_functions.cc graph_utilities.cc blockmodel.cc
        support/spence.cc support/cache.cc support/int_part.cc)

if (Boost_FOUND)
    target_link_libraries(mcmc ${Boost_LIBRARIES})
endif (Boost_FOUND)
