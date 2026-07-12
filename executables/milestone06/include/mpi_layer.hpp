#pragma once
#include <memory>

#include <lattice_boltzmann_types.hpp>
#include <mpi_data.hpp>
#include <mpi_impl.hpp>

class MPILayer {

    std::shared_ptr<MPIImplementation> mpi_communication_layer;

  public:
    std::shared_ptr<MPIData> mpi_data;

    int domain_width, domain_height;

    struct GhostLayers {
        int left = 0, right = 0, down = 0, up = 0;
    };

    GhostLayers ghost_layers{};
    int lattice_width, lattice_height;

    MPILayer(int domain_width, int domain_height);

    void communicate(const LatticeBoltzmann::DistributionFunction &);
};
