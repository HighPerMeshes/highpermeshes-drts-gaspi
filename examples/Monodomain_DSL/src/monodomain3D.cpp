/** ------------------------------------------------------------------------------ *
 * author(s)   : Merlind Schotte (schotte@zib.de)                                  *
 * institution : Zuse Institute Berlin (ZIB)                                       *
 * project     : HighPerMeshes (BMBF)                                              *
 *                                                                                 *
 * Description:                                                                    *
 * Implementation of monodomain example using the HighPerMeshes DSL.               *
 *                                                                                 *
 * Equation system: u'(t) = -div(\sigma \Nabla u) + I_{ion}(u,w)                   *
 *                  w'(t) = f(u,w)                                                 *
 * with \sigma as conductivity,I_{ion} as ion current and f(u,w) as gating dynamic.*
 *                                                                                 *
 *                  FitzHugh-Nagumo membrane model:                                *
 *                  I_{ion} = u(1-a)(u-a)-w                                        *
 *                  f(u,w)  = u - b*w                                              *
 *                                                                                 *
 * using the start vectors:                                                        *
 *                  u(0) = 1  on \Omega_1 and u(0) = 0  on \Omega_2,               *
 *                  w(0) = 0  on \Omega_1 and w(0) = 1  on \Omega_2.               *
 *                                                                                 *
 * last change: 30.12.2020                                                         *
 * ------------------------------------------------------------------------------ **/

#ifndef MONODOMAIN_CPP

#include <mutex>
#include <fstream>
#include <iostream>
#include <tuple>
#include <metis.h>

#include <HighPerMeshesDRTS.hpp>
#include <../../highpermeshes-drts-gaspi/build/highpermeshes-dsl/include/HighPerMeshes.hpp>

#include <../../highpermeshes-drts-gaspi/build/highpermeshes-dsl/include/HighPerMeshes/third_party/metis/Partitioner.hpp>
#include <HighPerMeshes/third_party/metis/Partitioner.hpp>
#include <HighPerMeshes/auxiliary/BufferOperations.hpp>
#include <HighPerMeshes/drts/UsingGaspi.hpp>
#include <../examples/Functions/outputWriter.hpp>

#include <../examples/Functions/simplexGradients.hpp>
#include <HighPerMeshes/auxiliary/ArrayOperations.hpp>

#include <../build/highpermeshes-dsl/utility/output/WriteLoop.hpp>
#include <../build/highpermeshes-dsl/tests/util/Grid.hpp>

#include <unistd.h>
#define GetCurrentDir getcwd

class DEBUG;
using namespace HPM;
using namespace ::HPM::auxiliary;
using namespace std;

template <size_t ...I>
using Dofs           = dataType::Dofs<I...>;
using Vector         = vector<float>;
using Matrix         = vector<Vector>;
using CoordinateType = dataType::Vec<float,3>;
//using Mesh           = HPM::mesh::Mesh<CoordinateType, HPM::entity::Simplex>; // for sequential case
using Mesh           = mesh::PartitionedMesh<CoordinateType, entity::Simplex>; //for distributed case
constexpr int dim    = Mesh::CellDimension;

/*-------------------------------------------------------------- (A) Functions: -------------------------------------------------------------------------------*/
void SetStartValues(int option, float& h, float& a, float& b, float& eps, float& sigma, float& u0L, float& u0R, float& w0L, float& w0R);

template<typename MeshT, typename BufferT, typename DispatcherT>
void CreateStartVector(const MeshT & mesh, BufferT & startVec, const float & startValLeft, const float & startValRight,
                       const float & value, DispatcherT & dispatcher);

template<typename MeshT, typename DispatcherT, typename BufferT>
void AssembleLumpedMassMatrix(const MeshT & mesh, DispatcherT & dispatcher, BufferT & lumpedMat) ;

template<typename MeshT, typename VectorT, typename DispatcherT, typename BufferT>
void AssembleMatrixVecProduct3D(const MeshT & mesh, const VectorT & d, DispatcherT & dispatcher, BufferT & sBuffer);

template<typename BufferT, typename VectorT, typename DispatcherT, typename MeshT>
void FWEuler(BufferT & vecOld, const VectorT & vecDeriv, const float & h, DispatcherT & dispatcher, const MeshT & mesh);

template<typename BufferT, typename MeshT, typename DispatcherT>
void computeIionUDerivWDeriv(BufferT & f, BufferT & u_deriv, BufferT & w_deriv, const MeshT & mesh, DispatcherT & dispatcher,
                             const BufferT & u, const BufferT & w, const BufferT & lumpedM, const float & sigma,
                             const float & a, const float & b, const float & eps, const bool & IStimOpt, const BufferT & IStim);

/*----------------------------------------------------------------- MAIN --------------------------------------------------------------------------------------*/
int main(int argc, char** argv)
{

    /*------------------------------------------(1a) Sequential case: -----------------------------------------------------------------------------------------*/
//    HPM::drts::Runtime hpm { HPM::GetBuffer{} };
//    HPM::SequentialDispatcher dispatcher;
//    ConfigParser CFG("config3D.cfg");
//    string meshFile = CFG.GetValue<string>("MeshFile");
//    const Mesh mesh = Mesh::template CreateFromFile<HPM::auxiliary::AmiraMeshFileReader>(meshFile);

    /*------------------------------------------(1b) Distributed case: ----------------------------------------------------------------------------------------*/
    drts::Runtime<GetDistributedBuffer<>, UsingDistributedDevices> hpm({}, forward_as_tuple(argc, argv));
    DistributedDispatcher dispatcher{hpm.gaspi_context, hpm.gaspi_segment, hpm};
    ConfigParser CFG("config3D.cfg");
    string meshFile = CFG.GetValue<string>("MeshFile");
    const Mesh mesh      = Mesh::template CreateFromFile<AmiraMeshFileReader, ::HPM::mesh::MetisPartitioner>
            (meshFile, {hpm.GetL1PartitionNumber(), hpm.GetL2PartitionNumber()}, hpm.gaspi_runtime.rank().get());

    /*------------------------------------------(2) Set directory-,folder- and filename of result -------------------------------------------------------------*/
    char buff[FILENAME_MAX]; //create string buffer to hold path
    GetCurrentDir(buff, FILENAME_MAX);
    string currentWorkingDir(buff);

    string foldername = "Test3D";
    string filename   = "Test3D_ParamSet2Distribute_";

    /*------------------------------------------(3) Set start values ------------------------------------------------------------------------------------------*/
    int numIt    = 500;
    int numNodes = mesh.template GetNumEntities<0>();
    float h; float a; float b; float eps; float sigma; float u0L; float u0R; float w0L; float w0R;
    SetStartValues(2, h, a, b, eps, sigma, u0L, u0R, w0L, w0R);

    const float maxZ = -1.5;//-3.5;

    Buffer<float, Mesh, Dofs<1, 0, 0, 0, 0>> u(mesh);
    CreateStartVector(mesh, u, u0L, u0R, maxZ, dispatcher);

    Buffer<float, Mesh, Dofs<1, 0, 0, 0, 0>> w(mesh);
    CreateStartVector(mesh, w, w0L, w0R, maxZ, dispatcher);

    Buffer<float, Mesh, Dofs<1, 0, 0, 0, 0>> IStim(mesh);
    CreateStartVector(mesh, u, 1.0, 0.0, maxZ, dispatcher);

    Buffer<float, Mesh, Dofs<1, 0, 0, 0, 0>> u_deriv(mesh);
    Buffer<float, Mesh, Dofs<1, 0, 0, 0, 0>> w_deriv(mesh);
    Buffer<float, Mesh, Dofs<1, 0, 0, 0, 0>> f(mesh);

    /*------------------------------------------(4) Create monodomain problem ---------------------------------------------------------------------------------*/
    Buffer<float, Mesh, Dofs<1, 0, 0, 0, 0>> lumpedMat(mesh);
    AssembleLumpedMassMatrix(mesh, dispatcher, lumpedMat);

    // compute u and w
    for (int j = 0; j < numIt; ++j)
    {
        bool IStimOpt;
        if (j == 0) IStimOpt = true;
        else IStimOpt = false;

        computeIionUDerivWDeriv(f, u_deriv, w_deriv, mesh, dispatcher, u, w, lumpedMat, sigma, a, b, eps, IStimOpt, IStim);
        FWEuler(u, u_deriv, h, dispatcher, mesh);
        FWEuler(w, w_deriv, h, dispatcher, mesh);

        const auto u_gather = HPM::auxiliary::AllGather<0>(u, static_cast<::HPM::UsingGaspi&>(hpm));
        std::vector<float> u_total(numNodes);
        const std::size_t num_buffers = u_gather.size ()/ numNodes;
        for (std::size_t i = 0; i < numNodes; ++i)
        {
            u_total[i] = 0;
            for (std::size_t k = 0; k < num_buffers; ++k)
                u_total[i] += u_gather[k * numNodes + i];
        }

        // write output files with result u
        if ((j+1)%20 == 0)
        {
            stringstream s; s << j+1;
            string name = filename + s.str();
            writeVTKOutputParabolicWO_BC(mesh, currentWorkingDir, foldername, name, u_total, "U");
        }
    }

    return 0;
}
#endif

/*----------------------------------------------------------------- (A) Functions (Implementation): -----------------------------------------------------------*/

//!
//! \brief Set start settings.
//!
void SetStartValues(int option, float& h, float& a, float& b, float& eps, float& sigma, float& u0L, float& u0R, float& w0L, float& w0R/*, ofstream& file*/)
{
    if (option == 1)
    {
        h     = 0.015; // step size
        a     = -0.1;
        b     = 0.008;
        eps   = 0.1;
        sigma = 0.1;
        u0L   = 0.1; // values for start vector u on \Omega_1 and \Omega_2
        u0R   = 0.F; // values for start vector u on \Omega_1 and \Omega_2
        w0L   = 0.F;  // values for start vector w on \Omega_1 and \Omega_2
        w0R   = 0.1;  // values for start vector w on \Omega_1 and \Omega_2
    }
    else if (option == 2)
    {
        h     = 0.00001; // time step size h <= 0.0005
        a     = 0.7;
        b     = 1e-4;
        eps   = 4.0;//0.5; //1;
        sigma = 10; // diffusion tensor sigma <= 0.1
        u0L   = 0.F; //1.F; // values for start vector u
        u0R   = 0.F; // values for start vector u
        w0L   = 0.F;  // values for start vector w
        w0R   = 0.F;  // values for start vector w
    }
    else if (option == 3)
    {
        // input options for bigger mesh 100x100 (config.cfg -> mesh2D.am)
        h     = 0.005; // step size
        a     = 0.1;
        b     = 1e-4;
        eps   = 1e-3;//0.001;
        sigma = 0.1;
        u0L   = 1.F; // values for start vector u on \Omega_1 and \Omega_2
        u0R   = 0.F; // values for start vector u on \Omega_1 and \Omega_2
        w0L   = 0.F;  // values for start vector w on \Omega_1 and \Omega_2
        w0R   = 0.F;  // values for start vector w on \Omega_1 and \Omega_2
    }
    else
        printf("There is no option choosen for start value settings. Create your own option or choose one of the existing.");

    return;
}

//!
//! \brief Create start vector.
//!
template<typename MeshT, typename BufferT, typename DispatcherT>
void CreateStartVector(const MeshT & mesh, BufferT & startVec, const float & startValLeft, const float & startValRight, const float & value, DispatcherT & dispatcher)
{
    auto nodes { mesh.template GetEntityRange<0>() };

    dispatcher.Execute(ForEachEntity(
                           nodes,
                           tuple(Write(Node(startVec))),
                           [&](auto const& node, const auto& iter, auto& lvs)
    {
        auto &startVec = HPM::dof::GetDofs<0>(std::get<0>(lvs));
        auto coords = node.GetTopology().GetVertices();
        if ( (coords[0][2] < value) )
            startVec[0] = startValLeft;
        else
            startVec[0] = startValRight;
    }));

    return;
}

////!
////! \brief Assemble row-sum lumped mass matrix
////!
template<typename MeshT, typename DispatcherT, typename BufferT>
void AssembleLumpedMassMatrix(const MeshT & mesh, DispatcherT & dispatcher, BufferT & lumpedMat)
{
    auto nodes {mesh.template GetEntityRange<0>()};
    dispatcher.Execute(ForEachEntity(nodes,
                                     tuple(ReadWrite(Node(lumpedMat))),
                                     [&](auto const& node, const auto& iter, auto& lvs)
    {
        auto& lumpedMat = dof::GetDofs<dof::Name::Node>(get<0>(lvs));
        const auto& cells = node.GetTopology().GetAllContainingCells();

        for (const auto& cell : cells)
        {
            const auto& J     = cell.GetGeometry().GetJacobian();
            const float detJ  = abs(J.Determinant());
            lumpedMat[0]     += detJ/60 + detJ/120 + detJ/120 + detJ/120;
        }
    }));
    return;
}

//!
//! matrix-vector product split into single scalar operations
//!
template<typename MeshT, typename VectorT, typename DispatcherT, typename BufferT>
void AssembleMatrixVecProduct3D(const MeshT & mesh, const VectorT & d, DispatcherT & dispatcher, BufferT & sBuffer)
{
    auto nodes {mesh.template GetEntityRange<0>()};
    dispatcher.Execute(ForEachEntity(nodes,
                                     tuple(Write(Node(sBuffer))),
                                     [&](auto const& node, const auto& iter, auto& lvs)
    {
        auto& sBuffer = dof::GetDofs<dof::Name::Node>(get<0>(lvs));
        constexpr int nrows = dim+1;
        const auto& gradients = GetGradients3DP1();
        const auto & cells = node.GetTopology().GetAllContainingCells();

        for (const auto& cell : cells)
        {
            const auto& nodeIdSet = cell.GetTopology().GetNodeIndices();
            int locID = -1;
            for (int i = 0; i < dim+1; ++i) {if (node.GetTopology().GetIndex() == nodeIdSet[i]) locID = i;}

            const auto& J     = cell.GetGeometry().GetJacobian();
            const float detJ  = abs(J.Determinant());
            const auto& invJ  = J.Invert();
            const auto& invJT = invJ.Transpose();

            // separate GATHER
            array<float, nrows> _d;
            for (int row = 0; row < nrows; ++row)
                _d[row] = d[nodeIdSet[row]];

            const auto& gc = invJT * gradients[locID] * (detJ/6);
            for (int row = 0; row < nrows; ++row)
            {
                const auto& gr = invJT * gradients[row];
                sBuffer[0]    += ((gc*gr)) * _d[row];
            }
        }
    }));

    return;
}

//!
//! \brief Forward (explicit) Euler algorithm.
//!
template<typename BufferT, typename VectorT, typename DispatcherT, typename MeshT>
void FWEuler(BufferT & vecOld, const VectorT & vecDeriv, const float & h, DispatcherT & dispatcher, const MeshT & mesh)
{
    auto vertices {mesh.template GetEntityRange<0>()};

    dispatcher.Execute(
                ForEachEntity(
                    vertices,
                    tuple(ReadWrite(Node(vecOld))),
                    [&](auto const& vertex, const auto& iter, auto& lvs) {
        vecOld[vertex.GetTopology().GetIndex()] += h*vecDeriv[vertex.GetTopology().GetIndex()];
    }));

    return;
}

//!
//! \brief Compute ion current, derivation of u and derivation of w at time step t.
//!
template<typename BufferT, typename MeshT, typename DispatcherT>
void computeIionUDerivWDeriv(BufferT & f, BufferT & u_deriv, BufferT & w_deriv, const MeshT & mesh, DispatcherT & dispatcher,
                             const BufferT & u, const BufferT & w, const BufferT & lumpedM, const float & sigma,
                             const float & a, const float & b, const float & eps, const bool & IStimOpt, const BufferT & IStim)
{
    Buffer<float, Mesh, Dofs<1, 0, 0, 0, 0>> s(mesh);
    AssembleMatrixVecProduct3D(mesh, u, dispatcher, s);

    auto vertices {mesh.template GetEntityRange<0>()};
    dispatcher.Execute(ForEachEntity(vertices, tuple(
                                         ReadWrite(Node(f)), Write(Node(u_deriv)), Write(Node(w_deriv))),
                                     [&](auto const& vertex, const auto& iter, auto& lvs)
    {    
        int id      = vertex.GetTopology().GetIndex();
        f[id]       = (u[id] * (1-u[id]) * (u[id]-a)) - w[id];
        if (IStimOpt) f[id] += IStim[id];
        u_deriv[id] = ((1/lumpedM[id]) * -1 * sigma * s[id]) + f[id];
        w_deriv[id] = eps*(u[id]-(b*w[id]));
    }));

    return;
}
