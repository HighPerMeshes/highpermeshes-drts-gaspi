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
 * last change: 29.05.2020                                                         *
 * ------------------------------------------------------------------------------ **/

#ifndef MONODOMAIN_CPP

#include <iostream>
#include <tuple>
#include <metis.h>
#include <HighPerMeshes.hpp>
#include <HighPerMeshes/third_party/metis/Partitioner.hpp>
#include <HighPerMeshesDRTS.hpp>
//#include <../../parser/WriteLoop.hpp>

#include <../examples/Functions/outputWriter.hpp>
#include <../examples/Functions/simplexGradients.hpp>

class DEBUG;
using namespace HPM;
using namespace ::HPM::auxiliary;
template <std::size_t ...I>
using Dofs           = HPM::dataType::Dofs<I...>;
using Vector         = std::vector<float>;
using Matrix         = std::vector<Vector>;
using CoordinateType = HPM::dataType::Vec<float,2>;
using Mesh           = HPM::mesh::PartitionedMesh<CoordinateType, HPM::entity::Simplex>;
constexpr int dim    = Mesh::CellDimension;

/*-------------------------------------------------------------- (A) Functions: -------------------------------------------------------------------------------*/
template<typename MeshT, typename BufferT, typename LoopBodyT>
void CreateStartVector(const MeshT & mesh, BufferT & startVec, const float & startValLeft, const float & startValRight, const int & maxX, const int & maxY, LoopBodyT bodyObj);

template<typename MeshT, typename LoopBodyT, typename BufferT>
void AssembleLumpedMassMatrix(const MeshT & mesh, LoopBodyT bodyObj, BufferT & lumpedMat) ;

template<typename MeshT, typename VectorT, typename LoopbodyT, typename BufferT>
void AssembleMatrixVecProduct2D(const MeshT & mesh, const VectorT & d, LoopbodyT bodyObj, BufferT & sBuffer);

template<typename BufferT, typename VectorT, typename LoopbodyT, typename MeshT>
void FWEuler(BufferT & vecOld, const VectorT & vecDeriv, const float & h, LoopbodyT body, const MeshT & mesh);

template<typename BufferT>
auto Iion(const BufferT & u, const BufferT & w, const float & a) -> Vector;

template<typename BufferT, typename MeshT, typename LoopBodyT, typename VectorT>
auto UDerivation(const BufferT & u, const MeshT & mesh, LoopBodyT bodyObj, const VectorT & Iion, const BufferT & lumpedM, const float & sigma) -> Vector;

template<typename BufferT>
auto WDerivation(const BufferT & u, const BufferT & w, const float & b, const float & eps) -> Vector;

/*----------------------------------------------------------------- MAIN --------------------------------------------------------------------------------------*/
int main(int argc, char** argv)
{
    /*------------------------------------------(1) Set run-time system and read mesh information: ------------------------------------------------------------*/
    HPM::drts::Runtime<HPM::GetDistributedBuffer<>, HPM::UsingDistributedDevices> hpm({}, std::forward_as_tuple(argc, argv));
    HPM::DistributedDispatcher body{hpm.gaspi_context, hpm.gaspi_segment, hpm};
    HPM::auxiliary::ConfigParser CFG("config.cfg");
    std::string meshFile = CFG.GetValue<std::string>("MeshFile");
    const Mesh mesh      = Mesh::template CreateFromFile<HPM::auxiliary::AmiraMeshFileReader, ::HPM::mesh::MetisPartitioner>
                           (meshFile, {hpm.GetL1PartitionNumber(), hpm.GetL2PartitionNumber()}, hpm.gaspi_runtime.rank().get());

    /*------------------------------------------(2) Set start values ------------------------------------------------------------------------------------------*/

    // input options for small mesh 5x5 (config.cfg -> mesh2D_test5x5.am)
//    int numIt = 2000; // number of iterations
//    float h   = 0.015; // step size
//    float a = -0.1; float b = 0.008; float eps = 0.1; float sigma = -0.1;
//    float v1 = 0.1; float v2 = 0.1; // value for start vector(s)

    // value option for bigger mesh 100x100 (config.cfg -> mesh2D.am)
    int numIt = 1500; // number of iterations
    float h   = 0.2; // step size
    float a = -0.1; float b = 1e-4; float eps = 0.005; float sigma = -0.1;
    float v1 = 1; float v2 = 0; // value for start vector(s)



    float u0L  = v1;  float u0R  = 0.F; // values for start vector u on \Omega_1 and \Omega_2
    float w0L  = 0.F; float w0R  = v2;  // values for start vector w on \Omega_1 and \Omega_2

    int numNodes = mesh.template GetNumEntities<0>();
    int maxX = std::ceil(std::sqrt(numNodes)/4);
    int maxY = std::ceil(std::sqrt(numNodes));

    HPM::Buffer<float, Mesh, Dofs<1, 0, 0, 0>> u(mesh);
    CreateStartVector(mesh, u, u0L, u0R, maxX, maxY, body);

    HPM::Buffer<float, Mesh, Dofs<1, 0, 0, 0>> w(mesh);
    CreateStartVector(mesh, w, w0L, w0R, maxX, maxY, body);

    Vector u_deriv;
    Vector w_deriv;
    Vector f_i;

    /*------------------------------------------(3) Create monodomain problem ---------------------------------------------------------------------------------*/
    HPM::Buffer<float, Mesh, Dofs<1, 0, 0, 0>> lumpedMat(mesh);
    AssembleLumpedMassMatrix(mesh, body, lumpedMat);

    // check if startvector was set correctly
    std::stringstream s;
    s << 0;
    std::string name = "test_100x100Mesh_newFWEuler" + s.str();
    writeVTKOutput2DTime(mesh, name, u, "resultU");

    for (int j = 0; j < numIt; ++j)
    {
        f_i     = Iion(u, w, a);
        u_deriv = UDerivation(u, mesh, body, f_i, lumpedMat, sigma);
        w_deriv = WDerivation(u, w, b, eps);
        FWEuler(u, u_deriv, h, body, mesh);
        FWEuler(w, w_deriv, h, body, mesh);

        if ((j+1)%10 == 0)
        {
            std::stringstream s;
            s << j+1;
            std::string name = "test_100x100Mesh_newFWEuler" + s.str();
            writeVTKOutput2DTime(mesh, name, u, "resultU");
        }
    }

    return 0;
}
#endif

/*----------------------------------------------------------------- (A) Functions (Implementation): -----------------------------------------------------------*/
//!
//! \brief Create start vector.
//!
template<typename MeshT, typename BufferT, typename LoopBodyT>
void CreateStartVector(const MeshT & mesh, BufferT & startVec, const float & startValLeft, const float & startValRight, const int & maxX, const int & maxY, LoopBodyT bodyObj)
{
    auto nodes { mesh.template GetEntityRange<0>() };

    bodyObj.Execute(HPM::ForEachEntity(
                  nodes,
                  std::tuple(Read(Node(startVec))),
                  [&](auto const& node, const auto& iter, auto& lvs)
    {
        auto coords = node.GetTopology().GetVertices();
        if ( (coords[0][0] < maxX) && (coords[0][1] < maxY) )
            startVec[node.GetTopology().GetIndex()] = startValLeft;
        else
            startVec[node.GetTopology().GetIndex()] = startValRight;
    }));

    return;
}

//!
//! \brief Assemble rom-sum lumped mass matrix
//!
template<typename MeshT, typename LoopBodyT, typename BufferT>
void AssembleLumpedMassMatrix(const MeshT & mesh, LoopBodyT bodyObj, BufferT & lumpedMat)
{
    auto cells {mesh.template GetEntityRange<2>()};
    bodyObj.Execute(HPM::ForEachEntity(
                        cells,
                        std::tuple(ReadWrite(Node(lumpedMat))),
                        [&](auto const& cell, const auto& iter, auto& lvs)
    {
        auto& lumpedMat = HPM::dof::GetDofs<HPM::dof::Name::Node>(std::get<0>(lvs));
        auto tmp        = cell.GetGeometry().GetJacobian();
        float detJ      = std::abs(tmp.Determinant());

        for (const auto& node1 : cell.GetTopology().template GetEntities<0>())
        {
            int id_node1 = node1.GetTopology().GetLocalIndex();
            for (const auto& node2 : cell.GetTopology().template GetEntities<0>())
            {
                if (node2.GetTopology().GetLocalIndex() == id_node1)
                    lumpedMat[id_node1][0] += detJ * 1/12;
                else
                    lumpedMat[id_node1][0] += detJ * 1/24;
            }
        }
    }));
    return;
}

//!
//! matrix-vector product split into single scalar operations
//!
template<typename MeshT, typename VectorT, typename LoopbodyT, typename BufferT>
void AssembleMatrixVecProduct2D(const MeshT & mesh, const VectorT & d, LoopbodyT bodyObj, BufferT & sBuffer)
{
    auto cells { mesh.template GetEntityRange<2>() };
    bodyObj.Execute(HPM::ForEachEntity(
                  cells,
                  std::tuple(ReadWrite(Node(sBuffer))),
                  [&](auto const& cell, const auto& iter, auto& lvs)
    {
        constexpr int nrows = dim+1;
        constexpr int ncols = dim+1;
        const auto& gradients = GetGradients2DP1();
        const auto& nodeIdSet = cell.GetTopology().GetNodeIndices();

        const auto& tmp  = cell.GetGeometry().GetJacobian();
        const float detJ = std::abs(tmp.Determinant());

        const auto& inv   = tmp.Invert();
        const auto& invJT = inv.Transpose();

        // separate GATHER
        std::array<float, nrows> _d;
        for (int row = 0; row < nrows; ++row)
            _d[row] = d[nodeIdSet[row]];

        // accumulate into contiguous block of memory
        std::array<float, ncols> result{};

        float val      = detJ * 0.5;
        for (int col = 0; col < ncols; ++col)
        {
            const auto& gc = invJT * gradients[col];
            for (int row = 0; row < nrows; ++row)
            {
                const auto& gr = invJT * gradients[row];
                result[col]   += ((gc*gr) * val) * _d[row];
            }
        }

        // separate SCATTER (accumulate)
        for (int col = 0; col < ncols; ++col)
            sBuffer[nodeIdSet[col]] += result[col];
    }));

    return;
}

//!
//! \brief Forward (explicit) Euler algorithm.
//!
template<typename BufferT, typename VectorT, typename LoopbodyT, typename MeshT>
void FWEuler(BufferT & vecOld, const VectorT & vecDeriv, const float & h, LoopbodyT body, const MeshT & mesh)
{
    auto vertices {mesh.template GetEntityRange<0>()};
    body.Execute(HPM::ForEachEntity(
                  vertices,
                  std::tuple(ReadWrite(Node(vecOld))),
                  [&](auto const& vertex, const auto& iter, auto& lvs)
    { vecOld[vertex.GetTopology().GetIndex()] += h*vecDeriv[vertex.GetTopology().GetIndex()]; }));
    return;
}

//!
//! \brief Compute ion current at time step t.
//!
template<typename BufferT>
auto Iion(const BufferT & u, const BufferT & w, const float & a) -> Vector
{
    Vector f_i; f_i.resize(u.GetSize());
    for (int i = 0; i < u.GetSize(); ++i)
        f_i[i] = (u[i] * (1-u[i]) * (u[i]-a)) - w[i];

    return f_i;
}

//!
//! \brief Compute derivation of u at time step t.
//!
template<typename BufferT, typename MeshT, typename LoopBodyT, typename VectorT>
auto UDerivation(const BufferT & u, const MeshT & mesh, LoopBodyT bodyObj, const VectorT & Iion, const BufferT & lumpedM, const float & sigma) -> Vector
{
    HPM::Buffer<float, Mesh, Dofs<1, 0, 0, 0>> s(mesh);
    AssembleMatrixVecProduct2D(mesh, u, bodyObj, s);

    Vector u_deriv; u_deriv.resize(u.GetSize());

    for (int i = 0; i < u.GetSize(); ++i)
        u_deriv[i] += ((1/lumpedM[i]) * sigma * s[i]) + Iion[i];

    return u_deriv;
}

//!
//! \brief Compute derivation of w at time step t.
//!
template<typename BufferT>
auto WDerivation(const BufferT & u, const BufferT & w, const float & b, const float & eps) -> Vector
{
    Vector w_deriv; w_deriv.resize(w.GetSize());
    for (int i = 0; i < w.GetSize(); ++i)
        w_deriv[i] = eps*(u[i]-b*w[i]);

    return w_deriv;
}
