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
 * last change: 26.08.2020                                                         *
 * ------------------------------------------------------------------------------ **/

#ifndef MONODOMAIN_CPP

#include <mutex>
#include <fstream>
#include <iostream>
#include <tuple>
#include <metis.h>

#include <HighPerMeshesDRTS.hpp>
#include <../../highpermeshes-drts-gaspi/build/highpermeshes-dsl/include/HighPerMeshes.hpp>

//#include <HighPerMeshes/third_party/metis/Partitioner.hpp>
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
using Mesh           = HPM::mesh::Mesh<CoordinateType, HPM::entity::Simplex>; // for sequential case
//using Mesh           = mesh::PartitionedMesh<CoordinateType, entity::Simplex>; //for distributed case
constexpr int dim    = Mesh::CellDimension;

/*-------------------------------------------------------------- (A) Functions: -------------------------------------------------------------------------------*/
void SetStartValues(int option, float& h, float& a, float& b, float& eps, float& sigma, float& u0L, float& u0R, float& w0L, float& w0R);

template<typename MeshT, typename BufferT, typename DispatcherT>
void CreateStartVector(const MeshT & mesh, BufferT & startVec, const float & startValLeft, const float & startValRight,
                       const float & value, DispatcherT & dispatcher);

template<typename MeshT, typename DispatcherT, typename BufferT>
void AssembleLumpedMassMatrix(const MeshT & mesh, DispatcherT & dispatcher, BufferT & lumpedMat) ;

template<typename MeshT, typename VectorT, typename DispatcherT, typename BufferT>
void AssembleMatrixVecProduct2D(const MeshT & mesh, const VectorT & d, DispatcherT & dispatcher, BufferT & sBuffer);

template<typename BufferT, typename VectorT, typename DispatcherT, typename MeshT>
void FWEuler(BufferT & vecOld, const VectorT & vecDeriv, const float & h, DispatcherT & dispatcher, const MeshT & mesh);

template<typename BufferT, typename BufferTGlobal, typename MeshT, typename DispatcherT>
void computeIionUDerivWDeriv(BufferT & f, BufferT & u_deriv, BufferT & w_deriv, const MeshT & mesh, DispatcherT & dispatcher,
                             const BufferTGlobal & u, const BufferT & w, const BufferT & lumpedM, const float & sigma,
                             const float & a, const float & b, const float & eps, const bool & IStimOpt);

/*----------------------------------------------------------------- MAIN --------------------------------------------------------------------------------------*/
int main(int argc, char** argv)
{

    /*------------------------------------------(1a) Sequential case: -----------------------------------------------------------------------------------------*/
    HPM::drts::Runtime hpm { HPM::GetBuffer{} };
    HPM::SequentialDispatcher dispatcher;
    ConfigParser CFG("config3D.cfg");
    string meshFile = CFG.GetValue<string>("MeshFile");
    const Mesh mesh = Mesh::template CreateFromFile<HPM::auxiliary::AmiraMeshFileReader>(meshFile);

    /*------------------------------------------(1b) Distributed case: ----------------------------------------------------------------------------------------*/
//    drts::Runtime<GetDistributedBuffer<>, UsingDistributedDevices> hpm({}, forward_as_tuple(argc, argv));
//    DistributedDispatcher dispatcher{hpm.gaspi_context, hpm.gaspi_segment, hpm};
//    ConfigParser CFG("config3D.cfg");
//    string meshFile = CFG.GetValue<string>("MeshFile");
//    const Mesh mesh      = Mesh::template CreateFromFile<AmiraMeshFileReader, ::HPM::mesh::MetisPartitioner>
//            (meshFile, {hpm.GetL1PartitionNumber(), hpm.GetL2PartitionNumber()}, hpm.gaspi_runtime.rank().get());

    /*------------------------------------------(2) Set directory-,folder- and filename of result -------------------------------------------------------------*/
    char buff[FILENAME_MAX]; //create string buffer to hold path
    GetCurrentDir(buff, FILENAME_MAX);
    string currentWorkingDir(buff);

    string foldername = "Test3D";
    string filename   = "Test3D_SequentialDispatcher1000It";

    /*------------------------------------------(3) Set start values ------------------------------------------------------------------------------------------*/
    int numIt = 200;
    float h; float a; float b; float eps; float sigma; float u0L; float u0R; float w0L; float w0R;
    SetStartValues(4, h, a, b, eps, sigma, u0L, u0R, w0L, w0R);

    const float maxZ = -3.5;

    Buffer<float, Mesh, Dofs<1, 0, 0, 0, 0>> u(mesh);
    CreateStartVector(mesh, u, u0L, u0R, maxZ, dispatcher);

    Buffer<float, Mesh, Dofs<1, 0, 0, 0, 0>> w(mesh);
    CreateStartVector(mesh, w, w0L, w0R, maxZ, dispatcher);

    Buffer<float, Mesh, Dofs<1, 0, 0, 0, 0>> u_deriv(mesh);
    Buffer<float, Mesh, Dofs<1, 0, 0, 0, 0>> w_deriv(mesh);
    Buffer<float, Mesh, Dofs<1, 0, 0, 0, 0>> f(mesh);

    /*------------------------------------------(4) Create monodomain problem ---------------------------------------------------------------------------------*/
    Buffer<float, Mesh, Dofs<1, 0, 0, 0, 0>> lumpedMat(mesh);
    AssembleLumpedMassMatrix(mesh, dispatcher, lumpedMat);

    // check if startvector was set correctly by creating output file at time step zero
    stringstream s; s << 0;
    string name = filename + s.str();
    writeVTKOutputParabolicWO_BC(mesh, currentWorkingDir, foldername, name, u, "resultU");

    // compute u and w
    for (int j = 0; j < numIt; ++j)
    {
        //cout << "-----------------------------Iterationstep(u):   " << j << "---------------------------------------" << endl;

        bool IStimOpt;
        if (j == 0) IStimOpt = true;
        else IStimOpt = false;

        computeIionUDerivWDeriv(f, u_deriv, w_deriv, mesh, dispatcher, u, w, lumpedMat, sigma, a, b, eps, IStimOpt);
        FWEuler(u, u_deriv, h, dispatcher, mesh);
        FWEuler(w, w_deriv, h, dispatcher, mesh);

    }

    // write output files with result u
    for (int k = 0; k < numIt; ++k)
    {
        stringstream s; s << k+1;

        if ((k+1)%10 == 0)
        {
            name = filename + s.str();
            writeVTKOutputParabolicWO_BC(mesh, currentWorkingDir, foldername, name, u, "resultU");
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
        // input options for small mesh 5x5 (config.cfg -> mesh2D_test5x5.am)
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
        // input options for bigger mesh 100x100 (config.cfg -> mesh2D.am)
        h     = 0.4; // step size
        a     = 0.1;
        b     = 1e-4;
        eps   = 0.005;
        sigma = 0.1;
        u0L   = 1.F; // values for start vector u on \Omega_1 and \Omega_2
        u0R   = 0.F; // values for start vector u on \Omega_1 and \Omega_2
        w0L   = 0.F;  // values for start vector w on \Omega_1 and \Omega_2
        w0R   = 0.F;  // values for start vector w on \Omega_1 and \Omega_2
    }
    else if (option == 3)
    {
        // input options for bigger mesh 100x100 (config.cfg -> mesh2D.am)
        h     = 0.4; // step size
        a     = 0.1;
        b     = 1e-4;
        eps   = 1e-4;
        sigma = 0.1;
        u0L   = 1.F; // values for start vector u on \Omega_1 and \Omega_2
        u0R   = 0.F; // values for start vector u on \Omega_1 and \Omega_2
        w0L   = 0.F;  // values for start vector w on \Omega_1 and \Omega_2
        w0R   = 0.F;  // values for start vector w on \Omega_1 and \Omega_2
    }
    else if (option == 4)
    {
        // input options for bigger mesh 100x100 (config.cfg -> mesh2D.am)
        h     = 0.0005; // time step size h <= 0.0005
        a     = 0.7;
        b     = 1e-4;
        eps   = 1;
        sigma = 0.1; // diffusion tensor sigma <= 0.1
        u0L   = 1.F; // values for start vector u
        u0R   = 0.F; // values for start vector u
        w0L   = 0.F;  // values for start vector w
        w0R   = 1.F;  // values for start vector w
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
        if ( (coords[0][2] < value) /*&& (coords[0][1] < maxY)*/ )
            startVec[0] = startValLeft; //startVec[node.GetTopology().GetIndex()] = startValLeft;
        else
            startVec[0] = startValRight; //startVec[node.GetTopology().GetIndex()] = startValRight;
    }));

    return;
}

//!
//! \brief Assemble rom-sum lumped mass matrix
//!
template<typename MeshT, typename DispatcherT, typename BufferT>
void AssembleLumpedMassMatrix(const MeshT & mesh, DispatcherT & dispatcher, BufferT & lumpedMat)
{
    auto cells {mesh.template GetEntityRange<dim>()};
    dispatcher.Execute(ForEachEntity(
                           cells,
                           tuple(ReadWrite(Node(lumpedMat))),
                           [&](auto const& cell, const auto& iter, auto& lvs)
    {
        auto& lumpedMat = dof::GetDofs<dof::Name::Node>(get<0>(lvs));
        auto tmp        = cell.GetGeometry().GetJacobian();
        float detJ      = abs(tmp.Determinant());

        for (const auto& node1 : cell.GetTopology().template GetEntities<0>())
        {
            int id_node1 = node1.GetTopology().GetLocalIndex();
            for (const auto& node2 : cell.GetTopology().template GetEntities<0>())
            {
                if (node2.GetTopology().GetLocalIndex() == id_node1)
                    lumpedMat[id_node1][0] += detJ/60;// detJ * 1/12;
                else
                    lumpedMat[id_node1][0] += detJ/120;//detJ * 1/24;
            }
        }
    }));

    return;
}

//!
//! matrix-vector product split into single scalar operations
//!
template<typename MeshT, typename VectorT, typename DispatcherT, typename BufferT>
void AssembleMatrixVecProduct2D(const MeshT & mesh, const VectorT & d, DispatcherT & dispatcher, BufferT & sBuffer)
{
    auto cells { mesh.template GetEntityRange<dim>() };
    dispatcher.Execute(ForEachEntity(
                           cells,
                           tuple(ReadWrite(Node(sBuffer))),
                           [&](auto const& cell, const auto& iter, auto& lvs)
    {
        constexpr int nrows = dim+1;
        constexpr int ncols = dim+1;
        const auto& gradients = GetGradientsDSL();
        const auto& nodeIdSet = cell.GetTopology().GetNodeIndices();

        const auto& tmp  = cell.GetGeometry().GetJacobian();
        const float detJ = abs(tmp.Determinant());

        const auto& inv   = tmp.Invert();
        const auto& invJT = inv.Transpose();

        // separate GATHER
        array<float, nrows> _d;
        for (int row = 0; row < nrows; ++row)
            _d[row] = d[nodeIdSet[row]];

        // accumulate into contiguous block of memory
        array<float, ncols> result{};

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
template<typename BufferT, typename BufferTGlobal, typename MeshT, typename DispatcherT>
void computeIionUDerivWDeriv(BufferT & f, BufferT & u_deriv, BufferT & w_deriv, const MeshT & mesh, DispatcherT & dispatcher,
                             const BufferTGlobal & u, const BufferT & w, const BufferT & lumpedM, const float & sigma,
                             const float & a, const float & b, const float & eps, const bool & IStimOpt)
{
    Buffer<float, Mesh, Dofs<1, 0, 0, 0, 0>> s(mesh);
    AssembleMatrixVecProduct2D(mesh, u, dispatcher, s);

    float IStimVal;
    if (IStimOpt) IStimVal = 10.0;
    else IStimVal = 0.0;

    auto vertices {mesh.template GetEntityRange<0>()};
    dispatcher.Execute(ForEachEntity(vertices, tuple(
                                         ReadWrite(Node(f)),/*Read*/Write(Node(u_deriv)),/*Read*/Write(Node(w_deriv))),
                                     [&](auto const& vertex, const auto& iter, auto& lvs)
    {    
        int id      = vertex.GetTopology().GetIndex();
        f[id]       = (u[id] * (1-u[id]) * (u[id]-a)) - w[id];
        u_deriv[id] = -1 * ((1/lumpedM[id]) * sigma * s[id]) + f[id];
        if (id == 0) u_deriv[id] += IStimVal; // random choice of node (here node with id 0)
        w_deriv[id] = eps*(u[id]-b*w[id]);
    }));

    return;
}
