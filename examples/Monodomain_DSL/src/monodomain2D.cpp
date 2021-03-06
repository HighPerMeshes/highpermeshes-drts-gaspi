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
 * last change: 04.06.2020                                                         *
 * ------------------------------------------------------------------------------ **/

#ifndef MONODOMAIN_CPP

#include <fstream>
#include <iostream>
#include <tuple>
#include <metis.h>
#include <HighPerMeshes.hpp>
#include <HighPerMeshes/third_party/metis/Partitioner.hpp>
#include <HighPerMeshesDRTS.hpp>
#include <HighPerMeshes/auxiliary/BufferOperations.hpp>
//#include <../../parser/WriteLoop.hpp>
#include <HighPerMeshes/drts/UsingGaspi.hpp>
#include <../examples/Functions/outputWriter.hpp>
#include <../examples/Functions/simplexGradients.hpp>
#include <HighPerMeshes/auxiliary/ArrayOperations.hpp>

#include <unistd.h>
#define GetCurrentDir getcwd

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
auto CreateFile(std::string const & pathToFolder, std::string const & foldername, std::string const & filename) -> std::ofstream;

//template<typename FileT, typename T, typename NameT, typename NameT2>
//void WriteParameterInfoIntoFile(FileT& file, const NameT& fileName, const T& parameter, const NameT2& parameterName);

void SetStartValues(int option, float& h, float& a, float& b, float& eps, float& sigma, float& u0L, float& u0R, float& w0L, float& w0R/*,
                    std::ofstream & file*/);

template<typename MeshT, typename BufferT, typename LoopBodyT>
void CreateStartVector(const MeshT & mesh, BufferT & startVec, const float & startValLeft, const float & startValRight,
                       const int & maxX, const int & maxY, LoopBodyT & body);

template<typename MeshT, typename LoopBodyT, typename BufferT>
void AssembleLumpedMassMatrix(const MeshT & mesh, LoopBodyT & body, BufferT & lumpedMat) ;

template<typename MeshT, typename VectorT, typename LoopbodyT, typename BufferT>
void AssembleMatrixVecProduct2D(const MeshT & mesh, const VectorT & d, LoopbodyT & body, BufferT & sBuffer);

template<typename BufferT, typename VectorT, typename LoopbodyT, typename MeshT>
void FWEuler(BufferT & vecOld, const VectorT & vecDeriv, const float & h, LoopbodyT & body, const MeshT & mesh);

template<typename BufferT, typename BufferTGlobal, typename MeshT, typename LoopBodyT>
void computeIionUDerivWDeriv(BufferT & f, BufferT & u_deriv, BufferT & w_deriv, const MeshT & mesh, LoopBodyT & body,
                             const BufferTGlobal & u, const BufferT & w, const BufferT & lumpedM, const float & sigma,
                             const float & a, const float & b, const float & eps);

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

    std::size_t i = 0;
    for (const auto& node : mesh.template GetEntities<0>())
        std::cout << "node " << i++ << ": " << node.GetTopology().GetNodes() << std::endl;

    /*------------------------------------------(2) Set directory-,folder- and filename of result -------------------------------------------------------------*/
    char buff[FILENAME_MAX]; //create string buffer to hold path
    GetCurrentDir(buff, FILENAME_MAX);
    std::string currentWorkingDir(buff);

    std::string foldername = "Test_new";
    std::string filename   = "Test_new_";
    //std::string parameterFilename = "testParameterfile";

    /*------------------------------------------(3) Set start values ------------------------------------------------------------------------------------------*/
    int numIt = 1000;
    float h; float a; float b; float eps; float sigma; float u0L; float u0R; float w0L; float w0R;
    //auto file = CreateFile(currentWorkingDir, foldername, parameterFilename);
    SetStartValues(1, h, a, b, eps, sigma, u0L, u0R, w0L, w0R/*, file*/);

    int numNodes = mesh.template GetNumEntities<0>();
    int maxX = std::ceil(std::sqrt(numNodes)/4);
    int maxY = std::ceil(std::sqrt(numNodes));

    //HPM::Buffer<float, Mesh, Dofs<1, 0, 0, 0>> u(mesh);
    //CreateStartVector(mesh, u, u0L, u0R, maxX, maxY, body);
    HPM::Buffer<float, Mesh, Dofs<1, 0, 0, 0>> u(mesh);
    CreateStartVector(mesh, u, u0L, u0R, maxX, maxY, body);

    HPM::Buffer<float, Mesh, Dofs<1, 0, 0, 0>> w(mesh);
    CreateStartVector(mesh, w, w0L, w0R, maxX, maxY, body);

    HPM::Buffer<float, Mesh, Dofs<1, 0, 0, 0>> u_deriv(mesh);
    HPM::Buffer<float, Mesh, Dofs<1, 0, 0, 0>> w_deriv(mesh);
    HPM::Buffer<float, Mesh, Dofs<1, 0, 0, 0>> f(mesh);

    /*------------------------------------------(4) Create monodomain problem ---------------------------------------------------------------------------------*/
    HPM::Buffer<float, Mesh, Dofs<1, 0, 0, 0>> lumpedMat(mesh);
    AssembleLumpedMassMatrix(mesh, body, lumpedMat);

    // read dofs
    const auto& dof_partitionU = u.GetDofPartition(0);
    std::cout<<"Num Global Dofs = "<<dof_partitionU.GetSize()<<std::endl;
//    Vector vec; vec.resize(numNodes);
//    for (std::size_t i = 0; i < dof_partitionU.GetSize(); ++i)
//        vec[i] = dof_partitionU[i];

    // check if startvector was set correctly by creating output file at time step zero
    std::stringstream s; s << 0;
    std::string name = filename + s.str();
    writeVTKOutput2DTime(mesh, currentWorkingDir, foldername, name, u, "resultU");
    //writeVTKOutput2DTime(mesh, currentWorkingDir, foldername, name, vec, "resultU");

    const std::size_t proc_id = hpm.gaspi_context.rank().get();

    // compute
    //HPM::UsingGaspi gaspi;
    for (int j = 0; j < 1; ++j)
    {
        computeIionUDerivWDeriv(f, u_deriv, w_deriv, mesh, body, u, w, lumpedMat, sigma, a, b, eps);
        FWEuler(u, u_deriv, h, body, mesh);
        FWEuler(w, w_deriv, h, body, mesh);
        //AllGather<0>(u, gaspi);
        const auto& u_gather = AllGather<0>(u, static_cast<::HPM::UsingGaspi&>(hpm));

        std::size_t index = 0;
        for (const auto& value : u_gather)
            std::cout << index % numNodes << ", " << index++ << ": " << value << std::endl;

        if (proc_id == 0) {
        std::vector<float> u_total(numNodes);
        const std::size_t num_buffers = u_gather.size ()/ numNodes;
        for (std::size_t i = 0; i < numNodes; ++i)
        {
            u_total[i] = 0;
            for (std::size_t k = 0; k < num_buffers; ++k)
                u_total[i] += u_gather[k * numNodes + i];
        }

        if ((j+1)%10 == 0)
        {
//            const auto& dof_partition = u.GetDofPartition(0);
//            Vector vec_u; vec_u.resize(dof_partition.GetSize());
//            for (std::size_t i = 0; i < dof_partition.GetSize(); ++i)
//                vec_u[i] = dof_partition[i];

            std::stringstream s; s << j+1;
            name = filename + s.str();
            writeVTKOutput2DTime(mesh, currentWorkingDir, foldername, name, u_total, "resultU");
            //writeVTKOutput2DTime(mesh, currentWorkingDir, foldername, name, vec_u, "resultU");
        }
        }
    }

    return 0;
}
#endif

/*----------------------------------------------------------------- (A) Functions (Implementation): -----------------------------------------------------------*/

//!
//! \brief Set start settings.
//!
void SetStartValues(int option, float& h, float& a, float& b, float& eps, float& sigma, float& u0L, float& u0R, float& w0L, float& w0R/*, std::ofstream& file*/)
{
    if (option == 1)
    {
        // input options for small mesh 5x5 (config.cfg -> mesh2D_test5x5.am)
        h     = 0.015; // step size
        a     = -0.1;
        b     = 0.008;
        eps   = 0.1;
        sigma = -0.1;
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
        sigma = -0.1;
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
        sigma = -0.1;
        u0L   = 1.F; // values for start vector u on \Omega_1 and \Omega_2
        u0R   = 0.F; // values for start vector u on \Omega_1 and \Omega_2
        w0L   = 0.F;  // values for start vector w on \Omega_1 and \Omega_2
        w0R   = 0.F;  // values for start vector w on \Omega_1 and \Omega_2
    }
    else
        printf("There is no option choosen for start value settings. Create your own option or choose one of the existing.");

    // add parameter to .txt file -> TODO: fix bug
//    std::string fileName = "testParameterfile.txt";
//    WriteParameterInfoIntoFile(file, fileName, h, "h");
//    WriteParameterInfoIntoFile(file, fileName, a, "a");
//    WriteParameterInfoIntoFile(file, fileName, b, "b");
//    WriteParameterInfoIntoFile(file, fileName, eps, "eps");
//    WriteParameterInfoIntoFile(file, fileName, sigma, "sigma");

    return;
}

//!
//! \brief Create start vector.
//!
template<typename MeshT, typename BufferT, typename LoopBodyT>
void CreateStartVector(const MeshT & mesh, BufferT & startVec, const float & startValLeft, const float & startValRight, const int & maxX, const int & maxY, LoopBodyT & body)
{
    auto nodes { mesh.template GetEntityRange<0>() };

    body.Execute(HPM::ForEachEntity(
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
void AssembleLumpedMassMatrix(const MeshT & mesh, LoopBodyT & body, BufferT & lumpedMat)
{
    auto cells {mesh.template GetEntityRange<2>()};
    body.Execute(HPM::ForEachEntity(
                        cells,
                        std::tuple(ReadWrite(Node(lumpedMat))),
                        [&](auto const& cell, const auto& iter, auto& lvs)
    {
        auto& lumpedMat = std::get<0>(lvs);
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
void AssembleMatrixVecProduct2D(const MeshT & mesh, const VectorT & d, LoopbodyT & body, BufferT & sBuffer)
{
    auto cells { mesh.template GetEntityRange<2>() };
    body.Execute(HPM::ForEachEntity(
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
void FWEuler(BufferT & vecOld, const VectorT & vecDeriv, const float & h, LoopbodyT & body, const MeshT & mesh)
{
    auto vertices {mesh.template GetEntityRange<0>()};
    body.Execute(HPM::ForEachEntity(
                  vertices,
                  std::tuple(ReadWrite(Node(vecOld))),
                  [&](auto const& vertex, const auto& iter, auto& lvs)
    {
        vecOld[vertex.GetTopology().GetIndex()] += h*vecDeriv[vertex.GetTopology().GetIndex()];
    }));
    return;
}

//!
//! \brief Compute ion current, derivation of u and derivation of w at time step t.
//!
template<typename BufferT, typename BufferTGlobal, typename MeshT, typename LoopBodyT>
void computeIionUDerivWDeriv(BufferT & f, BufferT & u_deriv, BufferT & w_deriv, const MeshT & mesh, LoopBodyT & body,
                             const BufferTGlobal & u, const BufferT & w, const BufferT & lumpedM, const float & sigma,
                             const float & a, const float & b, const float & eps)
{
    HPM::Buffer<float, Mesh, Dofs<1, 0, 0, 0>> s(mesh);
    AssembleMatrixVecProduct2D(mesh, u, body, s);

    auto vertices {mesh.template GetEntityRange<0>()};
    body.Execute(HPM::ForEachEntity(vertices, std::tuple(
                                    ReadWrite(Node(f)),/*Read*/Write(Node(u_deriv)),/*Read*/Write(Node(w_deriv))),
                                    [&](auto const& vertex, const auto& iter, auto& lvs)
    {
        int id      = vertex.GetTopology().GetIndex();
        f[id]       = (u[id] * (1-u[id]) * (u[id]-a)) - w[id];
        u_deriv[id] = ((1/lumpedM[id]) * sigma * s[id]) + f[id];
        w_deriv[id] = eps*(u[id]-b*w[id]);
    }));

    return;
}

//template<typename FileT, typename T, typename NameT, typename NameT2>
//void WriteParameterInfoIntoFile(FileT& file, const NameT& fileName, const T& parameter, const NameT2& parameterName)
//{
//    // TODO: fix bug at this code
//    file.open(fileName);
//    file.seekp(std::ios::end);
//    file << '\n' << parameterName << " = " << parameter << '\n';
//    file.close();
//    return;
//}

auto CreateFile(std::string const & pathToFolder, std::string const & foldername, std::string const & filename) -> std::ofstream
{
    std::string fname = pathToFolder + "/" + filename + ".txt";
    std::ofstream file(fname.c_str());
    file << "Some information about the parameter settings:" << '\n';
    return file;
}
