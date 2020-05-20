/** ------------------------------------------------------------------------------ *
 * author(s)   : Merlind Schotte (schotte@zib.de)                                  *
 * institution : Zuse Institute Berlin (ZIB)                                       *
 * project     : HighPerMeshes (BMBF)                                              *
 *                                                                                 *
 * Description:                                                                    *
 * Implementation of monodomain example using the HighPerMeshes DSL (more          *
 * descriptions later).                                                            *
 *                                                                                 *
 * Equation system: u'(t) = -div(\sigma \Nabla u) + I_{ion}(u,w)                   *
 *                  w'(t) = u - w - b                                              *
 *                                                                                 *
 *                  I_{ion} = u(1-a)(u-a)-w                                        *
 *                  u(0)    = 1                                                    *
 *                  w(0)    = 0                                                    *
 *                                                                                 *
 * last change: 03.02.2020                                                         *
 * ------------------------------------------------------------------------------ **/

#ifndef MONODOMAIN_CPP

#include <iostream>
#include <tuple>
#include <metis.h>
#include <HighPerMeshes.hpp>
#include <HighPerMeshes/third_party/metis/Partitioner.hpp>
#include <HighPerMeshesDRTS.hpp>
//#include <HighPerMeshes/GPLStuff.hpp>
//#include <../../parser/WriteLoop.hpp>

//#include "monodomainFunctions.hpp"
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
//using Mesh           = HPM::mesh::Mesh<CoordinateType, HPM::entity::Simplex>; //for sequential case
using Mesh           = HPM::mesh::PartitionedMesh<CoordinateType, HPM::entity::Simplex>; //for distributed case
constexpr int dim    = Mesh::CellDimension;

/*-------------------------------------------------------------- (A) Functions: -------------------------------------------------------------------------------*/
template<typename MeshT, typename BufferT, typename LoopBodyT>
void setStartVector(const MeshT & mesh, BufferT & startVec, const float & startValLeft, const float & startValRight, const int & maxX, const int & maxY, LoopBodyT bodyObj);

template<typename MeshT, typename LoopBodyT, typename SigmaT>
auto getStiffnessmatrix(const MeshT & mesh, bool optOutput, LoopBodyT bodyObj, SigmaT sigma) -> Matrix;

template<typename MeshT, typename LoopBodyT>
auto getLumpedMassmatrix(const MeshT & mesh, bool optOutput, LoopBodyT bodyObj) -> Vector;

template<typename VectorT>
void testLumpedVec(const VectorT & lumpedM, const float & tol);

template<typename VectorT, typename BufferT, typename NodesT>
void assembleLumpedVec(VectorT & lumpedM, const BufferT & LSM, const NodesT & MeshEntityNodes);

template<typename MatrixT, typename BufferT, typename NodesT>
void assembleGlobalStiffnessMatrixPerCell(MatrixT & GSM, const BufferT & LSM, const NodesT & MeshEntityNodes);

template<typename BufferT, typename VectorT>
void ExplEuler(BufferT & vecOld, const VectorT & vecOldDeriv, const float & h);

template<typename BufferT>
auto Iion(const BufferT & u, const BufferT & w, const float & a) -> Vector;

template<typename BufferT, typename VectorT, typename MatrixT>
auto UDerivation(const BufferT & u, const MatrixT & Stiffnessmatrix, const VectorT & Iion) -> Vector;

template<typename BufferT/*,typename VectorT*/>
auto WDerivation(const BufferT & u, const BufferT/*VectorT*/ & w, const float & b) -> Vector;

template <typename MatrixT>
void writeMatrixToMatlab(MatrixT const& mat, std::string const& basename);

template <typename MeshT, typename T>
void writeVTKOutput(const MeshT & mesh, std::string const & filename, T result, std::string const nameOfResult);

template <typename MeshT, typename T>
void writeVTKOutput2DTime(const MeshT & mesh, std::string const & filename, T result, std::string const nameOfResult);

/*----------------------------------------------------------------- MAIN --------------------------------------------------------------------------------------*/
int main(int argc, char** argv)
{
    /*------------------------------------------(1) Set run-time system and read mesh information: ------------------------------------------------------------*/
    // sequential case
//    HPM::drts::Runtime hpm { HPM::GetBuffer{} };
//    HPM::SequentialDispatcher body;
//    HPM::auxiliary::ConfigParser CFG("config.cfg");
//    std::string meshFile = CFG.GetValue<std::string>("MeshFile");
//    const Mesh mesh = Mesh::template CreateFromFile<HPM::auxiliary::AmiraMeshFileReader>(meshFile);

    // distrubuted case
    HPM::drts::Runtime<HPM::GetDistributedBuffer<>, HPM::UsingDistributedDevices> hpm({}, std::forward_as_tuple(argc, argv));
    HPM::DistributedDispatcher body{hpm.gaspi_context, hpm.gaspi_segment, hpm};
    HPM::auxiliary::ConfigParser CFG("config.cfg");
    std::string meshFile = CFG.GetValue<std::string>("MeshFile");
    const Mesh mesh      = Mesh::template CreateFromFile<HPM::auxiliary::AmiraMeshFileReader, ::HPM::mesh::MetisPartitioner>
                           (meshFile, {hpm.GetL1PartitionNumber(), hpm.GetL2PartitionNumber()}, hpm.gaspi_runtime.rank().get());


    /*------------------------------------------(2) Create monodomain problem ---------------------------------------------------------------------------------*/
    int numIt   = 20000;
    float h     = 0.01;
    float a     = 0.3;
    float b     = 0.7;
    float sigma = 1;

    float u0L  = 1.F; float u0R  = 0.F;
    float w0L  = 0.F; float w0R  = 1.F;

    bool outputOpt = false;

    int numNodes = mesh.template GetNumEntities<0>();
    int maxX = std::ceil(std::sqrt(numNodes)/4);
    int maxY = std::ceil(std::sqrt(numNodes));

    HPM::Buffer<float, Mesh, Dofs<1, 0, 0, 0>> u(mesh);
    setStartVector(mesh, u, u0L, u0R, maxX*2, maxY, body);
    // Open a file stream for each distributed context
    std::ofstream file { std::string { "OutU0_" } +  std::to_string(hpm.gaspi_context.rank().get()) + ".txt" };
    auto AllNodes { mesh.GetEntityRange<0>() } ;
    //std::mutex mutex;

    /*body.Execute(
      iterator::Range{ 0 },
      WriteLoop(mutex, file, AllNodes, u, EveryNthStep(1))
    );*/

    HPM::Buffer<float, Mesh, Dofs<1, 0, 0, 0>> w(mesh);
    setStartVector(mesh, w, w0L, w0R, maxX, maxY, body);

    Vector u_deriv;
    Vector w_deriv;
    Vector f_i;

    Vector lumpedM      = getLumpedMassmatrix(mesh, outputOpt, body);
    float tol           = 0.1;
    testLumpedVec(lumpedM, tol);

    Matrix A            = getStiffnessmatrix(mesh, false, body, sigma);

//    // Open a file stream for each distributed context
//    std::ofstream w { std::string { "Out_" } +  std::to_string(hpm.gaspi_context.rank().get()) + ".txt" };

//    std::mutex mutex;

//    // Pass WriteLoop to the rts
//    body.Execute(
//                WriteLoop(mutex, w, AllCells, buffer)
//                );


    // check if startvector was set correctly
    //std::stringstream s;
    //s << 0;
    //std::string name = "test" + s.str();
    //writeVTKOutput(mesh, name, u, "resultU");
    //writeVTKOutput2DTime(mesh, name, u, "resultU");

    for (int j = 0; j < numIt; ++j)
    {
        f_i     = Iion(u, w, a);
        u_deriv = UDerivation(u, A, f_i);
        w_deriv = WDerivation(u, w, b);
        ExplEuler(u, u_deriv, h);
        ExplEuler(w, w_deriv, h);
        setStartVector(mesh, w, 0, 0, maxX, maxY, body);

        if ((j+1)%100 == 0)
        {
            std::stringstream s;
            s << j+1;
            std::string name = "A_test_n1_d1_WithoutW_" + s.str();
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
void setStartVector(const MeshT & mesh, BufferT & startVec, const float & startValLeft, const float & startValRight, const int & maxX, const int & maxY, LoopBodyT bodyObj)
{
    auto nodes { mesh.template GetEntityRange<0>() };

    bodyObj.Execute(HPM::ForEachEntity(
                  nodes,
                  std::tuple(Read(Node(startVec))),
                  [&](auto const& node, const auto& iter, auto& lvs)
    {
        auto coords = node.GetTopology().GetVertices();
        if ( (coords[0][0] <= maxX) && (coords[0][1] <= maxY) )
            startVec[node.GetTopology().GetIndex()] = startValLeft;
        else
            startVec[node.GetTopology().GetIndex()] = startValRight;
    }));

    return;
}

//!
//! \brief Define storage buffers for stiffness matrix in form of local matrices to be scattered.
//!
//! \param mesh meshfile
//! \param localMatrices local matrices to be scattered
//! \param GSM global matrix
//! \param optOutput if true, output of vector localMatrices
//! \param bodyObj object of loop body
//!
template<typename MeshT, typename LoopBodyT, typename SigmaT>
auto getStiffnessmatrix(const MeshT & mesh, bool optOutput, LoopBodyT bodyObj, SigmaT sigma) -> Matrix
{
    HPM::Buffer<float, Mesh, Dofs<3, 0, 0, 0>> localMatrices(mesh);
    Matrix GSM; GSM.resize(mesh.template GetNumEntities<0>(), Vector(mesh.template GetNumEntities<0>()));

    auto cells { mesh.template GetEntityRange<2>() };

    bodyObj.Execute(HPM::ForEachEntity(
                        cells,
                        std::tuple(ReadWrite(Node(localMatrices))),
                        [&](auto const& cell, const auto& iter, auto& lvs)
    {
        auto& localMatrices   = HPM::dof::GetDofs<HPM::dof::Name::Node>(std::get<0>(lvs));
        const auto& gradients = GetGradients2DP1();
        const auto& nodeIdSet = cell.GetTopology().GetNodeIndices();

        auto tmp   = cell.GetGeometry().GetJacobian();
        float detJ = tmp.Determinant(); detJ = std::abs(detJ);
        auto inv   = tmp.Invert();
        auto invJT = inv.Transpose();

        for (int col = 0; col < dim+1; ++col)
        {
            auto gc = invJT * gradients[col];
            for (int row = 0; row < dim+1; ++row)
            {
                auto gr                  = invJT * gradients[row];
                localMatrices[row][col]  =  (-1) * sigma * detJ * 0.5 * (gc*gr);
            }
        }

        assembleGlobalStiffnessMatrixPerCell(GSM, localMatrices, nodeIdSet);
    }));

    if (optOutput)
        outputMat(GSM, "GSM",GSM.size(),GSM[0].size());

    return GSM;
}

//!
//! \brief Define storage buffers for stiffness matrix in form of local matrices to be scattered.
//!
//! \param mesh meshfile
//! \param localMatrices local matrices to be scattered
//! \param GSM global matrix
//! \param optOutput if true, output of vector localMatrices
//! \param bodyObj object of loop body
//!
template<typename MeshT, typename LoopBodyT>
auto getLumpedMassmatrix(const MeshT & mesh, bool optOutput, LoopBodyT bodyObj) -> Vector
{
    HPM::Buffer<float, Mesh, Dofs<3, 0, 0, 0>> localMatrices(mesh);
    Vector lumpedM; lumpedM.resize(mesh.template GetNumEntities<0>());
    auto cells { mesh.template GetEntityRange<2>() };

    bodyObj.Execute(HPM::ForEachEntity(
                        cells,
                        std::tuple(ReadWrite(Node(localMatrices))),
                        [&](auto const& cell, const auto& iter, auto& lvs)
    {
        auto& localMatrices   = HPM::dof::GetDofs<HPM::dof::Name::Node>(std::get<0>(lvs));
        const auto& nodeIdSet = cell.GetTopology().GetNodeIndices();
        auto tmp              = cell.GetGeometry().GetJacobian();
        float detJ            = std::abs(tmp.Determinant());

        for (int col = 0; col < dim+1; ++col)
            for (int row = 0; row < dim+1; ++row)
            {
                if ( row == col)
                    localMatrices[row][col] += detJ * 1/12 ;
                else
                    localMatrices[row][col] += detJ * 1/24;
            }

        if (optOutput)
            outputMat(localMatrices, "Local matrix per tetraeder",dim+1,dim+1);

        assembleLumpedVec(lumpedM, localMatrices, nodeIdSet);
    }));

    return lumpedM;
}

//!
//! \brief Create a global matrix by assembling local matrices.
//!
//! \param GSM global stiffness matrix
//! \param LSM local stiffness matrix
//! \param MeshEntityNodes global id's of element nodes (here: cell nodes)
//! \param numRows number of LSM rows
//! \param numCols number of LSM cols
//!
template<typename VectorT, typename BufferT, typename NodesT>
void assembleLumpedVec(VectorT & lumpedM, const BufferT & LSM, const NodesT & MeshEntityNodes)
{
    for (int i = 0; i < dim+1; ++i)
        for (int j = 0; j < dim+1; ++j)
            lumpedM[MeshEntityNodes[i]] += LSM[i][j];
    return;
}

//!
//! \brief Create a global stiffness matrix by assembling local matrices.
//!
//! \param GSM global stiffness matrix
//! \param LSM local stiffness matrix
//! \param MeshEntityNodes global id's of element nodes
//!
template<typename MatrixT, typename BufferT, typename NodesT>
void assembleGlobalStiffnessMatrixPerCell(MatrixT & GSM, const BufferT & LSM, const NodesT & MeshEntityNodes)
{
    for (int i = 0; i < dim+1; ++i)
        for (int j = 0; j < dim+1; ++j)
            GSM[MeshEntityNodes[i]][MeshEntityNodes[j]] += LSM[i][j];

    return;
}

//!
//! \brief Explizit Euler algorithm.
//!
template<typename BufferT, typename VectorT>
void ExplEuler(BufferT & vecOld, const VectorT & vecOldDeriv, const float & h)
{
    for (int i = 0; i < vecOld.GetSize(); ++i )
        vecOld[i] += h*vecOldDeriv[i];

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
template<typename BufferT, typename VectorT, typename MatrixT>
auto UDerivation(const BufferT & u, const MatrixT & Stiffnessmatrix, const VectorT & Iion) -> Vector
{
    Vector u_deriv; u_deriv.resize(u.GetSize());

    for (int i = 0; i < u.GetSize(); ++i)
    {
        double Du_ij = 0;
        for (int j = 0; j < u.GetSize(); ++j)
            Du_ij += Stiffnessmatrix[i][j] * u[j];
        u_deriv[i] += Du_ij + Iion[i];
    }

    return u_deriv;
}

//!
//! \brief Compute derivation of w at time step t.
//!
template<typename BufferT/*,typename VectorT*/>
auto WDerivation(const BufferT & u, const /*VectorT*/BufferT & w, const float & b) -> Vector
{
    Vector w_deriv; w_deriv.resize(w.GetSize());
    float eps = 0.5;

    for (int i = 0; i < w.GetSize(); ++i)
        w_deriv[i] = eps*(u[i]-b*w[i]);

    return w_deriv;
}

//!
//! \brief Check lumped mass matrix.
//!
template<typename VectorT>
void testLumpedVec(const VectorT & lumpedM, const float & tol)
{
    double val = 0;
    for (int i = 0; i < lumpedM.size(); ++i)
        val += lumpedM[i];

    val = val/lumpedM.size();
    if ( val < (1-tol))
        std::cout << "Ratio of lumping Massmatrix is:  " << val << ". Ratio could be too small!" << std::endl;

    return;
}

//!
//! \brief Creates a matlab file including the matrix mat.
//!
//! \param mat
//! \param basename
//!
template <typename MatrixT>
void writeMatrixToMatlab(MatrixT const& mat, std::string const& basename)
{
  std::string fname = basename + ".m";
  std::ofstream f(fname.c_str());

  f << "function [A] = " << basename << '\n'
      << " A = [\n";

  for (size_t i=0; i<mat.size(); ++i)
  {
      for (size_t j=0; j<mat[0].size(); ++j)
         f << mat[i][j] << ' ';
      f << '\n';
  }
  f << "];\nA = A';\n";

  std::cout<<"Write matrix information to file "<<basename<<std::endl;
}

//!
//! \brief Creates a vtk file
//!
//! \param mesh
//! \param filename
//! \param result
//! \param nameOfResult
//!
template <typename MeshT, typename T>
void writeVTKOutput(const MeshT & mesh, std::string const & filename, T result, std::string const nameOfResult)
{
    int numNodes = mesh.template GetNumEntities<0>();
    int numberOfCells  = mesh.template GetNumEntities<dim>();
    int cellType;
    if (dim == 3)
        cellType = 10; // tetrahedrons
    else if (dim == 2)
        cellType = 5; // triangles
    else
        cellType = 3; // line

    std::string fname = filename + ".vtu";
    std::ofstream f(fname.c_str());

    f << "<?xml version=\"1.0\"?>" << '\n'
      << "<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\">" << '\n'
      << "  <UnstructuredGrid>" << '\n'
      << "      <Piece NumberOfPoints=\"" << numNodes << "\" NumberOfCells=\"" << numberOfCells << "\">" << '\n'
      << "          <Points>" << '\n'
      << "              <DataArray type=\"Float32\" Name=\"Coordinates\" NumberOfComponents=\""<<dim+(3-dim)<<"\" format=\"ascii\">" << '\n'
      << "              ";

    //add node coordinates to file
    for (const auto& node : mesh.template GetEntities<0>() )
    {
        auto nodeCoords = node.GetTopology().GetVertices();
        int nodeID      = node.GetTopology().GetIndex();
        for (int j = 0; j < dim+1; ++j)
        {
            if (j < dim)
                f << nodeCoords[0][j] << ' ';
            else
                f << 0 << ' ';
        }
        if (((dim+1)*(nodeID+1)) % 12 == 0)
            f << '\n';
        if ((((dim+1)*(nodeID+1)) % 12 == 0) && (nodeID+1)!=numNodes)
            f << "              ";
    }

    if (((dim+1)*(numNodes)) % 12 != 0)
        f << '\n';

    f << "              </DataArray>" << '\n'
      << "          </Points>" << '\n'
      << "          <Cells>" << '\n'
      << "              <DataArray type=\"Int32\" Name=\"connectivity\" NumberOfComponents=\"1\" format=\"ascii\">" << '\n'
      << "              ";

    //add cell information (nodes of each cell using node id)
    int numNodesPerCell;
    bool setInfo = false;
    for (const auto& cell : mesh.template GetEntities<dim>() )
    {
        auto nodeIDs = cell.GetTopology().GetNodeIndices();
        int  cellID  = cell.GetTopology().GetIndex();
        if (!setInfo)
        {
            numNodesPerCell = nodeIDs.size();
            setInfo = true;
        }
        for (int j = 0; j < nodeIDs.size(); ++j)
            f << nodeIDs[j] << ' ';
        if ((numNodesPerCell*(cellID+1)) % 12 == 0)
            f << '\n';
        if ((numNodesPerCell*(cellID+1) % 12 == 0) && (cellID+1)!=numberOfCells)
            f << "              ";
    }

    if ((numNodesPerCell*(numberOfCells)) % 12 != 0)
        f << '\n';

    f << "              </DataArray>" << '\n'
      << "              <DataArray type=\"Int32\" Name=\"offsets\" NumberOfComponents=\"1\" format=\"ascii\">" << '\n'
      << "              ";

    //add cell information (connectivity -> each cell consists of certain nodes)
    for (int i = 0; i < numberOfCells; ++i)
    {
        f << (i+1)*numNodesPerCell << ' ';
        if ((i+1) % 12 == 0)
            f << '\n';
        if ((i+1) % 12 == 0 && (i+1)!=numberOfCells)
            f << "              ";
    }

    if ((numberOfCells+1) % 12 != 0)
        f << '\n';

    f << "              </DataArray>" << '\n'
      << "              <DataArray type=\"UInt8\" Name=\"types\" NumberOfComponents=\"1\" format=\"ascii\">" << '\n'
      << "              ";

    //add cell information (connectivity -> each cell consists of certain nodes)
    for (int i = 0; i < numberOfCells; ++i)
    {
        f << cellType << ' ';
        if ((i+1) % 12 == 0)
            f << '\n';
        if ((i+1) % 12 == 0 && (i+1)!=numberOfCells)
            f << "              ";
    }

    if ((numberOfCells+1) % 12 != 0)
        f << '\n';

    f << "              </DataArray>" << '\n'
      << "          </Cells>" << '\n'
      << "          <CellData>" << '\n'
      << "          </CellData>" << '\n'
      << "          <PointData Scalars=\""<<nameOfResult<<"\">" << '\n'
      << "              <DataArray type=\"Float64\" Name=\""<<nameOfResult<<"\" NumberOfComponents=\"1\" format=\"ascii\">" << '\n'
      << "              ";

    for (int i = 0; i < numNodes; i++)
    {
        f << result[i] << ' ';

        if ((i+1)%12==0)
            f << '\n';
        if ((i+1) % 12 == 0 && (i+1)!=numNodes)
            f << "              ";
    }

    if (numNodes % 12 != 0)
        f << '\n';

    f << "              </DataArray>" << '\n'
      << "          </PointData>" << '\n'
      << "      </Piece>" << '\n'
      << "  </UnstructuredGrid>" << '\n'
      << "</VTKFile>" << '\n';

    std::cout<<"Write file to "<<filename<<".vtu"<<std::endl;
}


//!
//! \brief Creates a vtk file, with t as z-axis
//!
//! \param mesh
//! \param filename
//! \param result
//! \param nameOfResult
//!
template <typename MeshT, typename T>
void writeVTKOutput2DTime(const MeshT & mesh, std::string const & filename, T result, std::string const nameOfResult)
{
    int numNodes = mesh.template GetNumEntities<0>();
    int numberOfCells  = mesh.template GetNumEntities<dim>();
    int cellType;
    if (dim == 3)
        cellType = 10; // tetrahedrons
    else if (dim == 2)
        cellType = 5; // triangles
    else
        cellType = 3; // line

    std::string fname = filename + ".vtu";
    std::ofstream f(fname.c_str());

    f << "<?xml version=\"1.0\"?>" << '\n'
      << "<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\">" << '\n'
      << "  <UnstructuredGrid>" << '\n'
      << "      <Piece NumberOfPoints=\"" << numNodes << "\" NumberOfCells=\"" << numberOfCells << "\">" << '\n'
      << "          <Points>" << '\n'
      << "              <DataArray type=\"Float32\" Name=\"Coordinates\" NumberOfComponents=\""<<dim+(3-dim)<<"\" format=\"ascii\">" << '\n'
      << "              ";

    //add node coordinates to file
    for (const auto& node : mesh.template GetEntities<0>() )
    {
        auto nodeCoords = node.GetTopology().GetVertices();
        int nodeID      = node.GetTopology().GetIndex();
        for (int j = 0; j < dim+1; ++j)
        {
            if (j < dim)
                f << nodeCoords[0][j] << ' ';
            else
                f << result[nodeID] << ' ';
        }
        if (((dim+1)*(nodeID+1)) % 12 == 0)
            f << '\n';
        if ((((dim+1)*(nodeID+1)) % 12 == 0) && (nodeID+1)!=numNodes)
            f << "              ";
    }

    if (((dim+1)*(numNodes)) % 12 != 0)
        f << '\n';

    f << "              </DataArray>" << '\n'
      << "          </Points>" << '\n'
      << "          <Cells>" << '\n'
      << "              <DataArray type=\"Int32\" Name=\"connectivity\" NumberOfComponents=\"1\" format=\"ascii\">" << '\n'
      << "              ";

    //add cell information (nodes of each cell using node id)
    int numNodesPerCell;
    bool setInfo = false;
    for (const auto& cell : mesh.template GetEntities<dim>() )
    {
        auto nodeIDs = cell.GetTopology().GetNodeIndices();
        int  cellID  = cell.GetTopology().GetIndex();
        if (!setInfo)
        {
            numNodesPerCell = nodeIDs.size();
            setInfo = true;
        }
        for (int j = 0; j < nodeIDs.size(); ++j)
            f << nodeIDs[j] << ' ';
        if ((numNodesPerCell*(cellID+1)) % 12 == 0)
            f << '\n';
        if ((numNodesPerCell*(cellID+1) % 12 == 0) && (cellID+1)!=numberOfCells)
            f << "              ";
    }

    if ((numNodesPerCell*(numberOfCells)) % 12 != 0)
        f << '\n';

    f << "              </DataArray>" << '\n'
      << "              <DataArray type=\"Int32\" Name=\"offsets\" NumberOfComponents=\"1\" format=\"ascii\">" << '\n'
      << "              ";

    //add cell information (connectivity -> each cell consists of certain nodes)
    for (int i = 0; i < numberOfCells; ++i)
    {
        f << (i+1)*numNodesPerCell << ' ';
        if ((i+1) % 12 == 0)
            f << '\n';
        if ((i+1) % 12 == 0 && (i+1)!=numberOfCells)
            f << "              ";
    }

    if ((numberOfCells+1) % 12 != 0)
        f << '\n';

    f << "              </DataArray>" << '\n'
      << "              <DataArray type=\"UInt8\" Name=\"types\" NumberOfComponents=\"1\" format=\"ascii\">" << '\n'
      << "              ";

    //add cell information (connectivity -> each cell consists of certain nodes)
    for (int i = 0; i < numberOfCells; ++i)
    {
        f << cellType << ' ';
        if ((i+1) % 12 == 0)
            f << '\n';
        if ((i+1) % 12 == 0 && (i+1)!=numberOfCells)
            f << "              ";
    }

    if ((numberOfCells+1) % 12 != 0)
        f << '\n';

    f << "              </DataArray>" << '\n'
      << "          </Cells>" << '\n'
      << "          <CellData>" << '\n'
      << "          </CellData>" << '\n'
      << "          <PointData Scalars=\""<<nameOfResult<<"\">" << '\n'
      << "              <DataArray type=\"Float64\" Name=\""<<nameOfResult<<"\" NumberOfComponents=\"1\" format=\"ascii\">" << '\n'
      << "              ";

    for (int i = 0; i < numNodes; i++)
    {
        f << result[i] << ' ';

        if ((i+1)%12==0)
            f << '\n';
        if ((i+1) % 12 == 0 && (i+1)!=numNodes)
            f << "              ";
    }

    if (numNodes % 12 != 0)
        f << '\n';

    f << "              </DataArray>" << '\n'
      << "          </PointData>" << '\n'
      << "      </Piece>" << '\n'
      << "  </UnstructuredGrid>" << '\n'
      << "</VTKFile>" << '\n';

    std::cout<<"Write file to "<<filename<<".vtu"<<std::endl;
}
