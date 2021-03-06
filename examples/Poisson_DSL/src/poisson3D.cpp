/** ------------------------------------------------------------------------------ *
 * author(s)   : Merlind Schotte (schotte@zib.de)                                  *
 * institution : Zuse Institute Berlin (ZIB)                                       *
 * project     : HighPerMeshes (BMBF)                                              *
 *                                                                                 *
 * Description:                                                                    *
 * Implementation of a simple Poisson problem using the HighPerMeshes DSL. This is *
 * intended to check the completeness and usability of the DSL.                    *
 * Features that differ from the MIDG2 example:                                    *
 * - degrees of freedom associated to vertices (instead of cells)                  *
 * - assembly of stiffness matrix (not only right-hand sides)                      *
 * - mesh in 3D (could be postponed - Poisson problem can be easily defined on 2D  *
 *   triangles as well)                                                            *
 * - solvers (using self implemented solvers)                                      *
 *                                                                                 *
 * Poisson equation    : div(D \Nabla u) = f        on \Omega                      *
 * with D as diffusion tensor and f = 1                                            *
 * Boundary conditions :  u_{D} = 0                 on \Omega_{D}                  *
 *                        u_{N} = n^{T} \Nabla u    on \Omega_{N}                  *
 *                                                                                 *
 * last change: 21.11.2019                                                         *
 * ------------------------------------------------------------------------------ **/

#ifndef POISSON_CPP

#include <iostream>
#include <tuple>
#include <metis.h>
#include <HighPerMeshes.hpp>
#include <HighPerMeshes/third_party/metis/Partitioner.hpp>
#include <HighPerMeshesDRTS.hpp>

#include <../examples/Functions/outputWriter.hpp>
#include <../examples/Functions/simplexGradients.hpp>
#include <../examples/Functions/solver.hpp>
#include <../examples/Functions/mathFunctions.hpp>

class DEBUG;
using namespace HPM;
using namespace ::HPM::auxiliary;
template <std::size_t ...I>
using Dofs           = HPM::dataType::Dofs<I...>;
using Vector         = std::vector<float>;
using Matrix         = std::vector<Vector>;
using CoordinateT    = HPM::dataType::Vec<float,3>;
using Mesh           = HPM::mesh::PartitionedMesh<CoordinateT, HPM::entity::Simplex>;
constexpr int dim    = Mesh::CellDimension;

/*-------------------------------------------------------------- (A) Functions: -------------------------------------------------------------------------------*/
template<typename MeshT, typename TupleT, typename PairT>
void CreateBoundaryConditionTuple(const MeshT & mesh, const TupleT & boundaryConditions, PairT & BCPerFaceId, const bool optOutput);
template<typename MeshT, typename TupleT, typename PairT>
void CreateMaterialTuple(const MeshT & mesh, const TupleT & materialsPerCell, PairT & materialsPerCellId, const bool optOutput);

template<typename MeshT, typename BufferT, typename ItLoopBodyObjT>
void SetRHS(const MeshT & mesh, BufferT & rhs, bool optOutput, ItLoopBodyObjT bodyObj);
template<typename MeshT, typename BufferT, typename MatrixT, typename ItLoopBodyObjT, typename PairT>
void SetGSM(const MeshT & mesh, BufferT & localMatrices, MatrixT & GSM, bool optOutput, ItLoopBodyObjT bodyObj, PairT materialsPerCellId);
template<typename MatrixT, typename BufferT, typename NodesT>
void assembleGlobalStiffnessMatrixPerCell(MatrixT & GSM, const BufferT & LSM, const NodesT & MeshEntityNodes, const int & numRows, const int & numCols);

template<typename MeshT, typename BufferT, typename PairT, typename ItLoopBodyObjT>
void SetBoundaryConditions(const MeshT & mesh, BufferT & rhs, bool optOutput, PairT & BCPerFaceId, ItLoopBodyObjT bodyObj);
template<typename FaceT, typename RhsT>
void SetHomogeneousDirichlet(FaceT face, RhsT rhs);
template<typename FaceT, typename RhsT>
void SetInhomogeneousNeumann(FaceT face, RhsT rhs);

template<typename TupleT, typename BCIdT>
void CreateReducedBoundaryNodeIDSet(const TupleT & boundaryConditions, const BCIdT & bcId, std::vector<BCIdT> & boundaryNodes);
template<typename RhsT, typename BCIdT>
void CreateReducedRHS(const RhsT & rhs, Vector & reducedRHS, const std::vector<BCIdT> & homDirichletNodes, bool optOutput);
template<typename BCIdT>
void CreateReducedGSM(const Matrix & GSM, Matrix & reducedGSM, const std::vector<BCIdT> & homDirichletNodes, bool optOutput);
template<typename ElementT, typename T>
auto FindElement(const ElementT & element, const T & set) -> bool;

/*------------------------new functions------------------------------------*/

template<typename MeshT, typename BufferT, typename LoopbodyT, typename PairT, typename VecDslT>
BufferT GetResiduum(const MeshT & mesh, const BufferT & rhs, LoopbodyT bodyObj, const PairT & materials,
                    const VecDslT & x0,const std::vector<int> & homDirichletNodes);

template<typename MeshT, typename BufferT, typename LoopbodyT, typename PairT>
auto cgSolver(const MeshT & mesh, const BufferT & rhs, LoopbodyT bodyObj, const PairT & materials, const int & numSolverIt, const float & tol,
              const std::vector<int> & homDirichletNodes)->Vector;

template<typename MeshT, typename BufferT, typename LoopbodyT, typename PairT, typename VecDslT>
auto cgSolverMatFree(const MeshT & mesh, const BufferT & rhs, LoopbodyT bodyObj, const PairT & materials, VecDslT & x, const int & numSolverIt,
                     const float & tol, const std::vector<int> & homDirichletNodes)->VecDslT;

template<typename MeshT, typename VectorT, typename LoopbodyT, typename PairT, typename BufferT>
void computeMatrixVecProdukt(const MeshT & mesh, const VectorT & dk, LoopbodyT bodyObj, const PairT & materials,
                             BufferT & z, const std::vector<int> & homDirichletNodes);

template<typename MeshT, typename VectorT, typename LoopbodyT, typename PairT, typename BufferT>
void GetMatrixVecProduct(const MeshT & mesh, const VectorT & dk, LoopbodyT bodyObj, const PairT & materials,
                         BufferT & z, const std::vector<int> & homDirichletNodes);

template<typename NodeT, typename MeshT, typename PairT, typename BufferT>
void GetRowOfStiffnessMatrix(const NodeT& node, const MeshT& mesh, const PairT & materialsPerCellId,
                             BufferT & rowGSM, const std::vector<int> & homDirichletNodes);

/*----------------------------------------------------------------- MAIN --------------------------------------------------------------------------------------*/
int main(int argc, char** argv)
{
    HPM::drts::Runtime<HPM::GetDistributedBuffer<>, HPM::UsingDistributedDevices> hpm({}, std::forward_as_tuple(argc, argv));
    HPM::DistributedDispatcher body{hpm.gaspi_context, hpm.gaspi_segment, hpm};

    using ReaderType = HPM::auxiliary::AmiraMeshFileReader<CoordinateT, std::array<std::size_t, 4>>;
    using Pair       = std::vector<std::pair<std::size_t,std::size_t>>;

    using BoundaryType     = std::array<std::size_t, 1>;
    using BoundaryFaceType = std::array<std::size_t, 3>;
    using TupleBoundaries  = std::tuple<std::vector<BoundaryFaceType>, std::vector<BoundaryType>>;

    using CellType       = std::array<std::size_t, 4>;
    using MaterialType   = std::array<std::size_t, 1>;
    using TupleMaterials = std::tuple<std::vector<MaterialType>, std::vector<CellType>>;    

    /*------------------------------------------(1) Read mesh information: ------------------------------------------------------------------------------------*/
    HPM::auxiliary::ConfigParser CFG("config.cfg");
    std::string meshFile = CFG.GetValue<std::string>("MeshFile");
    const Mesh mesh      = Mesh::template CreateFromFile<HPM::auxiliary::AmiraMeshFileReader, ::HPM::mesh::MetisPartitioner>
                                         (meshFile, {hpm.GetL1PartitionNumber(), hpm.GetL2PartitionNumber()}, hpm.gaspi_runtime.rank().get());
    ReaderType reader;
     std::cout<<"hallo2"<<std::endl;
    // read boundary conditions (e.g. in Amira files @6=boundary faces as node index set and @5=boundary condition as id)
    TupleBoundaries boundaryConditions = reader.ReadGroups<BoundaryFaceType, BoundaryType>(meshFile, "@6", "@5", true, false);
    Pair BCPerFaceId; CreateBoundaryConditionTuple(mesh, boundaryConditions, BCPerFaceId, true);

    // read materials (e.g. in Amira files @4=material id's, @3=cells as node index set)
    TupleMaterials materialsPerCell = reader.ReadGroups<MaterialType, CellType>(meshFile, "@4", "@3", false, true);
    Pair materialsPerCellId; CreateMaterialTuple(mesh, materialsPerCell, materialsPerCellId, true);

    /*------------------------------------------(2) Create right-hand side (rhs) and global stiffness matrix (GSM) --------------------------------------------*/
    HPM::Buffer<float, Mesh, Dofs<1, 0, 0, 0, 0>> rhs(mesh);
    SetRHS(mesh, rhs, true, body);

    /**
      * - initialisations of local matrices (as buffer) and global stiffness matrix (no buffer)
      * - compute local matrices (per cell) and assemble global stiffness matrix (for the whole mesh)
    **/
    HPM::Buffer<float, Mesh, Dofs<4, 0, 0, 0, 0>> localMatrices(mesh);
    Matrix GSM; GSM.resize(mesh.template GetNumEntities<0>(), Vector(mesh.template GetNumEntities<0>()));
    SetGSM(mesh, localMatrices, GSM, true, body, materialsPerCellId);

    /*------------------------------------------(3) Add boundary conditions (homogeneous Dirichlet and inhomogeneous Neumann) ---------------------------------*/
    SetBoundaryConditions(mesh, rhs, true, BCPerFaceId, body);

    /*------------------------------------------(4) Create reduced system: ------------------------------------------------------------------------------------*/
    /**
      * - in this example/inputfile the hom. Dirichlet bc has the id 2
      * - the function CreateReducedBoundaryNodeIDSet() is needed because (here) homogeneous Dirichlet boundary conditions (bc) are integrated by deleting
      *   rows and columns and NOT by using a penalty factor (but it is also possible to add hom. Dirichlet bc using a penalty factor by changing the function
      *   SetBoundaryConditions() and commenting out the following part)
    **/
    const int homDirichletID = 2;  std::vector<int> homDirichletNodes;
    CreateReducedBoundaryNodeIDSet(boundaryConditions, homDirichletID, homDirichletNodes);

    Vector reducedRHS((mesh.template GetNumEntities<0>())-(homDirichletNodes.size()));
    CreateReducedRHS(rhs, reducedRHS, homDirichletNodes, true);

    Matrix ReducedGSM;
    ReducedGSM.resize(mesh.template GetNumEntities<0>()-homDirichletNodes.size(), Vector(mesh.template GetNumEntities<0>()-homDirichletNodes.size()));
    CreateReducedGSM(GSM, ReducedGSM, homDirichletNodes, true);

    /*------------------------------------------(5a) Solve system using inverse: ------------------------------------------------------------------------------*/
    Matrix InvMat = invertMatrix(ReducedGSM);
    outputMat(InvMat, "inverse using gauß-jordan", InvMat.size(), InvMat[0].size());
    auto solve1 = matrixVecProduct(InvMat,reducedRHS,reducedRHS.size());
    outputVec(solve1,"result using inverse (gauß-jordan)", 4);

    /*------------------------------------------(5b) Solve system using Jacobi algorithm: ---------------------------------------------------------------------*/
    Vector xStart; xStart = {0,0,0,0};
    Vector x_jacob = JacobiSolver(ReducedGSM, reducedRHS, xStart, 31, reducedRHS.size());
    outputVec(x_jacob,"jacobi", x_jacob.size());

    /*------NEW-----------------------------------------------------------------------*/
    writeVTKOutput(mesh, "test", x_jacob, "jacobi", homDirichletNodes, hpm);

    auto resultVec = cgSolver(mesh, rhs, body, materialsPerCellId, 30, 0.0001, homDirichletNodes);
    //auto vec = matrixFreeSolver(mesh, rhs, 23, body, materialsPerCellId);
    //outputVec(resultVec, "result of matrix free parts", 8);

    int const numNodes = 8;//mesh.template GetNumEntities<0>;
    HPM::dataType::Vec<float, numNodes> xk;
    for (int i = 0; i < numNodes; ++i) {xk[i]=0;}

    cgSolverMatFree(mesh, rhs, body, materialsPerCellId, xk, 3, 0.00001, homDirichletNodes);
    outputVec(xk, "xk", numNodes);

    // Open a file stream for each distributed context
    /*std::ofstream file { std::string { "ResultCGPoisson_" } +  std::to_string(hpm.gaspi_context.rank().get()) + ".txt" };
    auto AllNodes { mesh.GetEntityRange<0>() } ;
    std::mutex mutex;

    body.Execute(
      iterator::Range{ 0 },
      WriteLoop(mutex, file, AllNodes, xk, EveryNthStep(1))
    );*/


    return 0;
}
#endif

/*----------------------------------------------------------------- (A) Functions (Implementation): -----------------------------------------------------------*/
//!
//! \brief Create a pair of a boundaryface-id-container and boundarycondition-id-container.
//!
//! \param mesh meshfile
//! \param boundaryConditions tuple with faces as node index set and boundary conditions (BC) as id
//! \param BCPerFaceId container to put in the data
//! \param optOutput output option for the BCPerFaceId data tuple
//!
template<typename MeshT, typename TupleT, typename PairT>
void CreateBoundaryConditionTuple(const MeshT & mesh, const TupleT & boundaryConditions, PairT & BCPerFaceId, const bool optOutput)
{
    int numBoundaries              = std::get<1>(boundaryConditions).size();
    BCPerFaceId.resize(numBoundaries);

    int it = 0;
    for (const auto& element : mesh.GetEntities())
    {
        for (const auto& face : element.GetTopology().template GetEntities<dim-1>())
        {
            if (face.GetTopology().IsElementOfBoundary())
            {
                auto faceId = face.GetTopology().GetIndex();
                for ( int boundaryEntity = 0; boundaryEntity < numBoundaries; ++boundaryEntity )
                {
                    auto tmp = std::get<0>(boundaryConditions)[boundaryEntity];
                    int faceIndex = mesh.template GetIndex<2>(tmp);
                    if (faceId == faceIndex)
                    {
                        BCPerFaceId[it] = std::make_pair(face.GetTopology().GetIndex(),
                                                         std::get<1>(boundaryConditions)[boundaryEntity][0]);
                        ++it;
                    }
                }
            }
        }
    }

    if (optOutput)
    {
        std::cout<<"------ CreateBoundaryConditionTuple(): --------------"<<std::endl;
        for (int i = 0; i<numBoundaries; ++i)
            std::cout<<"FaceId:  "<<BCPerFaceId[i].first<< "\t" <<
                       "BCId  :  "<<BCPerFaceId[i].second<<std::endl;
    }

    return;
}

//!
//! \brief Create a pair of a material-id-container and cell-id-container.
//!
//! \param mesh meshfile
//! \param materialsPerCell tuple with materials as id and cells as node index set
//! \param materialsPerCellId container to put in the data
//! \param optOutput output option for the materialsPerCellId data tuple
//!
template<typename MeshT, typename TupleT, typename PairT>
void CreateMaterialTuple(const MeshT & mesh, const TupleT & materialsPerCell, PairT & materialsPerCellId, const bool optOutput)
{
    int numMaterials = std::get<0>(materialsPerCell).size();
    materialsPerCellId.resize(numMaterials);

    for (const auto& element : mesh.GetEntities())
    {
        int elementId = element.GetTopology().GetIndex();
        for (int i = 0; i < numMaterials; ++i)
        {
            int cellId = mesh.template GetIndex<3>(std::get<1>(materialsPerCell)[i]);
            if (elementId == cellId )
            {
                materialsPerCellId[elementId] = std::make_pair(elementId,std::get<0>(materialsPerCell)[elementId][0]);
                break;
            }
        }
    }

    if (optOutput)
    {
        std::cout<<"------ CreateMaterialTuple(): --------------"<<std::endl;
        for (int i = 0; i<numMaterials; ++i)
            std::cout<<"CellId:  "<<materialsPerCellId[i].first<<"\t"<<
                       "MaterialId    :  "<<materialsPerCellId[i].second<<std::endl;
    }

    return;
}

//!
//! \brief Define the right-hand side assembly. This is specialized for source density 1 (f_i=1) and P1 elements,
//!        i.e. it computes int_T x*1 * detJ dx on the unit tetrahedron.
//!
//! \param mesh unitcube splitted into 5 tetrahedrons
//! \param rhs right-hand side which should be assembled in this looptype as vector per tetrahedron
//! \param optOutput output option for rhs
//! \param bodyObj object of loop body
//!
template<typename MeshT, typename BufferT, typename ItLoopBodyObjT>
void SetRHS(const MeshT & mesh, BufferT & rhs, bool optOutput, ItLoopBodyObjT bodyObj)
{
    auto cells { mesh.template GetEntityRange<3>() };

    bodyObj.Execute(HPM::ForEachEntity(
                  cells,
                  std::tuple(ReadWrite(Node(rhs))),
                  [&](auto const& cell, const auto& iter, auto& lvs)
    {
        auto& rhs                 = std::get<0>(lvs);
        auto jacobianMat          = cell.GetGeometry().GetJacobian();
        double detJac             = jacobianMat.Determinant(); detJac = std::abs(detJac);
        const auto& node_indices  = cell.GetTopology().GetNodeIndices();

        for (const auto& node : cell.GetTopology().template GetEntities<0>())
        {
            int id = node.GetTopology().GetLocalIndex();
            rhs[id][0] += detJac/24;
        }
    }));

    if (optOutput)
        outputVec(rhs,  "Buffer RHS in function SetRHS()", rhs.GetSize());

    return;
}

//!
//! \brief Define the boundary condition's (bc) contribution to the right-hand side.
//!
//! We impose homogeneous Dirichlet bc by u_D = 0 and inhomogeneous Neumann bc by a random pull direction,
//! e.g. int_{\Omega_N} g*u.
//!
//! \param mesh unitcube
//! \param rhs right-hand side
//! \param optOutput output option for rhs
//! \param BCPerFaceId pair of face id and boundary condition
//! \param bodyObj object of loop body
//!
template<typename MeshT, typename BufferT, typename PairT, typename ItLoopBodyObjT>
void SetBoundaryConditions(const MeshT & mesh, BufferT & rhs, bool optOutput, PairT & BCPerFaceId, ItLoopBodyObjT bodyObj)
{
    auto faces { mesh.template GetEntityRange<2>([] (const auto& entity) { return entity.GetTopology().IsElementOfBoundary(); }) };

    bodyObj.Execute(HPM::ForEachEntity(
                  faces,
                  std::tuple(ReadWrite(Node(rhs))),
                  [&](auto const& face, const auto& iter, auto& lvs)
    {
        std::vector<std::pair<std::size_t,std::size_t>>::iterator it;
        int i = 0;

        for (it = BCPerFaceId.begin(); it != BCPerFaceId.end(); ++it)
        {
            if (BCPerFaceId[i].first == face.GetTopology().GetIndex())
            {
                auto& rhs = std::get<0>(lvs);

                if(BCPerFaceId[i].second == 1) // inhom Neumann
                    SetInhomogeneousNeumann(face, rhs);
                else if(BCPerFaceId[i].second == 2) //homogeneous Dirichlet
                    SetHomogeneousDirichlet(face, rhs);
                else
                    std::cout<<"Boundary Condition is not defined."<<std::endl;
            }
            ++i;
        }
    })
    );

    if (optOutput)
        outputVec(rhs, "Rhs with boundary conditions", rhs.GetSize());

    return;
}

//!
//! \brief Define storage buffers for stiffness matrix in form of local matrices to be scattered.
//!        There is one local matrix per cell, which has as many entries as the product of the two
//!        involved FE spaces each have on the (here) CellClosure.
//!
//! \param mesh meshfile
//! \param localMatrices local matrices to be scattered
//! \param GSM global stiffness matrix
//! \param optOutput if true, output of vector localMatrices
//! \param bodyObj object of loop body
//!
template<typename MeshT, typename BufferT, typename MatrixT, typename ItLoopBodyObjT, typename PairT>
void SetGSM(const MeshT & mesh, BufferT & localMatrices, MatrixT & GSM, bool optOutput, ItLoopBodyObjT bodyObj, PairT materialsPerCellId)
{
    auto cells { mesh.template GetEntityRange<3>() };

    bodyObj.Execute(HPM::ForEachEntity(
                  cells,
                  std::tuple(ReadWrite(Node(localMatrices))),
                  [&](auto const& cell, const auto& iter, auto& lvs)
    {
        const int nrows = dim+1; const int ncols = dim+1;
        auto& localMatrices   = std::get<0>(lvs);
        const auto& gradients = GetGradientsDSL();
        const auto& nodeIdSet = cell.GetTopology().GetNodeIndices();

        auto tmp   = cell.GetGeometry().GetJacobian();
        float detJ = tmp.Determinant(); detJ = std::abs(detJ);
        auto inv   = tmp.Invert();
        auto invJT = inv.Transpose();

        // Material information is integrate as diffusion tensor D = v * I with v as random scalar value.
        // For example: v = 3 if material id is 1 and v = 6 if material id is 2
        float value;
        if (materialsPerCellId[cell.GetTopology().GetIndex()].second == 1)
            value = 3;
        else if (materialsPerCellId[cell.GetTopology().GetIndex()].second == 2)
            value = 6;
        else
            value = 1; // default

        for (int col = 0; col < ncols; ++col)
        {
            auto gc = invJT*gradients[col]*value;
            for (int row = 0; row < nrows; ++row)
            {
                auto gr                  = invJT* gradients[row];
                localMatrices[row][col]  = detJ/6 * (gc*gr);
            }
        }

        if (optOutput)
            outputMat(localMatrices, "Local stiffnessmatrix per tetrahedron",dim+1,dim+1);

        assembleGlobalStiffnessMatrixPerCell(GSM, localMatrices, nodeIdSet, dim+1, dim+1);
    }));

    if (optOutput)
        outputMat(GSM,"Global stiffness matrix",GSM.size(),GSM[0].size());

    return;
}

//!
//! \brief Create a global stiffness matrix by assembling local matrices.
//!
//! \param GSM global stiffness matrix
//! \param LSM local stiffness matrix
//! \param MeshEntityNodes global id's of element nodes (here: cell nodes)
//! \param numRows number of LSM rows
//! \param numCols number of LSM cols
//!
template<typename MatrixT, typename BufferT, typename NodesT>
void assembleGlobalStiffnessMatrixPerCell
(MatrixT & GSM, const BufferT & LSM, const NodesT & MeshEntityNodes, const int & numRows, const int & numCols)
{
    for (int i = 0; i < numRows; ++i)
        for (int j = 0; j < numCols; ++j)
            GSM[MeshEntityNodes[i]][MeshEntityNodes[j]] += LSM[i][j];

    return;
}

//!
//! \brief Add homogeneous Dirichlet bc by integrating u_D = 0 in rhs.
//!
//! \param face boundary face with Dirichlet bc
//! \param rhs right-hand side, which is edit
//!
template<typename FaceT, typename RhsT>
void SetHomogeneousDirichlet(FaceT face, RhsT rhs)
{
    for (const auto& node : face.GetTopology().template GetEntities<0>())
        rhs[node.GetTopology().GetLocalIndex()][0] = 0;

    return;
}

//!
//! \brief Add inhomogeneous Neumann bc to rhs.
//!
//! \param face boundary face with Neumann bc
//! \param rhs right-hand side, which is edit
//!
//! Inhom. Neumann:
//!  --> traction: t = force / area    with area = detJac
//!  --> g = t * detJac = force
//!  --> G = g * u; G' = g * phi_i
//!  --> int_{\Omega_{N}} G' = traction / 6
template<typename FaceT, typename RhsT>
void SetInhomogeneousNeumann(FaceT face, RhsT rhs)
{
    float traction = -4.34; // e.g. traction should be 4.34[N] (random value)
    for (const auto& node : face.GetTopology().template GetEntities<0>() )
        rhs[node.GetTopology().GetLocalIndex()][0] += traction/6;

    return;
}

//!
//! \brief Create a pair of a boundaryface-id-container and boundarycondition-id-container.
//!
//! \param mesh meshfile
//! \param boundaryConditions tuple with faces as node index set and boundary conditions (BC) as id
//! \param bcId id of searched boundary condition (e.g. homogeneous Dirichlet = 2)
//! \param boundaryNodes container to put in the boundary nodes
//! \param optOutput output option for the BCPerFaceId data tuple
//!
template<typename TupleT, typename BCIdT>
void CreateReducedBoundaryNodeIDSet(const TupleT & boundaryConditions, const BCIdT & bcId, std::vector<BCIdT> & boundaryNodes)
{
    for (int i = 0; i < std::get<1>(boundaryConditions).size(); ++i)
        if (std::get<1>(boundaryConditions)[i][0] == bcId)
            for (int j = 0; j < std::get<0>(boundaryConditions)[i].size(); ++j)
            {
                bool foundElement = FindElement(std::get<0>(boundaryConditions)[i][j], boundaryNodes);
                if (!foundElement)
                    boundaryNodes.push_back(std::get<0>(boundaryConditions)[i][j]);
            }

    std::sort(boundaryNodes.begin(), boundaryNodes.end());
    return;
}

//!
//! \brief Create reduced right-hand side
//!
//! \param rhs right-hand side
//! \param reducedRHS reduced right-hand side which will be set
//! \param homDirichletNodes dirichlet node information
//! \param optOutput output option for result of reducedRHS
//!
template<typename RhsT, typename BCIdType>
void CreateReducedRHS(const RhsT & rhs, Vector & reducedRHS, const std::vector<BCIdType> & homDirichletNodes, bool optOutput)
{
    int row = 0;
    for (int n = 0; n < rhs.GetSize(); ++n)
    {
        bool reducedInfo = FindElement(n,homDirichletNodes);
        if (!reducedInfo)
        {
            reducedRHS[row] = rhs[n];
            ++row;
        }
    }

    if (optOutput)
        outputVec(reducedRHS,"Reduced rhs (deleting of columns and rows at rhs with hom. Dirichlet bc)",reducedRHS.size());

    return;
}

//!
//! \brief Create reduced global stiffness matrix
//!
//! \param GSM global stiffness matrix
//! \param reducedGSM reduced global stiffness matrix which will be set
//! \param homDirichletNodes dirichlet node information
//! \param optOutput output option for result of reducedGSM
//!
template<typename BCIdT>
void CreateReducedGSM(const Matrix & GSM, Matrix & reducedGSM, const std::vector<BCIdT> & homDirichletNodes, bool optOutput)
{
    int row = 0; int col = 0;

    for (int i = 0; i < GSM.size(); ++i)
    {
        bool reducedInfoN = FindElement(i,homDirichletNodes);
        if (!reducedInfoN)
        {
            for (int j = 0; j < GSM.size (); ++j)
            {
                bool reducedInfoM = FindElement(j,homDirichletNodes);
                if (!reducedInfoM)
                {
                    reducedGSM[row][col] = GSM[i][j];
                    ++col;
                }
            }
            col = 0;
            ++row;
        }
    }

    if (optOutput)
        outputMat(reducedGSM,"Reduced global stiffness matrix (deleting of columns and rows at GSM with hom. Dirichlet bc)",reducedGSM.size(), reducedGSM[0].size());

    return;
}

//!
//! \brief Find an element in a set of elements.
//!
//! \param element searched element
//! \param set set of elements
//! \return 'true' if element was found, otherwise 'false'
//!
template<typename ElementT, typename T>
auto FindElement(const ElementT & element, const T & set) -> bool
{
    for (int i = 0; i < set.size(); ++i)
        if (element == set[i])
            return true;
    return false;
}

//!
//! \brief Create residuum for cg-solver.
//!
template<typename MeshT, typename BufferT, typename LoopbodyT, typename PairT, typename VecDslT>
BufferT GetResiduum(const MeshT & mesh, const BufferT & rhs, LoopbodyT bodyObj, const PairT & materials,
            const VecDslT & x0, const std::vector<int> & homDirichletNodes)
{
    HPM::Buffer<float, Mesh, Dofs<1, 0, 0, 0, 0>> r0(mesh); //residuum

    auto nodes { mesh.template GetEntityRange<0>() };
    bodyObj.Execute(HPM::ForEachEntity(
                  nodes,
                  std::tuple(ReadWrite(Node(r0))),
                  [&](auto const& node, const auto& iter, auto& lvs)
    {
        HPM::Buffer<float, Mesh, Dofs<1, 0, 0, 0, 0>> rowGSM(mesh);
    int nodeID = node.GetTopology().GetIndex(); // global index of node
    GetRowOfStiffnessMatrix(node, mesh, materials, rowGSM, homDirichletNodes);

    float val = 0;
    for (int i = 0; i < rhs.GetSize(); ++i)
        val += rowGSM[i]*x0[i];
    r0[nodeID] = rhs[nodeID]-val;
    }));

    outputVec(r0, "r0", 8);
    return r0;
}

//!
//! \brief Conjugated gradient algorithm.
//!
template<typename MeshT, typename BufferT, typename LoopbodyT, typename PairT>
auto cgSolver(const MeshT & mesh, const BufferT & rhs, LoopbodyT bodyObj, const PairT & materials, const int & numSolverIt, const float & tol,const std::vector<int> & homDirichletNodes)->Vector
{
    Vector x {0,0,0,0,0,0,0,0}; //start vector
    auto rBuffer = rhs; // set residuum
    Vector r     = Convert2(rBuffer);
    Vector d     = r; //search direction
    float r_scPr = 0; float a = 0; float b = 0; float eps = tol - 0.00001;
    
    bool abortCriterium = false;

    for (int i = 0; i < numSolverIt; ++i)
    {
        HPM::Buffer<float, Mesh, Dofs<1, 0, 0, 0, 0>> zBuffer(mesh);
        computeMatrixVecProdukt(mesh, d, bodyObj, materials, zBuffer, homDirichletNodes);
        
        Vector z = Convert2(zBuffer);
        r_scPr   = scPr(r,r);
        a        = r_scPr/scPr(d,z);
        x        = plus(x,msv(a,d));
        r        = minus(r, msv(a,z));
        b        = scPr(r,r)/r_scPr;
        d        = plus(r,msv(b,d));

        eps = std::sqrt(scPr(r, r));
        if (eps < tol)
            i = numSolverIt;
    }
    return x;
}

//!
//! \brief Conjugated gradient algorithm (matrix-free).
//!
template<typename MeshT, typename BufferT, typename LoopbodyT, typename PairT, typename VecDslT>
auto cgSolverMatFree(const MeshT & mesh, const BufferT & rhs, LoopbodyT bodyObj, const PairT & materials, VecDslT & x, const int & numSolverIt,
                     const float & tol, const std::vector<int> & homDirichletNodes)->VecDslT
{
    int const size = 8;
    using VecDSL = HPM::dataType::Vec<float,size>;
    float r_scPr = 0; float a = 0; float b = 0; float eps = tol - 0.00001;

    HPM::Buffer<float, Mesh, Dofs<1, 0, 0, 0, 0>> rBuffer(mesh);
    GetMatrixVecProduct(mesh, x, bodyObj, materials, rBuffer, homDirichletNodes);//A*x

    VecDSL r; Convert(rhs,r);
    VecDSL d = r; //search direction

    for (int it = 0; it < numSolverIt; ++it)
    {
        HPM::Buffer<float, Mesh, Dofs<1, 0, 0, 0, 0>> zBuffer(mesh);
        GetMatrixVecProduct(mesh, d, bodyObj, materials, zBuffer, homDirichletNodes);// A*d
        VecDSL z; Convert(zBuffer, z);
        r_scPr = r * r;
        a      = r_scPr/(d*z);
        x      = x + (a*d);
        r      = r - (a*z); 
        b      = (r*r)/r_scPr;
        d      = r + (b * d);

        eps = std::sqrt(r * r);
        if (eps < tol)
            it = numSolverIt;

    }
    return x;
}

//!
//! \brief Compute matrix-vector product directly without assembling the whole stiffness matrix
//! in advance (per rows of global stiffness matrix).
//!
template<typename MeshT, typename VectorT, typename LoopbodyT, typename PairT, typename BufferT>
void computeMatrixVecProdukt(const MeshT & mesh, const VectorT & dk, LoopbodyT bodyObj, const PairT & materials,
                     BufferT & z, const std::vector<int> & homDirichletNodes)
{
    auto nodes { mesh.template GetEntityRange<0>() };

    bodyObj.Execute(HPM::ForEachEntity(
                  nodes,
                  std::tuple(ReadWrite(Node(z))),
                  [&](auto const& node, const auto& iter, auto& lvs)
    {
        HPM::Buffer<float, Mesh, Dofs<1, 0, 0, 0, 0>> rowGSM(mesh);
        int nodeID = node.GetTopology().GetIndex(); // global index of node
        GetRowOfStiffnessMatrix(node, mesh, materials, rowGSM, homDirichletNodes);

        for (int i = 0; i < z.GetSize(); ++i)
            z[nodeID] += rowGSM[i]*dk[i];
    }));

    return;
}

//!
//! \brief Compute matrix-vector product directly without assembling the whole stiffness matrix
//! in advance (per scalars of global stiffnessmatrix).
//!
template<typename MeshT, typename VectorT, typename LoopbodyT, typename PairT, typename BufferT>
void GetMatrixVecProduct(const MeshT & mesh, const VectorT & dk, LoopbodyT bodyObj, const PairT & materials,
                         BufferT & z, const std::vector<int> & homDirichletNodes)
{
    HPM::Buffer<float, Mesh, Dofs<1, 0, 0, 0, 0>> zBuffer(mesh);
    auto cells { mesh.template GetEntityRange<3>() };
    bodyObj.Execute(HPM::ForEachEntity(
                  cells,
                  std::tuple(ReadWrite(Node(z))),
                  [&](auto const& cell, const auto& iter, auto& lvs)
    {
        const int nrows = dim+1; const int ncols = dim+1;
        const auto& gradients = GetGradientsDSL();
        const auto& nodeIdSet = cell.GetTopology().GetNodeIndices();

        auto tmp   = cell.GetGeometry().GetJacobian();
        float detJ = tmp.Determinant(); detJ = std::abs(detJ);
        auto inv   = tmp.Invert();
        auto invJT = inv.Transpose();

        // Material information is integrate as diffusion tensor D = sigma * I with sigma as random scalar value.
        // For example: sigma = 3 if material id is 1 and sigma = 6 if material id is 2
        float sigma; float gamma = 100;
        if (materials[cell.GetTopology().GetIndex()].second == 1) sigma = 3;
        else if (materials[cell.GetTopology().GetIndex()].second == 2) sigma = 6;
        else sigma = 1; // default

        for (int col = 0; col < ncols; ++col)
        {
            bool DirichletNode = false;
            if (FindElement(nodeIdSet[col], homDirichletNodes))
                DirichletNode = true; //z[nodeIdSet[col]] += gamma;

            auto gc = invJT*gradients[col]*sigma;
            for (int row = 0; row < nrows; ++row)
            {
                bool DirichletNode2 = false;
                if (FindElement(nodeIdSet[row], homDirichletNodes))
                    DirichletNode2 = true; //z[nodeIdSet[col]] += gamma;

                if ((DirichletNode) || (DirichletNode2))
                    z[nodeIdSet[col]] += 0;
                else
                {
                    auto gr            = invJT* gradients[row];
                    z[nodeIdSet[col]] += detJ/6 * (gc*gr) * dk[nodeIdSet[row]];
                }
            }
        }
    }));

    return;
}

//!
//! \brief Assemble rom of global stiffness matrix for function computeMatrixVecProdukt().
//!
template<typename NodeT, typename MeshT, typename PairT, typename BufferT>
void GetRowOfStiffnessMatrix(const NodeT& node, const MeshT& mesh, const PairT & materialsPerCellId,
                             BufferT & rowGSM, const std::vector<int> & homDirichletNodes)
{
    int nodeID = node.GetTopology().GetIndex(); // global index of node
    bool DirichletNode = false;
    if (FindElement(nodeID, homDirichletNodes))
         DirichletNode = true;

    if (DirichletNode)
        for (int i = 0; i < 8/*mesh.template GetNumEntities<0>*/; ++i)
    {
        if (i == nodeID)
            rowGSM[i] = 1;
        else
            rowGSM[i] = 0;
    }
    else
    {
        const auto& containing_cells = node.GetTopology().GetAllContainingCells();
        for (const auto& cell : containing_cells)
        {
            const int nrows = dim+1;
            const auto& gradients = GetGradientsDSL();
            const auto& nodeIdSet = cell.GetTopology().GetNodeIndices();

            // To-Do: find better syntax
            int localPositionOfNode;
            for (int i = 0; i < dim+1; ++i)
                if (nodeIdSet[i] == nodeID)
                    localPositionOfNode = i;

            auto tmp   = cell.GetGeometry().GetJacobian();
            float detJ = tmp.Determinant(); detJ = std::abs(detJ);
            auto inv   = tmp.Invert();
            auto invJT = inv.Transpose();

            // Material information is integrate as diffusion tensor D = v * I with v as random scalar value.
            // For example: v = 3 if material id is 1 and v = 6 if material id is 2
            float value;
            if (materialsPerCellId[cell.GetTopology().GetIndex()].second == 1)      value = 3;
            else if (materialsPerCellId[cell.GetTopology().GetIndex()].second == 2) value = 6;
            else value = 1; // default

            auto gc = invJT * gradients[localPositionOfNode] * value * (detJ/6);
            for (int row = 0; row < nrows; ++row)
            {
        if (FindElement(nodeIdSet[row], homDirichletNodes))
            rowGSM[nodeIdSet[row]] = 0;
        else
        {
                    auto gr                 = invJT* gradients[row];
                    rowGSM[nodeIdSet[row]] += gc*gr;
        }
            }
        }
    }

    return;
}

