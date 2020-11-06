/** ------------------------------------------------------------------------------ *
 * author(s)   : Merlind Schotte (schotte@zib.de)                                  *
 * institution : Zuse Institute Berlin (ZIB)                                       *
 * project     : HighPerMeshes (BMBF)                                              *
 *                                                                                 *
 * Description:                                                                    *
 * Implementation of a simple Poisson problem using the HighPerMeshes lib. This is *
 * intended to check the completeness and usability of the DSL/DRTS.               *
 * Features that differ from the MIDG2 example:                                    *
 * - degrees of freedom associated to vertices (instead of cells)                  *                      
 * - mesh in 3D (could be postponed - Poisson problem can be easily defined on 2D  *
 *   triangles as well)                                                            *
 * - using mass term for an unambiguous solution of u by adding "+u" to lhs        *
 *   (no boundary conditions)  							   *
 * - solvers (using self implemented cg matrix free solver)                        *
 *                                                                                 *
 * Poisson equation      : $\sigma \Delta u = f$        $on \Omega$                *
 * with $\sigma$ as constant diffusion and $f = 1$                                 *
 * - adding $u$ leads to : $\sigma \Delta u + u = 1$                               * 
 *                                                                                 *
 * last change: 20.05.2020                                                         *
 * ------------------------------------------------------------------------------ **/

#ifndef POISSON_CPP

#include <iostream>
#include <tuple>
#include <metis.h>
#include <HighPerMeshes.hpp>
#include <HighPerMeshes/third_party/metis/Partitioner.hpp>
#include <HighPerMeshesDRTS.hpp>
//#include <../../parser/WriteLoop.hpp>

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
using Mesh           = HPM::mesh::PartitionedMesh<CoordinateT, HPM::entity::Simplex>; //for distributed case
constexpr int dim    = Mesh::CellDimension;


/*-------------------------------------------------------------- (A) Functions: -------------------------------------------------------------------------------*/
template<typename MeshT, typename BufferT, typename ItLoopBodyObjT>
void SetRHS(const MeshT & mesh, BufferT & rhs, bool optOutput, ItLoopBodyObjT bodyObj);

template<typename MeshT, typename VectorT, typename LoopbodyT, typename BufferT>
void AssembleMatrixVecProduct(const MeshT & mesh, const VectorT & d, LoopbodyT bodyObj, BufferT & sBuffer);

template<typename MeshT, typename VectorT, typename LoopbodyT, typename BufferT>
void AssembleMatrixVecProductPerVector(const MeshT & mesh, const VectorT & d, LoopbodyT bodyObj, BufferT & sBuffer);

template<typename NodeT, typename MeshT, typename BufferT>
void AssembleRowOfStiffnessMatrix(const NodeT& node, const MeshT& mesh, BufferT & rowGSM);

template<typename MeshT, typename BufferT, typename LoopbodyT, typename VecDslT>
auto CGSolver(const MeshT & mesh, const BufferT & rhs, LoopbodyT bodyObj, VecDslT & x, const int & numSolverIt, const float & tol)->VecDslT;

template<typename MeshT, typename BufferT, typename LoopbodyT, typename VecT>
void CGSolver2(const MeshT & mesh, const BufferT & rhs, LoopbodyT bodyObj, VecT & x, const int & numSolverIt, const float & tol);

/*----------------------------------------------------------------- MAIN --------------------------------------------------------------------------------------*/
int main(int argc, char** argv)
{
    HPM::drts::Runtime<HPM::GetDistributedBuffer<>, HPM::UsingDistributedDevices> hpm({}, std::forward_as_tuple(argc, argv));
    HPM::DistributedDispatcher body{hpm.gaspi_context, hpm.gaspi_segment, hpm};
    HPM::auxiliary::ConfigParser CFG("configMatFree.cfg");
    std::string meshFile = CFG.GetValue<std::string>("MeshFile");
    const Mesh mesh      = Mesh::template CreateFromFile<HPM::auxiliary::AmiraMeshFileReader, ::HPM::mesh::MetisPartitioner>
                                         (meshFile, {hpm.GetL1PartitionNumber(), hpm.GetL2PartitionNumber()}, hpm.gaspi_runtime.rank().get());

    int const numNodes = 8;//mesh.template GetNumEntities<0>;

    /*------------------------------------------(2) Set right hand side: --------------------------------------------------------------------------------------*/
    HPM::Buffer<float, Mesh, Dofs<1, 0, 0, 0, 0>> rhs(mesh);
    SetRHS(mesh, rhs, false, body);

    /*------------------------------------------(3a) Solve: CGSolver ------------------------------------------------------------------------------------------*/
    HPM::dataType::Vec<float, numNodes> x2;
    for (int i = 0; i < numNodes; ++i) {x2[i]=0;} // set start vector

    CGSolver(mesh, rhs, body, x2, 10, 0.00001);
    outputVec(x2, "resultVec CGSolver", numNodes /*8*/);

    /*------------------------------------------(3b) Solve: CGSolver2 -----------------------------------------------------------------------------------------*/
    Vector x1 {0,0,0,0,0,0,0,0}; // set start vector
    CGSolver2(mesh, rhs, body, x1, 10, 0.00001);
    outputVec(x1, "resultVec cgSolver", numNodes);


    return 0;
}
#endif

/*----------------------------------------------------------------- (A) Functions (Implementation): -----------------------------------------------------------*/
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
        auto& rhs        = std::get<0>(lvs);
        auto jacobianMat = cell.GetGeometry().GetJacobian();
        double detJac    = jacobianMat.Determinant(); detJac = std::abs(detJac);

        for (const auto& node : cell.GetTopology().template GetEntities<0>())
        {
            int id      = node.GetTopology().GetLocalIndex();
            rhs[id][0] += detJac/24;
        }
    }));

    if (optOutput)
        outputVec(rhs,  "Buffer RHS in function SetRHS()", rhs.GetSize());

    return;
}

//!
//! matrix-vector product split into single scalar operations
//!
template<typename MeshT, typename VectorT, typename LoopbodyT, typename BufferT>
void AssembleMatrixVecProduct(const MeshT & mesh, const VectorT & d, LoopbodyT bodyObj, BufferT & sBuffer)
{
    auto cells { mesh.template GetEntityRange<3>() };
    bodyObj.Execute(HPM::ForEachEntity(
                  cells,
                  std::tuple(ReadWrite(Node(sBuffer))),
                  [&](auto const& cell, const auto& iter, auto& lvs)
    {
        constexpr int nrows = dim+1;
        constexpr int ncols = dim+1;
        const auto& gradients = GetGradientsDSL();
        const auto& nodeIdSet = cell.GetTopology().GetNodeIndices();

        const auto& tmp  = cell.GetGeometry().GetJacobian();
        const float detJ = std::abs(tmp.Determinant());

        // store detJ information beforehand
        const auto _detJ = [&]() {
                ::HPM::dataType::Matrix<float, nrows, ncols> matrix;
                for (int col = 0; col < ncols; ++col)
                {
                    for (int row = 0; row < nrows; ++row)
                    {
                        matrix[row][col] = (col == row ? detJ / 60.0 : detJ / 120.0);
                    }
                }
                return matrix;
            }();

        const auto& inv   = tmp.Invert();
        const auto& invJT = inv.Transpose();

        // sigma: random scalar value
        float sigma = 2;

        // separate GATHER
        std::array<float, nrows> _d;
        for (int row = 0; row < nrows; ++row)
            _d[row] = d[nodeIdSet[row]];

        // accumulate into contiguous block of memory
        std::array<float, ncols> result{};

        for (int col = 0; col < ncols; ++col)
        {
            const auto& gc = invJT * gradients[col] * sigma * (detJ/6);
            for (int row = 0; row < nrows; ++row)
            {
                float val      = _detJ[row][col];
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
//! matrix-vector product split into single vector products
//!
template<typename MeshT, typename VectorT, typename LoopbodyT, typename BufferT>
void AssembleMatrixVecProductPerVector(const MeshT & mesh, const VectorT & d, LoopbodyT bodyObj, BufferT & sBuffer)
{
    auto nodes { mesh.template GetEntityRange<0>() };

    bodyObj.Execute(HPM::ForEachEntity(
                  nodes,
                  std::tuple(ReadWrite(Node(sBuffer))),
                  [&](auto const& node, const auto& iter, auto& lvs)
    {
        HPM::Buffer<float, Mesh, Dofs<1, 0, 0, 0, 0>> rowGSM(mesh);
        int nodeID = node.GetTopology().GetIndex(); // global index of node
        AssembleRowOfStiffnessMatrix(node, mesh, rowGSM);

        for (int i = 0; i < sBuffer.GetSize(); ++i)
            sBuffer[nodeID] += rowGSM[i]*d[i];
    }));

    return;
}

//!
//! assemble local rows of stiffness matrix for function AssembleMatrixVecProductPerVector()
//!
template<typename NodeT, typename MeshT, typename BufferT>
void AssembleRowOfStiffnessMatrix(const NodeT& node, const MeshT& mesh, BufferT & rowGSM)
{
    int nodeID                   = node.GetTopology().GetIndex(); // global index of node
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

        // sigma random scalar value
        float sigma = 1;
        auto gc     = invJT * gradients[localPositionOfNode] * sigma * (detJ/6);
        for (int row = 0; row < nrows; ++row)
        {
            auto gr                 = invJT * gradients[row];
            rowGSM[nodeIdSet[row]] += gc*gr;
            if (row == localPositionOfNode)
                rowGSM[nodeIdSet[row]] += detJ/60;
            else
                rowGSM[nodeIdSet[row]] += detJ/120;
        }
    }

    return;
}

//!
//! conjugated gradient method without matrix vector operations
//!
template<typename MeshT, typename BufferT, typename LoopbodyT, typename VecDslT>
auto CGSolver(const MeshT & mesh, const BufferT & rhs, LoopbodyT bodyObj, VecDslT & x, const int & numSolverIt, const float & tol)->VecDslT
{
    int const size = 8;//x.size();
    using VecDSL = HPM::dataType::Vec<float,size>;
    float r_scPr = 0; float a = 0; float b = 0; float eps = tol - 0.00001;

    VecDSL r; Convert(rhs,r); // r = residuum (with x = null vector as start vector: r = rhs-A*x = rhs-0 = rhs)
    VecDSL d = r;             // d = (search) direction

    for (int it = 0; it < numSolverIt; ++it)
    {
        HPM::Buffer<float, Mesh, Dofs<1, 0, 0, 0, 0>> sBuffer(mesh);
        AssembleMatrixVecProduct(mesh, d, bodyObj, sBuffer);
        VecDSL s; Convert(sBuffer, s);
        r_scPr = r * r; // <r,r> scalar product
        a      = r_scPr/(d*s);
        x      = x + (a*d);
        r      = r - (a*s);
        b      = (r*r)/r_scPr; // here: r_scPr = <rOld,rOld>
        d      = r + (b * d);

        eps = std::sqrt(r * r);
        if (eps < tol)
            it = numSolverIt;
    }

    return x;
}

//!
//! conjugated gradient method
//!
template<typename MeshT, typename BufferT, typename LoopbodyT, typename VecT>
void CGSolver2(const MeshT & mesh, const BufferT & rhs, LoopbodyT bodyObj, VecT & x, const int & numSolverIt, const float & tol)
{
    float r_scPr = 0; float a = 0; float b = 0; float eps = tol - 0.00001;
    Vector r = Convert2(rhs); // set residuum (rhs-A*x = rhs-0 = rhs)
    Vector d = r;            // search direction

    for (int it = 0; it < numSolverIt; ++it)
    {
        HPM::Buffer<float, Mesh, Dofs<1, 0, 0, 0, 0>> sBuffer(mesh);
        AssembleMatrixVecProductPerVector(mesh, d, bodyObj, sBuffer);

        Vector s = Convert2(sBuffer);
        r_scPr   = scPr(r,r); // <r,r> (scalar product)
        a        = r_scPr/scPr(d,s);
        x        = plus(x, msv(a, d));
        r        = minus(r, msv(a,s)); // r - a*s
        b        = scPr(r,r)/r_scPr; // (rNew*rNew)/(rOld*rOld)
        d        = plus(r,msv(b,d)); // r + b*d

        eps = std::sqrt(scPr(r, r)); // sqrt(rNew*rNew)
        if (eps < tol)
            it = numSolverIt;
    }

    return;
}



