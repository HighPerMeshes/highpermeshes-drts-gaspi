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
 *   (no boundary conditions)  							                           *
 * - solvers (using self implemented cg matrix free solver)                        *
 *                                                                                 *
 * Poisson equation      : $\sigma \Delta u = f$        $on \Omega$                *
 * with $\sigma$ as constant diffusion and $f = 1$                                 *
 * - adding $u$ as mass term leads to : $\sigma \Delta u + u                       *
 *                                                                                 *
 * last change: 14.12.2020                                                         *
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
#include <../examples/Functions/mathFunctions.hpp>

class DEBUG;
using namespace HPM;
using namespace ::HPM::auxiliary;
using namespace std;
//using namespace ::HPM::dataType;
template <std::size_t ...I>
using Dofs           = HPM::dataType::Dofs<I...>;
using CoordinateT    = HPM::dataType::Vec<float,3>;
using Mesh           = HPM::mesh::PartitionedMesh<CoordinateT, HPM::entity::Simplex>; //for distributed case
constexpr int dim    = Mesh::CellDimension;


/*-------------------------------------------------------------- (A) Functions: -------------------------------------------------------------------------------*/
template<typename MeshT, typename BufferT, typename ItLoopBodyObjT>
void SetRHS(const MeshT & mesh, BufferT & rhs, bool optOutput, ItLoopBodyObjT bodyObj);

template<typename MatrixT>
void GetMassTerms(MatrixT & matrix);

template<typename MeshT, typename VectorT, typename LoopbodyT, typename BufferT, typename MatrixT>
void AssembleMatrixVecProduct(const MeshT & mesh, const VectorT & d, LoopbodyT bodyObj, BufferT & sBuffer, MatrixT & massTerms, const float & sigma);

template<typename MeshT, typename LoopbodyT, typename BufferT, typename MatrixT, typename VecDslT>
auto CGSolver(const MeshT & mesh, LoopbodyT bodyObj, const BufferT & rhs, MatrixT & massTerms, const float & sigma,
              VecDslT & x, const int & numSolverIt, const float & tol)->VecDslT;

/*----------------------------------------------------------------- MAIN --------------------------------------------------------------------------------------*/
int main(int argc, char** argv)
{
    drts::Runtime<HPM::GetDistributedBuffer<>, HPM::UsingDistributedDevices> hpm({}, forward_as_tuple(argc, argv));
    DistributedDispatcher body{hpm.gaspi_context, hpm.gaspi_segment, hpm};
    ConfigParser CFG("configMatFree.cfg");
    string meshFile = CFG.GetValue<string>("MeshFile");
    const Mesh mesh = Mesh::template CreateFromFile<HPM::auxiliary::AmiraMeshFileReader, ::HPM::mesh::MetisPartitioner>
                            (meshFile, {hpm.GetL1PartitionNumber(), hpm.GetL2PartitionNumber()}, hpm.gaspi_runtime.rank().get());

    int const numNodes = 8;//mesh.template GetNumEntities<0>;

    /*------------------------------------------(2) Set right hand side and mass terms: -----------------------------------------------------------------------*/
    Buffer<float, Mesh, Dofs<1, 0, 0, 0, 0>> rhs(mesh);
    SetRHS(mesh, rhs, true, body);

    // store mass information beforehand
    dataType::Matrix<float, dim+1, dim+1> massTerms;
    GetMassTerms(massTerms);

    /*------------------------------------------(3) Solve: CGSolver -------------------------------------------------------------------------------------------*/
    dataType::Vec<float, numNodes> x;
    for (int i = 0; i < numNodes; ++i) {x[i]=0;} // set start vector

    float sigma = 1; // here: random scalar value without 0
    CGSolver(mesh, body, rhs, massTerms, sigma, x, 10, 0.00001);
    outputVec(x, "resultVec CGSolver", numNodes);

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
    auto nodes { mesh.template GetEntityRange<0>() };

    bodyObj.Execute(ForEachEntity(
                         nodes,
                         tuple(ReadWrite(Node(rhs))),
                         [&](auto const& vertex, const auto& iter, auto& lvs)
    {
        auto& rhs         = dof::GetDofs<HPM::dof::Name::Node>(get<0>(lvs));
        const auto &cells = vertex.GetTopology().GetAllContainingCells();

        for (const auto &cell : cells)
        {
            auto jacobianMat = cell.GetGeometry().GetJacobian();
            double detJac    = jacobianMat.Determinant();
            detJac           = abs(detJac);
            rhs[0]          += detJac/24;
        }
    }));

    if (optOutput)
        outputVec(rhs,  "Buffer RHS in function SetRHS()", rhs.GetSize());

    return;
}

//!
//! store mass terms into (dim+1)x(dim+1)-matrix
//!
template<typename MatrixT>
void GetMassTerms(MatrixT & matrix)
{
    for (int col = 0; col < dim+1; ++col)
    {
        for (int row = 0; row < dim+1; ++row)
        {
            matrix[row][col] = (col == row ? 1 / 60.0 : 1 / 120.0);
        }
    }
    return;
}

//!
//! matrix-vector product split into single scalar operations
//!
template<typename MeshT, typename VectorT, typename LoopbodyT, typename BufferT, typename MatrixT>
void AssembleMatrixVecProduct(const MeshT & mesh, const VectorT & d, LoopbodyT bodyObj, BufferT & sBuffer, MatrixT & massTerms, const float & sigma)
{
    auto nodes { mesh.template GetEntityRange<0>() };
    bodyObj.Execute(ForEachEntity(
                  nodes,
                  tuple(ReadWrite(Node(sBuffer))),
                  [&](auto const& node, const auto& iter, auto& lvs)
    {
        constexpr int nrows = dim+1;
        const auto& gradients = GetGradientsDSL();
        auto& sBuffer         = dof::GetDofs<HPM::dof::Name::Node>(get<0>(lvs));
        const auto & cells    = node.GetTopology().GetAllContainingCells();

        for (const auto &cell : cells)
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

            const auto& gc = invJT * gradients[locID] * sigma * (detJ/6);
            for (int row = 0; row < nrows; ++row)
            {
                const auto& gr = invJT * gradients[row];
                sBuffer[0] += ((gc*gr) + (detJ * massTerms[row][locID])) * _d[row];
            }
        }
    }));

    return;
}

//!
//! conjugated gradient method without matrix vector operations
//!
template<typename MeshT, typename LoopbodyT, typename BufferT, typename MatrixT, typename VecDslT>
auto CGSolver(const MeshT & mesh, LoopbodyT bodyObj, const BufferT & rhs, MatrixT & massTerms, const float & sigma,
              VecDslT & x, const int & numSolverIt, const float & tol)->VecDslT
{
    int const size = 8;//x.size();
    using VecDSL = dataType::Vec<float,size>;
    float r_scPr = 0; float a = 0; float b = 0; float eps = tol - 0.00001;

    VecDSL r; Convert(rhs,r); // r = residuum (with x = null vector as start vector: r = rhs-A*x = rhs-0 = rhs)
    VecDSL d = r;             // d = (search) direction

    for (int it = 0; it < numSolverIt; ++it)
    {
        Buffer<float, Mesh, Dofs<1, 0, 0, 0, 0>> sBuffer(mesh);
        AssembleMatrixVecProduct(mesh, d, bodyObj, sBuffer, massTerms, sigma);
        VecDSL s; Convert(sBuffer, s);
        r_scPr = r * r; // <r,r> scalar product
        a      = r_scPr/(d*s);
        x      = x + (a*d);
        r      = r - (a*s);
        b      = (r*r)/r_scPr; // here: r_scPr = <rOld,rOld>
        d      = r + (b * d);

        eps = sqrt(r * r);
        if (eps < tol)
            it = numSolverIt;
    }

    return x;
}
