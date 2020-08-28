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
 * - mesh in 2D                                                                    *
 * - using mass term for an unambiguous solution of u by adding "+u" to lhs        *
 *   (no boundary conditions)  	                         						   *
 * - solvers (using self implemented cg matrix free solver)                        *
 *                                                                                 *
 * Poisson equation      : $\sigma \Delta u = f$        $on \Omega$                *
 * with $\sigma$ as constant diffusion and $f = 1$                                 *
 * - adding $u$ leads to : $\sigma \Delta u + u = 1$                               * 
 *                                                                                 *
 * last change: 21.08.2020                                                         *
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

#include <unistd.h>
#define GetCurrentDir getcwd

class DEBUG;
using namespace std;
using namespace HPM;
using namespace ::HPM::auxiliary;
template <std::size_t ...I>
using Dofs           = HPM::dataType::Dofs<I...>;
using Vector         = std::vector<float>;
using Matrix         = std::vector<Vector>;
using CoordinateT    = HPM::dataType::Vec<float,2>;
//using Mesh           = HPM::mesh::PartitionedMesh<CoordinateT, HPM::entity::Simplex>; //for distributed case
using Mesh           = HPM::mesh::Mesh<CoordinateT, HPM::entity::Simplex>; // for sequential case
constexpr int dim    = Mesh::CellDimension;

int const numNodes = 22;
using VecDSL       = HPM::dataType::Vec<float,numNodes>;

/*-------------------------------------------------------------- (A) Functions: -------------------------------------------------------------------------------*/
template<typename MeshT, typename BufferT, typename DispatcherT>
void CreateStartVector(const MeshT & mesh, BufferT & startVec, DispatcherT & dispatcher);

template<typename MeshT, typename BufferT, typename ItLoopBodyObjT>
void SetRHS(const MeshT & mesh, BufferT & rhs, bool optOutput, ItLoopBodyObjT bodyObj);

template<typename MeshT, typename VectorT, typename LoopbodyT, typename BufferT>
void AssembleMatrixVecProduct(const MeshT & mesh, const VectorT & d, LoopbodyT bodyObj, BufferT & sBuffer);

template<typename MeshT, typename BufferT, typename LoopbodyT, typename VecDslT>
auto CGSolver(const MeshT & mesh, const BufferT & rhs, LoopbodyT bodyObj, VecDslT & x, /*const*/ int const & size,
              const int & numSolverIt, const float & tol)->VecDslT;

/*----------------------------------------------------------------- MAIN --------------------------------------------------------------------------------------*/
int main(int argc, char** argv)
{
    /*------------------------------------------(1a) Sequential case: -----------------------------------------------------------------------------------------*/
        HPM::drts::Runtime hpm { HPM::GetBuffer{} };
        HPM::SequentialDispatcher dispatcher;
        ConfigParser CFG("config2DMatFree.cfg");
        string meshFile = CFG.GetValue<string>("MeshFile");
        const Mesh mesh = Mesh::template CreateFromFile<HPM::auxiliary::AmiraMeshFileReader>(meshFile);
    /*------------------------------------------(1b) Distributed case: --------------------------------------------------------------------------------------*/
//    drts::Runtime<GetDistributedBuffer<>, UsingDistributedDevices> hpm({}, forward_as_tuple(argc, argv));
//    DistributedDispatcher dispatcher{hpm.gaspi_context, hpm.gaspi_segment, hpm};
//    ConfigParser CFG("config2DMatFree.cfg");
//    string meshFile = CFG.GetValue<string>("MeshFile");
//    const Mesh mesh      = Mesh::template CreateFromFile<AmiraMeshFileReader, ::HPM::mesh::MetisPartitioner>
//                                          (meshFile, {hpm.GetL1PartitionNumber(), hpm.GetL2PartitionNumber()}, hpm.gaspi_runtime.rank().get());

    //int const numNodes = 25;//11;//mesh.template GetNumEntities<0>;

    /*------------------------------------------(2) Set right hand side: --------------------------------------------------------------------------------------*/
    HPM::Buffer<float, Mesh, Dofs<1, 0, 0, 0>> rhs(mesh);
    SetRHS(mesh, rhs, true, dispatcher);

    /*------------------------------------------(3) Solve: CGSolver -------------------------------------------------------------------------------------------*/
    char buff[FILENAME_MAX]; //create string buffer to hold path
    GetCurrentDir(buff, FILENAME_MAX);
    string currentWorkingDir(buff);
    string foldername = "results2DPoisson";// "TestAllGather2_20x20Mesh_DistrCaseNuma2";
    string filename   = "Sequential_2DPoisson_irregular22NodeMesh";//"TestAllGather2_20x20Mesh_DistrCaseNuma2_";

    HPM::dataType::Vec<float, numNodes> x;
    for (int i = 0; i < numNodes; ++i) {x[i]=0;} // set start vector

    writeVTKOutputParabolicWO_BC(mesh, currentWorkingDir, foldername, "Sequential_2DPoisson_irregular22NodeMesh_xStart", x, "resultX");

    int size = 25;
    CGSolver(mesh, rhs, dispatcher, x, size, 50, 0.001);

    writeVTKOutputParabolicWO_BC(mesh, currentWorkingDir, foldername, filename, x, "resultX");

    return 0;
}
#endif

/*----------------------------------------------------------------- (A) Functions (Implementation): -----------------------------------------------------------*/
template<typename MeshT, typename BufferT, typename DispatcherT>
void CreateStartVector(const MeshT & mesh, BufferT & startVec, DispatcherT & dispatcher)
{
    auto nodes { mesh.template GetEntityRange<0>() };
    dispatcher.Execute(ForEachEntity(
                           nodes,
                           tuple(Write(Node(startVec))),
                           [&](auto const& node, const auto& iter, auto& lvs)
    {
        auto& startVec = HPM::dof::GetDofs<HPM::dof::Name::Node>(std::get<0>(lvs));
        startVec[0]    = 0;
    }));

    return;
}

//!
//! \param mesh unitcube splitted into 5 tetrahedrons
//! \param rhs right-hand side which should be assembled in this looptype as vector per tetrahedron
//! \param optOutput output option for rhs
//! \param bodyObj object of loop body
//!
template<typename MeshT, typename BufferT, typename ItLoopBodyObjT>
void SetRHS(const MeshT & mesh, BufferT & rhs, bool optOutput, ItLoopBodyObjT bodyObj)
{
    auto cells { mesh.template GetEntityRange<dim>() };

    bodyObj.Execute(HPM::ForEachEntity(
                         cells,
                         std::tuple(ReadWrite(Node(rhs))),
                         [&](auto const& cell, const auto& iter, auto& lvs)
    {
        auto& rhs        = HPM::dof::GetDofs<HPM::dof::Name::Node>(std::get<0>(lvs));
        auto jacobianMat = cell.GetGeometry().GetJacobian();
        double detJac    = jacobianMat.Determinant(); detJac = std::abs(detJac);

        for (const auto& node : cell.GetTopology().template GetEntities<0>())
        {
            int id      = node.GetTopology().GetLocalIndex();
            rhs[id][0] += detJac/2;
        }
    }));

//    if (optOutput)
//        outputVec(rhs,  "Buffer RHS in function SetRHS()", rhs.GetSize());
//    for (int i = 0; i < rhs.GetSize(); ++i)
//        cout << "rhs["<<i<<"]:" << rhs[i] << endl;

    return;
}

//!
//! matrix-vector product split into single scalar operations
//!
template<typename MeshT, typename VectorT, typename LoopbodyT, typename BufferT>
void AssembleMatrixVecProduct(const MeshT & mesh, const VectorT & d, LoopbodyT bodyObj, BufferT & sBuffer)
{
    auto cells { mesh.template GetEntityRange<dim>() };
    bodyObj.Execute(HPM::ForEachEntity(
                  cells,
                  std::tuple(ReadWrite(Node(sBuffer))),
                  [&](auto const& cell, const auto& iter, auto& lvs)
    {
        constexpr int nrows = dim+1;
        constexpr int ncols = dim+1;
        const auto& gradients = GetGradients2DP1();
        const auto& nodeIdSet = cell.GetTopology().GetNodeIndices();
        const auto& nodes = cell.GetTopology().GetNodes();
        const auto& tmp  = cell.GetGeometry().GetJacobian();
//        const /*Matrix*/float tmp[dim][dim] = { (nodes[0][0] - nodes[1][0]), (nodes[2][0] - nodes[1][0]),
//                            (nodes[0][1] - nodes[1][1]), (nodes[2][1] - nodes[1][1])
//                            };
        const float detJ = std::abs(tmp.Determinant());//std::abs((tmp[0][0]*tmp[1][1])-(tmp[0][1]*tmp[1][0]));

        if (detJ == 0.0)
        {
            for (const auto& node : cell.GetTopology().template GetEntities<0>() )
            {
                auto coords = node.GetTopology().GetVertices();
                cout<<coords[0][0]<<"     "<< coords[0][1] << endl;
            }
        }

        const auto M = [&]() {
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
        // invert and transpose tmp
//        const /*Matrix*/float invJT[dim][dim] = { ( 1/detJ) * tmp[1][1], (-1/detJ) * tmp[1][0],
//                              (-1/detJ) * tmp[0][1], ( 1/detJ) * tmp[0][0]
//                              };

        // sigma: random scalar value
        float sigma = 2;

        // separate GATHER
        std::array<float, nrows> _d;
        for (int row = 0; row < nrows; ++row)
            _d[row] = d[nodeIdSet[row]];

        // accumulate into contiguous block of memory
        std::array<float, ncols> result{};

        const float v = (detJ/2) * sigma;

        for (int col = 0; col < ncols; ++col)
        {
            const auto& gc = invJT * gradients[col];
//            float gc[2] = { v * (invJT[0][0] * gradients[col][0]) + (invJT[0][1] * gradients[col][1]),
//                               v * (invJT[1][0] * gradients[col][0]) + (invJT[1][1] * gradients[col][1])
//                               };
            for (int row = 0; row < nrows; ++row)
            {
                //float val      = 0.5;//_detJ[row][col];
                //const auto& gr = invJT * gradients[row];
//                float gr[2] = { (invJT[0][0] * gradients[row][0]) + (invJT[0][1] * gradients[row][1]),
//                                   (invJT[1][0] * gradients[row][0]) + (invJT[1][1] * gradients[row][1])
//                                   };
                //result[col]   += ((gc[0]*gr[0])+(gc[1]*gr[1])) * _d[row];
                float val      = M[row][col];
                const auto& gr = invJT * gradients[row];
                result[col]   += ((gc*gr) * val) * _d[row];
            }
        }

        // separate SCATTER (accumulate)
        for (int col = 0; col < ncols; ++col)
            sBuffer[nodeIdSet[col]] += result[col];
    }));

//    for (int i = 0; i < sBuffer.GetSize(); ++i)
//        cout << "sBuffer["<<i<<"]:" << sBuffer[i] << endl;

    return;
}

//!
//! conjugated gradient method without matrix vector operations
//!
template<typename MeshT, typename BufferT, typename LoopbodyT, typename VecDslT>
auto CGSolver(const MeshT & mesh, const BufferT & rhs, LoopbodyT bodyObj, VecDslT & x, /*const*/ int const & size,
              const int & numSolverIt, const float & tol)->VecDslT
{
    //int const size = 25;//11;//8;//x.size();
    //using VecDSL = HPM::dataType::Vec<float,25>;
    float r_scPr = 0; float a = 0; float b = 0; float eps = tol - 0.00001;

    VecDSL r; Convert(rhs,r); // r = residuum (with x = null vector as start vector: r = rhs-A*x = rhs-0 = rhs)
    VecDSL d = r;             // d = (search) direction

    for (int it = 0; it < numSolverIt; ++it)
    {
        HPM::Buffer<float, Mesh, Dofs<1, 0, 0, 0>> sBuffer(mesh);
        //cout << "size sBuffer = " << sBuffer.GetSize() << endl;
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

        outputVec(x, "resultVec CGSolver", numNodes);
    }

    return x;
}
