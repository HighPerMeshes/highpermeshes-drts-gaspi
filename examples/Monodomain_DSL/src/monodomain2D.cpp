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
 * last change: 17.12.2020                                                         *
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
#include <../examples/Monodomain_DSL/src/monodomain2DSettings.hpp>
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
using CoordinateType = dataType::Vec<float,2>;
using Mesh           = mesh::PartitionedMesh<CoordinateType, entity::Simplex>;
constexpr int dim    = Mesh::CellDimension;

/*-------------------------------------------------------------- (A) Functions: -------------------------------------------------------------------------------*/
auto CreateFile(string const & pathToFolder, string const & foldername, string const & filename) -> ofstream;

//template<typename FileT, typename T, typename NameT, typename NameT2>
//void WriteParameterInfoIntoFile(FileT& file, const NameT& fileName, const T& parameter, const NameT2& parameterName);

template<typename MeshT, typename BufferT, typename DispatcherT>
void CreateStartVector(const MeshT & mesh, BufferT & startVec, const float & startValLeft, const float & startValRight,
                       const int & maxX, const int & maxY, DispatcherT & dispatcher);

template<typename T, typename VectorT>
void GatherVec(const T & gatherVec, VectorT & resultVec, const int & numNodes);

template<typename MeshT, typename DispatcherT, typename BufferT>
void AssembleLumpedMassMatrix(const MeshT & mesh, DispatcherT & dispatcher, BufferT & lumpedMat) ;

template<typename MeshT, typename VectorT, typename DispatcherT, typename BufferT>
void AssembleMatrixVecProduct2D(const MeshT & mesh, const VectorT & d, DispatcherT & dispatcher, BufferT & sBuffer);

template<typename BufferT, typename DispatcherT, typename MeshT, typename MutexT>
void FWEuler(const MeshT & mesh, DispatcherT & dispatcher, BufferT & vecOld, BufferT & vecDeriv, const float & h,
             const bool & optionWrite, MutexT & mutex, const stringstream & fstreamNumber);

template<typename BufferT, typename MeshT, typename DispatcherT>
void computeIionUDerivWDeriv(const MeshT & mesh, DispatcherT & dispatcher, BufferT & f, BufferT & u_deriv, BufferT & w_deriv,
                             BufferT & u, BufferT & w, BufferT & lumpedM, BufferT & s, const float & sigma,
                             const float & a, const float & b, const float & eps);

template<typename ArrayT, typename CharT>
void WriteFStreamToArray(const CharT * filename, ArrayT & array, mutex & mtx);

/*----------------------------------------------------------------- MAIN --------------------------------------------------------------------------------------*/
int main(int argc, char** argv)
{
    bool optionAllGather = true;
    bool optionWriteLoop = true; // if optionWriteLoop = true -> optionWriteOut = true!
    bool optionWriteOut  = true;

    /*------------------------------------------(1) Set run-time system and read mesh information: ------------------------------------------------------------*/
    drts::Runtime<GetDistributedBuffer<>, UsingDistributedDevices> hpm({}, forward_as_tuple(argc, argv));
    DistributedDispatcher dispatcher{hpm.gaspi_context, hpm.gaspi_segment, hpm};
    ConfigParser CFG("config.cfg");
    string meshFile = CFG.GetValue<string>("MeshFile");
    const Mesh mesh      = Mesh::template CreateFromFile<AmiraMeshFileReader, ::HPM::mesh::MetisPartitioner>
            (meshFile, {hpm.GetL1PartitionNumber(), hpm.GetL2PartitionNumber()}, hpm.gaspi_runtime.rank().get());

    /*------------------------------------------(2) Set directory-,folder- and filename of result -------------------------------------------------------------*/
    char buff[FILENAME_MAX]; //create string buffer to hold path
    GetCurrentDir(buff, FILENAME_MAX);
    string currentWorkingDir(buff);

    string foldername = "testProcessesWithMD2D";// "testOption2AsCandidate0";//"test3DOptFor2DCase";// "TestAllGather2_20x20Mesh_DistrCaseNuma2";
    string filename   = "MD2D_2Processes";//"test3DOptFor2DCase100x100SmallTau_";//"TestAllGather2_20x20Mesh_DistrCaseNuma2_";
    //string parameterFilename = "testParameterfile";

    /*------------------------------------------(3) Set start values ------------------------------------------------------------------------------------------*/
    int numIt = 100;//150;//500;//2;//200;//400;//1000;
    float h; float a; float b; float eps; float sigma; float u0L; float u0R; float w0L; float w0R;
    //auto file = CreateFile(currentWorkingDir, foldername, parameterFilename);
    SetStartValues(1, h, a, b, eps, sigma, u0L, u0R, w0L, w0R/*, file*/);

    int numNodes = mesh.template GetNumEntities<0>();
    int maxX = ceil(sqrt(numNodes)/4);
    int maxY = ceil(sqrt(numNodes));

    Buffer<double, Mesh, Dofs<1, 0, 0, 0>> u(mesh);
    CreateStartVector(mesh, u, u0L, u0R, maxX, maxY, dispatcher);

    Buffer<double, Mesh, Dofs<1, 0, 0, 0>> w(mesh);
    CreateStartVector(mesh, w, w0L, w0R, maxX, maxY, dispatcher);

    Buffer<double, Mesh, Dofs<1, 0, 0, 0>> u_deriv(mesh);
    Buffer<double, Mesh, Dofs<1, 0, 0, 0>> w_deriv(mesh);
    Buffer<double, Mesh, Dofs<1, 0, 0, 0>> f(mesh);
    Buffer<double, Mesh, Dofs<1, 0, 0, 0>> StiffVecU(mesh);

    /*------------------------------------------(4) Create monodomain problem ---------------------------------------------------------------------------------*/
    Buffer<double, Mesh, Dofs<1, 0, 0, 0>> lumpedMat(mesh);
    AssembleLumpedMassMatrix(mesh, dispatcher, lumpedMat);

    // check if startvector was set correctly by creating output file at time step zero
    const auto u0_gather = AllGather<0>(u, static_cast<UsingGaspi&>(hpm));
    vector<float> u0_total(numNodes);
    GatherVec(u0_gather, u0_total, numNodes);

    stringstream s; s << 0;
    string name = filename + s.str();
    writeVTKOutput2DTime(mesh, currentWorkingDir, foldername, name, u0_total, "resultU");

    mutex mtx;

    // compute u and w
    for (int j = 0; j < numIt; ++j)
    {
        stringstream s; s << j+1;

        computeIionUDerivWDeriv(mesh, dispatcher, f, u_deriv, w_deriv, u, w, lumpedMat, StiffVecU, sigma, a, b, eps);
        FWEuler(mesh, dispatcher, u, u_deriv, h, optionWriteOut, mtx, s);
        FWEuler(mesh, dispatcher, w, w_deriv, h, false, mtx, s);

        if (optionAllGather) //create output files using AllGather
        {
            const auto u_gather = AllGather<0>(u, static_cast<UsingGaspi&>(hpm));
            vector<float> u_total(numNodes);
            GatherVec(u_gather, u_total, numNodes);

            if ((j+1)%2 == 0)
            {
                const size_t proc_id = hpm.gaspi_context.rank().get();
                cout << "Process id: " << proc_id << endl;

                name = filename + s.str();
                writeVTKOutput2DTime(mesh, currentWorkingDir, foldername, name, u_total, "resultU");
            }
        }
    }

    if (optionWriteLoop) //create output files using WriteLoop
    {
        Vector array; array.resize(numNodes);
        for (int k = 0; k < numIt; ++k)
        {
            if ((k+1)%2 == 0)
            {
                stringstream s; s << k+1;
                string distFileName = "WriteLoop" + s.str() + ".txt";
                WriteFStreamToArray(distFileName.c_str(), array, mtx);
                name = "MD2D_2Processes_WL_" + s.str();
                writeVTKOutput2DTime(mesh, currentWorkingDir, foldername, name, array, "resultU");
            }
        }
    }

    return 0;
}
#endif

/*----------------------------------------------------------------- (A) Functions (Implementation): -----------------------------------------------------------*/
//!
//! \brief Create a start vector.
//!
template<typename MeshT, typename BufferT, typename DispatcherT>
void CreateStartVector(const MeshT & mesh, BufferT & startVec, const float & startValLeft, const float & startValRight,
                       const int & maxX, const int & maxY, DispatcherT & dispatcher)
{
    auto nodes {mesh.template GetEntityRange<0>()};

    dispatcher.Execute(ForEachEntity(
                           nodes,
                           tuple(Write(Node(startVec))),
                           [&](auto const& node, const auto& iter, auto& lvs)
    {
        auto& startVec = dof::GetDofs<dof::Name::Node>(get<0>(lvs));
        auto coords    = node.GetTopology().GetVertices();

        if ( (coords[0][0] < maxX) && (coords[0][1] < maxY) )
            startVec[0] = startValLeft;
        else
            startVec[0] = startValRight;
    }));

    return;
}

//!
//! \brief Assemble row-sum lumped mass matrix
//!
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
            lumpedMat[0]     += detJ/6; //detJ*1/12 + detJ*1/24 + detJ*1/24
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
    auto nodes {mesh.template GetEntityRange<0>()};
    dispatcher.Execute(ForEachEntity(nodes,
                                     tuple(Write(Node(sBuffer))),
                                     [&](auto const& node, const auto& iter, auto& lvs)
    {
        auto& sBuffer = dof::GetDofs<dof::Name::Node>(get<0>(lvs));
        constexpr int nrows = dim+1;
        const auto& gradients = GetGradients2DP1();
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

            const auto& gc = invJT * gradients[locID] * (detJ/2);
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
template<typename BufferT, typename DispatcherT, typename MeshT, typename MutexT>
void FWEuler(const MeshT & mesh, DispatcherT & dispatcher, BufferT & vecOld, BufferT & vecDeriv, const float & h,
             const bool & optionWrite, MutexT & mutex, const stringstream & fstreamNumber)
{
    auto nodes {mesh.template GetEntityRange<0>()};

    if (optionWrite)
    {
        string distFileName = "WriteLoop" + fstreamNumber.str() + ".txt";
        ofstream fstream {distFileName};
        dispatcher.Execute(ForEachEntity(nodes,
                                         tuple(ReadWrite(Node(vecOld)), Read(Node(vecDeriv))),
                                         [&](auto const& node, const auto& iter, auto& lvs)
        {
            auto& vecOld   = dof::GetDofs<dof::Name::Node>(get<0>(lvs));
            auto& vecDeriv = dof::GetDofs<dof::Name::Node>(get<1>(lvs));
            vecOld[0] += h*vecDeriv[0];
        }),
            WriteLoop(mutex, fstream, nodes, vecOld)
      );
        fstream.close();
    }
    else {
        dispatcher.Execute(ForEachEntity(nodes,
                                         tuple(Write(Node(vecOld)), Read(Node(vecDeriv))),
                                         [&](auto const& node, const auto& iter, auto& lvs)
        {
            //int id      = node.GetTopology().GetIndex();
            //vecOld[id] += h*vecDeriv[id];
            auto& vecOld   = dof::GetDofs<dof::Name::Node>(get<0>(lvs));
            auto& vecDeriv = dof::GetDofs<dof::Name::Node>(get<1>(lvs));
            vecOld[0] += h*vecDeriv[0];
        }));
    }

    return;
}

//!
//! \brief Compute ion current, derivation of u and derivation of w at time step t.
//!
template<typename BufferT, typename MeshT, typename DispatcherT>
void computeIionUDerivWDeriv(const MeshT & mesh, DispatcherT & dispatcher, BufferT & f, BufferT & u_deriv, BufferT & w_deriv, BufferT & u,
                             BufferT & w, BufferT & lumpedM, BufferT & s, const float & sigma, const float & a, const float & b, const float & eps)
{
    AssembleMatrixVecProduct2D(mesh, u, dispatcher, s);

    auto nodes {mesh.template GetEntityRange<0>()};
    dispatcher.Execute(ForEachEntity(nodes,
                                     tuple(ReadWrite(Node(f)),Write(Node(u_deriv)),Write(Node(w_deriv)),
                                         Read(Node(u)), Read(Node(w)), Read(Node(lumpedM)), ReadWrite(Node(s))),
                                     [&](auto const& node, const auto& iter, auto& lvs)
    {
//        auto& f       = dof::GetDofs<dof::Name::Node>(std::get<0>(lvs));
//        auto& u_deriv = dof::GetDofs<dof::Name::Node>(std::get<1>(lvs));
//        auto& w_deriv = dof::GetDofs<dof::Name::Node>(std::get<2>(lvs));

//        auto& u       = dof::GetDofs<dof::Name::Node>(std::get<3>(lvs));
//        auto& w       = dof::GetDofs<dof::Name::Node>(std::get<4>(lvs));
//        auto& lumpedM = dof::GetDofs<dof::Name::Node>(std::get<5>(lvs));
//        auto& s       = dof::GetDofs<dof::Name::Node>(std::get<6>(lvs));

//        f[0]       = (u[0] * (1-u[0]) * (u[0]-a)) - w[0];
//        u_deriv[0] = ((1/lumpedM[0]) * sigma * s[0]) + f[0];
//        w_deriv[0] = eps*(u[0]-(b*w[0]));
        int id      = node.GetTopology().GetIndex();
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
//    file.seekp(ios::end);
//    file << '\n' << parameterName << " = " << parameter << '\n';
//    file.close();
//    return;
//}

auto CreateFile(string const & pathToFolder, string const & foldername, string const & filename) -> ofstream
{
    string fname = pathToFolder + "/" + filename + ".txt";
    ofstream file(fname.c_str());
    file << "Some information about the parameter settings:" << '\n';
    return file;
}

template<typename ArrayT, typename CharT>
void WriteFStreamToArray(const CharT * filename, ArrayT & array, mutex & mtx)
{
    std::lock_guard guard {mtx};
    FILE* f = fopen(filename,"r");

    if (f!=NULL)
    {
        const int size = 100; char str[size];
        int ID; float val;
        bool findID, findVal; findID = findVal = false;

        // reading file line by line and searching for 'keyletters' of keywords
        while (feof(f) == 0)
        {
            fgets(str, size, f);
            if (str[5] == 'x') // searching for 'x' of keyword 'index' at file f
            {
                string s;
                s.append(1,str[8]);

                int num = 1;
                while (str[8+num] !='\n')
                {
                    s.append(1,str[8+num]);
                    ++num;
                }

                ID = stoi(s);
                findID = true;
            }

            if (str[1] == 'V') // searching for 'v' of keyword 'value' at file f
            {
                string s2;
                s2.append(1,str[8]);

                int num = 1;
                while (str[8+num] !='\n')
                {
                    s2.append(1,str[8+num]);
                    ++num;
                }

                val = stof(s2);
                findVal = true;
            }

            if (findID == true && findVal == true)
            {
                array[ID] = val;
                findID = findVal = false;
            }
        }
    }
    else
        cout<<"Error: can't open file."<<endl;

    fclose(f);
    return;
}

template<typename T, typename VectorT>
void GatherVec(const T & gatherVec, VectorT & resultVec, const int & numNodes)
{
    const size_t numBuffers = gatherVec.size()/numNodes;
    for (size_t i = 0; i < numNodes; ++i)
    {
        resultVec[i] = 0;
        for (size_t k = 0; k < numBuffers; ++k)
            resultVec[i] += gatherVec[k * numNodes + i];
    }
}
