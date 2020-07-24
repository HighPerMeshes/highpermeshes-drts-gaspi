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
 * last change: 10.07.2020                                                         *
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
using Mesh           = mesh::PartitionedMesh<CoordinateType, entity::Simplex>;
constexpr int dim    = Mesh::CellDimension;

/*-------------------------------------------------------------- (A) Functions: -------------------------------------------------------------------------------*/
auto CreateFile(string const & pathToFolder, string const & foldername, string const & filename) -> ofstream;

void SetStartValues(int option, float& h, float& a, float& b, float& eps, float& sigma, float& u0L, float& u0R, float& w0L, float& w0R/*,
                                        ofstream & file*/);

template<typename MeshT, typename BufferT, typename DispatcherT>
void CreateStartVector(const MeshT & mesh, BufferT & startVec, const float & startValLeft, const float & startValRight,
                       const int & maxX, const int & maxY, DispatcherT & dispatcher);

template<typename MeshT, typename DispatcherT, typename BufferT>
void AssembleLumpedMassMatrix(const MeshT & mesh, DispatcherT & dispatcher, BufferT & lumpedMat) ;

template<typename MeshT, typename VectorT, typename DispatcherT, typename BufferT>
void AssembleMatrixVecProduct2D(const MeshT & mesh, const VectorT & d, DispatcherT & dispatcher, BufferT & sBuffer);

template<typename BufferT, typename VectorT, typename DispatcherT, typename MeshT, typename MutexT, typename OfstreamT>
void FWEuler(BufferT & vecOld, const VectorT & vecDeriv, const float & h, DispatcherT & dispatcher, const MeshT & mesh,
             const bool & optionWrite, MutexT & mutex, OfstreamT & fstream);

template<typename BufferT, typename BufferTGlobal, typename MeshT, typename DispatcherT>
void computeIionUDerivWDeriv(BufferT & f, BufferT & u_deriv, BufferT & w_deriv, const MeshT & mesh, DispatcherT & dispatcher,
                             const BufferTGlobal & u, const BufferT & w, const BufferT & lumpedM, const float & sigma,
                             const float & a, const float & b, const float & eps);

template<typename ArrayT, typename CharT>
void WriteFStreamToArray(const CharT * filename, ArrayT & array, mutex & mtx);

/*----------------------------------------------------------------- MAIN --------------------------------------------------------------------------------------*/
int main(int argc, char** argv)
{

    /*------------------------------------------(1) Set run-time system and read mesh information: ------------------------------------------------------------*/
    drts::Runtime<GetDistributedBuffer<>, UsingDistributedDevices> hpm({}, forward_as_tuple(argc, argv));
    DistributedDispatcher dispatcher{hpm.gaspi_context, hpm.gaspi_segment, hpm};
    ConfigParser CFG("config3D.cfg");
    string meshFile = CFG.GetValue<string>("MeshFile");
    const Mesh mesh      = Mesh::template CreateFromFile<AmiraMeshFileReader, ::HPM::mesh::MetisPartitioner>
            (meshFile, {hpm.GetL1PartitionNumber(), hpm.GetL2PartitionNumber()}, hpm.gaspi_runtime.rank().get());
    //Grid<3> grid {{ 5 , 5, 5 }};
    //const auto& mesh = grid.mesh;

    /*------------------------------------------(2) Set directory-,folder- and filename of result -------------------------------------------------------------*/
    char buff[FILENAME_MAX]; //create string buffer to hold path
    GetCurrentDir(buff, FILENAME_MAX);
    string currentWorkingDir(buff);

    string foldername = "TestForDistributedOutput";
    string filename   = "TestForDistributedOutput_";
    //string parameterFilename = "testParameterfile";

    /*------------------------------------------(3) Set start values ------------------------------------------------------------------------------------------*/
    int numIt = 10;//1000;
    float h; float a; float b; float eps; float sigma; float u0L; float u0R; float w0L; float w0R;
    //auto file = CreateFile(currentWorkingDir, foldername, parameterFilename);
    SetStartValues(1, h, a, b, eps, sigma, u0L, u0R, w0L, w0R/*, file*/);

    int numNodes = mesh.template GetNumEntities<0>();
    int maxX = ceil(sqrt(numNodes)/4);
    int maxY = ceil(sqrt(numNodes));

    Buffer<float, Mesh, Dofs<1, 0, 0, 0, 0>> u(mesh);
    CreateStartVector(mesh, u, u0L, u0R, maxX, maxY, dispatcher);

    Buffer<float, Mesh, Dofs<1, 0, 0, 0, 0>> w(mesh);
    CreateStartVector(mesh, w, w0L, w0R, maxX, maxY, dispatcher);

    Buffer<float, Mesh, Dofs<1, 0, 0, 0, 0>> u_deriv(mesh);
    Buffer<float, Mesh, Dofs<1, 0, 0, 0, 0>> w_deriv(mesh);
    Buffer<float, Mesh, Dofs<1, 0, 0, 0, 0>> f(mesh);

    /*------------------------------------------(4) Create monodomain problem ---------------------------------------------------------------------------------*/
    Buffer<float, Mesh, Dofs<1, 0, 0, 0, 0>> lumpedMat(mesh);
    AssembleLumpedMassMatrix(mesh, dispatcher, lumpedMat);

    // check if startvector was set correctly by creating output file at time step zero
    stringstream s; s << 0;
    string name = filename + s.str();
    writeVTKOutput2DTime(mesh, currentWorkingDir, foldername, name, u, "resultU");

    mutex mtx;
    //ofstream fstream {"testDist.txt"/*, ofstream::out | ofstream::trunc*/};
    Vector array; array.resize(numNodes);

    // compute u (and w)
    for (int j = 0; j < numIt; ++j)
    {
        stringstream s; s << j+1;
        string distFileName = "testDist" + s.str() + ".txt";
        ofstream fstream {distFileName};

        cout << "-----------------------------Iterationstep(u):   " << j << "---------------------------------------" << endl;

        computeIionUDerivWDeriv(f, u_deriv, w_deriv, mesh, dispatcher, u, w, lumpedMat, sigma, a, b, eps);
        FWEuler(u, u_deriv, h, dispatcher, mesh, true, mtx, fstream);
        FWEuler(w, w_deriv, h, dispatcher, mesh, false, mtx, fstream);
        fstream.close();

        // controll u
//        for (int ka = 0; ka < numNodes; ++ka)
//            cout << "u[" << ka << "]:  " << u[ka] << endl;

        //if ((j+1)%10 == 0)
        //{

        //stringstream s; s << j+1;
        //name = filename + s.str();
        //writeVTKOutput2DTime(mesh, currentWorkingDir, foldername, name, u, "resultU");
        //}

    }

    // write output files with result u
    for (int k = 0; k < numIt; ++k)
    {
        stringstream s; s << k+1;
        string distFileName = "testDist" + s.str() + ".txt";
        WriteFStreamToArray(distFileName.c_str(), array, mtx);

        cout << "-----------------------------Iterationstep(array):   " << k << "---------------------------------------" << endl;
        for (int ka = 0; ka < numNodes; ++ka)
            cout << "Array[" << ka << "]:  " << array[ka] << endl;

        //if ((k+1)%10 == 0)
        //{
            name = filename + s.str();
            writeVTKOutput2DTime(mesh, currentWorkingDir, foldername, name, array, "resultU");
        //}
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
    //    string fileName = "testParameterfile.txt";
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
template<typename MeshT, typename BufferT, typename DispatcherT>
void CreateStartVector(const MeshT & mesh, BufferT & startVec, const float & startValLeft, const float & startValRight, const int & maxX, const int & maxY, DispatcherT & dispatcher)
{
    auto nodes { mesh.template GetEntityRange<0>() };

    dispatcher.Execute(ForEachEntity(
                           nodes,
                           tuple(Write(Node(startVec))),
                           [&](auto const& node, const auto& iter, auto& lvs)
    {
        auto &startVec = HPM::dof::GetDofs<0>(std::get<0>(lvs));
        auto coords = node.GetTopology().GetVertices();
        if ( (coords[0][0] < maxX) && (coords[0][1] < maxY) )
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
        //const auto& gradients = GetGradients2DP1();
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
template<typename BufferT, typename VectorT, typename DispatcherT, typename MeshT, typename MutexT, typename OfstreamT>
void FWEuler(BufferT & vecOld, const VectorT & vecDeriv, const float & h, DispatcherT & dispatcher, const MeshT & mesh,
             const bool & optionWrite, MutexT & mutex, OfstreamT & fstream)
{
    //mutex mtx;
    //ofstream fstream { "test.txt" };

    auto vertices {mesh.template GetEntityRange<0>()};

    if (optionWrite) {
        dispatcher.Execute(
                    ForEachEntity(
                        vertices,
                        tuple(ReadWrite(Node(vecOld))),
                        [&](auto const& vertex, const auto& iter, auto& lvs) {
            vecOld[vertex.GetTopology().GetIndex()] += h*vecDeriv[vertex.GetTopology().GetIndex()];
            //cout<<"u :    "<< vecOld[vertex.GetTopology().GetIndex()] <<endl;
        }),
                    WriteLoop(mutex, fstream, vertices, vecOld)
                    );
    }
    else {
        dispatcher.Execute(
                    ForEachEntity(
                        vertices,
                        tuple(ReadWrite(Node(vecOld))),
                        [&](auto const& vertex, const auto& iter, auto& lvs) {
            vecOld[vertex.GetTopology().GetIndex()] += h*vecDeriv[vertex.GetTopology().GetIndex()];
        }));
    }

    return;
}

//!
//! \brief Compute ion current, derivation of u and derivation of w at time step t.
//!
template<typename BufferT, typename BufferTGlobal, typename MeshT, typename DispatcherT>
void computeIionUDerivWDeriv(BufferT & f, BufferT & u_deriv, BufferT & w_deriv, const MeshT & mesh, DispatcherT & dispatcher,
                             const BufferTGlobal & u, const BufferT & w, const BufferT & lumpedM, const float & sigma,
                             const float & a, const float & b, const float & eps)
{
    Buffer<float, Mesh, Dofs<1, 0, 0, 0, 0>> s(mesh);
    AssembleMatrixVecProduct2D(mesh, u, dispatcher, s);

    auto vertices {mesh.template GetEntityRange<0>()};
    dispatcher.Execute(ForEachEntity(vertices, tuple(
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
    std::lock_guard guard { mtx };
    //FILE* f = fopen("testDist.txt","r");
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
                //cout << "i:  " << ID << endl;

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
                //cout << "val:  " << val << endl;

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
