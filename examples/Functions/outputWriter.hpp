/** ------------------------------------------------------------------------------ *
 * author(s)   : Merlind Schotte (schotte@zib.de)                                  *
 * institution : Zuse Institute Berlin (ZIB)                                       *
 * project     : HighPerMeshes (BMBF)                                              *
 *                                                                                 *
 * Description:                                                                    *
 * Implementation of some simple output functions.                                 *
 *                                                                                 *
 * last change: 19.05.20                                                           *
 * -----------------------------------------------------------------------------  **/

#include </usr/include/c++/7/iostream>
#include <HighPerMeshes.hpp>

//!
//! \brief  Output of input vector at terminal/console.
//!
//! \tparam the type of the vector
//! \param  vec vector which should be outputed in terminal
//! \param  name name of vector
//! \param  vecSize size of vector
//!
template <typename VecT>
void outputVec(const VecT & vec, const std::string name, const int vecSize)
{
    std::cout<<"---------------------"<<name<<"----------------------------------------"<<std::endl;
    std::cout<<"size() of "<< name <<":  " << vecSize << std::endl;
    for (int i = 0; i < vecSize; ++i)
        std::cout<<"[ " << i << "]: " << vec[i] << std::endl;
    return;
}

//!
//! \brief  Output of input matrix at terminal/console.
//!
//! \tparam the type of the matrix
//! \param  mat matrix which should be outputed in terminal
//! \param  name name of matrix
//! \param  rows number of matrix rows
//! \param  columns number of matrix columns
//!
template <typename MatT>
void outputMat(const MatT & mat, const std::string & name,
               const int & rows, const int & columns)
{
    std::cout<<"---------------------"<<name<<"----------------------------------------"<<std::endl;
    std::cout<<"Row size of "<<name<<": " << rows << std::endl;
    std::cout<<"         ";

    for (int a = 0; a < columns; ++a)
    {
        if (a==0)
            std::cout<<"[" << a << "] "<< "\t";
        else
            std::cout<<" "<<"[" << a << "] "<< "\t";
    }
    std::cout<<""<<std::endl;

    for (int i = 0; i < rows; ++i)
    {
        std::cout<<"[" << i << "]: ";
        for (int j = 0; j < columns; ++j)
        {   if (mat[i][j] >= 0 && mat[i][j] < 10)
                std::cout<<"\t"<<"  "<< mat[i][j];
            else if (mat[i][j] < 0 && mat[i][j] > -10)
                std::cout<<"\t" <<" "<< mat[i][j];
            else
                std::cout<<"\t"<< mat[i][j];
        }
        std::cout<<""<<std::endl;
    }

    return;
}


//!
//! \brief  Output of vtk file (without parser / distributed case).
//!
//! \param mesh
//! \param filename
//! \param resultVec
//! \param nameOfResultVec
//! \param homDirichletNodes
//!
/*template <typename MeshT, typename VectorT, typename Runtime>
void writeVTKOutput(const MeshT & mesh, std::string const & filename, const VectorT& resultVec, std::string const nameOfResultVec,
                    std::vector<int> homDirichletNodes, const Runtime& rt)
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
      << "              <DataArray type=\"Float64\" Name=\"Coordinates\" NumberOfComponents=\""<<dim<<"\" format=\"ascii\">" << '\n'
      << "              ";

    //add node coordinates to file
    for (const auto& node : mesh.template GetEntities<0>() )
    {
        auto nodeCoords = node.GetTopology().GetVertices();
        int nodeID      = node.GetTopology().GetIndex();
        for (int j = 0; j < dim; ++j)
            f << nodeCoords[0][j] << ' ';
        if ((dim*(nodeID+1)) % 12 == 0)
            f << '\n';
        if (((dim*(nodeID+1)) % 12 == 0) && (nodeID+1)!=numNodes)
            f << "              ";
    }

    if ((dim*(numNodes)) % 12 != 0)
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
      << "          <PointData Scalars=\""<<nameOfResultVec<<"\">" << '\n'
      << "              <DataArray type=\"Float64\" Name=\""<<nameOfResultVec<<"\" NumberOfComponents=\"1\" format=\"ascii\">" << '\n'
      << "              ";

    int counterA = 0;
    int counterB = 0;
    const int id = rt.gaspi_runtime.rank().get();
    std::cout << "Size: " << resultVec.size() << std::endl;
    if (homDirichletNodes.size() != 0)
        for (int i = 0; i < numNodes; i++)
        {
            if (homDirichletNodes[counterA] == i)
            {
                f << 0.0 << ' ';
                ++counterA;
            }
            else
            {
                double val = resultVec.at(counterB);
                std::cout<<"resultVec:   "<<id<<":"<<val<<std::endl;
                f << id <<":"<<val << ' ';
                ++counterB;
            }

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
}*/
