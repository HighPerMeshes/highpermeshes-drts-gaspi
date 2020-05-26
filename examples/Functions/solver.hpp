/** ------------------------------------------------------------------------------ *
 * author(s)   : Merlind Schotte (schotte@zib.de)                                  *
 * institution : Zuse Institute Berlin (ZIB)                                       *
 * project     : HighPerMeshes (BMBF)                                              *
 *                                                                                 *
 * Description:                                                                    *
 * Implementation of some simple solver.                                           *
 *                                                                                 *
 * last change: 17.03.2020                                                         *
 * -----------------------------------------------------------------------------  **/

#include </usr/include/c++/7/iostream>
#include <HighPerMeshes.hpp>


//!
//! \brief Jacobi solver for quadratic matrix A
//!
//! \param A matrix (nxn)
//! \param b vector of right-hand-side (nx1)
//! \param xStart start vector(nx1),
//! \param maxIt number of maximal iterations
//! \param bSize size of right-hand-side vector
//! \return result vector
//!
template <typename MatrixType, typename BufferType, typename VectorType>
inline VectorType JacobiSolver(MatrixType A, BufferType b, VectorType xStart, int maxIt, int bSize)
{
    VectorType x; x.resize(bSize);
    int m = 0;

    while (m < maxIt)
    {
        for (int id = 0; id < bSize; ++id)
            x[id]=b[id];

        for (int i = 0; i < bSize; ++i)
        {
            for (int j = 0; j < bSize; ++j)
            {
                if (j != i)
                    x[i] = x[i] - A[i][j]*xStart[j];
            }
            x[i] = x[i]/A[i][i];
        }
        xStart = x;
        ++m;
    }

    return xStart;
}

//!
//! \brief  Swap two rows of a matrix.
//!
//! \param  mat input matrix
//! \param  line1 and line2 the Ids of the two lines to be swaped
//! \return matrix with two swaped lines
//! \ref    https://www.virtual-maxim.de/matrix-invertieren-in-c-plus-plus/
//!
template <typename Matrix>
bool swapLine(Matrix mat, std::size_t line1, std::size_t line2)
{
    std::size_t N = mat.size(); std::size_t M = 2*mat.size();
    if(line1 >= N || line2 >= N)
    {
        std::cout<<"Error: swapLine()"<<std::endl;
        return false;
    }

    for(std::size_t i = 0; i < M; ++i)
    {
        double t = mat[line1][i];
        mat[line1][i] = mat[line2][i];
        mat[line2][i] = t;
    }
    return true;
}

//!
//! \brief  Inverte a quadratic matrix using Gauß-Jordan-Algorithm.
//!
//! \param  mat matrix to be inverted
//! \return inverse matrix of input matrix if existing
//!
template <typename Matrix>
Matrix invertMatrix(const Matrix mat)
{
    std::size_t nrows = mat.size(); std::size_t ncols = mat[0].size();
    Matrix inv; inv.resize(ncols, std::vector<float>(nrows));

    // Eine Nx2N Matrix für den Gauß-Jordan-Algorithmus aufbauen
    std::size_t N = mat.size();
    std::vector<std::vector<double>> A; A.resize(N,std::vector<double>(2*N));
    for(size_t i = 0; i < N; ++i)
    {
        for(size_t j = 0; j < N; ++j)
            A[i][j] = mat[i][j];
        for(size_t j = N; j < 2*N; ++j)
            A[i][j] = (i==j-N) ? 1.0 : 0.0;
    }

    // Gauß-Algorithmus.
    for(size_t k = 0; k < N-1; ++k)
    {
        // Zeilen vertauschen, falls das Pivotelement eine Null ist
        if(A[k][k] == 0.0)
        {
            for(size_t i = k+1; i < N; ++i)
            {
                if(A[i][k] != 0.0)
                {
                    swapLine(A,k,i);
                    break;
                }
                else if(i==N-1)
                {
                    return mat; // Es gibt kein Element != 0
                }

            }
        }

        // Eintraege unter dem Pivotelement eliminieren
        for(size_t i = k+1; i < N; ++i)
        {
            double p = A[i][k]/A[k][k];
            if (A[k][k]==0.0)
            std::cout<<"Attention!! Wrong computation of inverse matrix!"
                       "Divison by zero!Use an other algorithm to compute the inverse!"<<std::endl;
            for(size_t j = k; j < 2*N; ++j)
            {
                if (A[k][j] != 0.0)
                    A[i][j] -= A[k][j]*p;
            }
        }
    }

    // Determinante der Matrix berechnen
    double det = 1.0;
    for(size_t k = 0; k < N; ++k)
    {
        det *= A[k][k];
    }

    std::cout<<"det: "<<det<<std::endl;
    if(det == 0.0)  // Determinante ist =0 -> Matrix nicht invertierbar
    {
        std::cout<<"Error: invertMatrix(), 2"<<std::endl;
        return mat/*false*/;
    }

    // Jordan-Teil des Algorithmus durchfuehren
    for(int k = N-1; k > 0; --k)
    {
        for(int i = k-1; i >= 0; --i)
        {
            double p = A[i][k]/A[k][k];
            for(int j = k; j < 2*N; ++j)
            {
                if (A[k][j] != 0.0)
                    A[i][j] -= A[k][j]*p;
            }
        }
    }

    // Eintraege in der linker Matrix auf 1 normieren und in inv schreiben
    for(size_t i = 0; i < N; ++i)
    {
        double f = A[i][i];
        for(size_t j = N; j < 2*N; ++j)
        {   double tmp   = A[i][j]/f;
            inv[i][j-N] = tmp;
        }
    }

    return inv;
}


//!
//! matrix-free conjugated gradient method 
//!
/*template<typename MeshT, typename BufferT, typename LoopbodyT, typename VecDslT>
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
}*/
