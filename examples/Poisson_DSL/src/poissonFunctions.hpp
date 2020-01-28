/** ------------------------------------------------------------------------------ *
 * author(s)   : Merlind Schotte (schotte@zib.de)                                  *
 * institution : Zuse Institute Berlin (ZIB)                                       *
 * project     : HighPerMeshes (BMBF)                                              *
 *                                                                                 *
 * Description:                                                                    *
 * Implementation of some simple functions.                                        *
 *                                                                                 *
 * last change: 21.11.19                                                           *
 * -----------------------------------------------------------------------------  **/

#include </usr/include/c++/7/iostream>
#include <HighPerMeshes.hpp>



//!
//! \brief  Compute the cross product for different vector types.
//!
//! \param  p1, p2 input vectors (dim = 3)
//! \return vector, which is orthogonal to input vectors
//!
template <typename Vector>
inline Vector crossProduct(Vector p1, Vector p2)
{
    Vector result (3);
    if ( p1.size() != 3 || p2.size() != 3 )
        throw std::runtime_error("Wrong dimension of vectors!");
    else
    {
        result[0] =      (p1[1]*p2[2]) - (p2[1]*p1[2]) ;
        result[1] = -1* ((p1[0]*p2[2]) - (p2[0]*p1[2]));
        result[2] =      (p1[0]*p2[1]) - (p2[0]*p1[1]) ;
    }

    return result;
}

//!
//!  \brief  Compute l2 norm.
//!
//!  \param  p1 input vector (dim > 1)
//!  \return norm of input vector
//!
template <typename Vector>
inline float l2Norm(Vector p)
{
    float result = 0.F;
    for (int i = 0; i < p.size(); ++i)
        result += p[i]*p[i];
    return std::sqrt(result) ;
}

//!
//!  \brief  Compute determinant of 3x3 matrix using sarrus rule.
//!
//!  \param  A matrix
//!  \return determinant of A
//!
template <typename Matrix>
inline float detViaSarrus(Matrix A)
{
    float detA = (A[0][0]*A[1][1]*A[2][2])
               + (A[0][1]*A[1][2]*A[2][0])
               + (A[0][2]*A[1][0]*A[2][1])
               - (A[0][2]*A[1][1]*A[2][0])
               - (A[0][0]*A[1][2]*A[2][1])
               - (A[0][1]*A[1][0]*A[2][2]);

    return detA;
}

//!
//! \brief Matrix-vector product.
//!
//! \param matrix input matrix (mxn)
//! \param vec input vector (nx1 or 1xn)
//!         -> note: there is no! differentiation of row-/column-vectors
//! \return result vector (mx1/1xm)
//!
enum class MatrixFormat {RowMajor, ColumnMajor};
template <typename Vector,typename Matrix, MatrixFormat format = MatrixFormat::ColumnMajor>
inline Vector matrixVecProduct(Matrix matrix, Vector vec)
{
    if ( vec.size() != matrix[0].size() )
        throw std::runtime_error("In poissonFunctions.hpp at function matrixVecProduct()."
                                 "Number of matrix columns dismatch with the vector length!");
    else
    {
        Vector resultVec(vec.size());
        int size = matrix[0].size();
        for (int i = 0; i < size; ++i)
            for (int j = 0; j < matrix[0].size(); ++j)
                resultVec[i] += matrix[i][j] * vec[j];
        return resultVec;
    }
}

//!
//! \brief  Dot product.
//!
//! \param  vec1, vec2 input vectors which should have the same length
//! \return result of dot product
//!
template <typename Vector>
inline float dot_product(Vector vec1, Vector vec2)
{
    if ( vec1.size() != vec2.size() )
        throw std::runtime_error("In poissonFunctions.hpp at function <dot_Product()>"
                                 "The input vectors have not the same length!");
    else
    {
        float scalar = 0;
        for (size_t i = 0; i < vec1.size(); ++i)
                scalar += vec1[i] * vec2[i];
        return scalar;
    }
}

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
//! \brief  Compute local to global transformation of 1 element (here: Tetrahedron).
//!
//! \param  element input element
//! \return transformation of element from local into a "normalized" coordinate system
//! \note   normalized coordinate system means, that the coordinate system for all elements
//!         (here: Tetrahedrons) should be uniform. This makes possible to use always the
//!         same shape/ansatz functions in the weak formulation/variational functional.
//! \note   The points/nodes will be set by rows and not by columns:
//!                     |x1 y1 z1|
//!         jacTrans =  |x2 y2 z2|  .
//!                     |x3 y3 z3|
//!
template <typename Matrix>
inline Matrix JacobianTransformation3D(Matrix element)
{
    using Vector = std::vector<float>;
    Vector p0 =  element[0]; Vector p1 =  element[1]; // TODO: find a more elegant way of implementation
    Vector p2 =  element[2]; Vector p3 =  element[3]; // TODO: find a more elegant way of implementation

    Matrix jacobi = {{p1[0]-p0[0], p2[0]-p0[0], p3[0]-p0[0]},
                     {p1[1]-p0[1], p2[1]-p0[1], p3[1]-p0[1]},
                     {p1[2]-p0[2], p2[2]-p0[2], p3[2]-p0[2]}};

    return jacobi;
}

//!
//! \brief  Compute volume of a tetrahedron using determinant of the jacobian transformation.
//!
//! \param  element input element
//! \return volume of tetrahedron
//!
template <typename Matrix>
float getVolume(Matrix element)
{
    Matrix jacTrans = JacobianTransformation3D(element);
    return detViaSarrus(jacTrans);
}

//!
//! \brief  Transpose matrix.
//!
//! \param  mat input matrix
//! \return transposed matrix
//!
template <typename Matrix>
Matrix transposeMatrix(Matrix mat)
{
    size_t rows = mat[0].size();
    size_t cols = mat.size();
    Matrix transp; transp.resize(rows,std::vector<float>(cols));

    for(size_t i = 0; i < rows; ++i)
        for(size_t j = 0; j < cols; ++j)
            transp[i][j] = mat[j][i];
    return transp;
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
//! \brief  Inverte a quadratic matrix using adjoint matrix.
//!
//! \param  mat matrix to be inverted
//! \return inverse matrix of input matrix if existing
//!
template <typename Matrix>
Matrix invertMatrixAdj(const Matrix mat)
{
    using ElementType = typename Matrix::value_type::value_type;

    if ( mat.size() != mat[0].size())
    {
        std::cout<<"poissonFunctions::invertMatrixAdj(): Matrix is not quadratic!"<< std::endl;
        return mat;
    }

    std::size_t nrows = mat.size(); std::size_t ncols = mat[0].size();
    Matrix inv; inv.resize(ncols, std::vector<float>(nrows));

    ElementType det = 0.0;
    if (nrows == 2)
        det = (mat[0][0]*mat[1][1])-(mat[0][1]*mat[1][0]);
    else if (nrows == 3)
        det = detViaSarrus(mat);
    else
    {
        std::cout<<"Not Implemented!Use an other algorithm to compute the inverse!" << std::endl;
        return inv;
    }

    if (det==0.0)
    {
        std::cout<<"poissonFunctions::invertMatrixAdj(): Determinante = 0! Their exist no inverse" << std::endl;
        return inv;
    }

    if (nrows == 2)
    {
        inv = Matrix{{ mat[1][1]/det, -mat[0][1]/det},
               {-mat[1][0]/det,  mat[0][0]/det}};
    }
    else if (nrows == 3)
    {   // use sarrows rule and det() of submatrizes to compute "minoren"
        auto a = mat[0][0]; auto b = mat[0][1]; auto c = mat[0][2];
        auto d = mat[1][0]; auto e = mat[1][1]; auto f = mat[1][2];
        auto g = mat[2][0]; auto h = mat[2][1]; auto i = mat[2][2];

        inv = {{((e*i)-(f*h))/det,((c*h)-(b*i))/det,((b*f)-(c*e))/det},
               {((f*g)-(d*i))/det,((a*i)-(c*g))/det,((c*d)-(a*f))/det},
               {((d*h)-(e*g))/det,((b*g)-(a*h))/det,((a*e)-(b*d))/det}};
    }
    return inv;
}

//!
//! \brief  Scalar-vector multiplication.
//!
//! \param  scalar value of scalar
//! \param  vec vector
//! \return result vector
//!
template <typename Vector>
Vector scalarVecMultiplication(const size_t scalar, const Vector vec )
{
    Vector vecRes; vecRes.resize(vec.size());
    for (int i = 0; i < vec.size(); ++i)
        vecRes[i]= scalar*vec[i];
    return vecRes;
}

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
//! \return gradients (p1-elements, dim = 3)
//!
auto GetGradientsDSL()
{
    HPM::dataType::Matrix<float,4,3> gradientsDSL;
    gradientsDSL[0][0]= -1; gradientsDSL[0][1]= -1; gradientsDSL[0][2]= -1;
    gradientsDSL[1][0]=  1; gradientsDSL[1][1]=  0; gradientsDSL[1][2]=  0;
    gradientsDSL[2][0]=  0; gradientsDSL[2][1]=  1; gradientsDSL[2][2]=  0;
    gradientsDSL[3][0]=  0; gradientsDSL[3][1]=  0; gradientsDSL[3][2]=  1;
    return gradientsDSL;
}
