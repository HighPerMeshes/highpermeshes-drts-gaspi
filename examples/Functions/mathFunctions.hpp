/** ------------------------------------------------------------------------------ *
 * author(s)   : Merlind Schotte (schotte@zib.de)                                  *
 * institution : Zuse Institute Berlin (ZIB)                                       *
 * project     : HighPerMeshes (BMBF)                                              *
 *                                                                                 *
 * Description:                                                                    *
 * Implementation of some simple functions.                                        *
 *                                                                                 *
 * last change: 19.05.20                                                           *
 * -----------------------------------------------------------------------------  **/

#include </usr/include/c++/7/iostream>
#include <HighPerMeshes.hpp>

//!
//! \brief Matrix-vector product.
//!
//! \param matrix input matrix (mxn)
//! \param vec input vector (nx1 or 1xn)
//!         -> note: there is no! differentiation of row-/column-vectors
//! \return result vector (mx1/1xm)
//!
enum class MatrixFormat {RowMajor, ColumnMajor};
template <typename BufferT,typename Matrix, MatrixFormat format = MatrixFormat::ColumnMajor>
inline auto matrixVecProduct(Matrix matrix, BufferT vec, int sizeVec) -> std::vector<float>
{
    if ( sizeVec != matrix[0].size() )
        throw std::runtime_error("In mathFunctions.hpp at function matrixVecProduct()."
                                 "Number of matrix columns dismatch with the vector length!");
    else
    {
        std::vector<float> resultVec; resultVec.resize(sizeVec);
        int size = matrix[0].size();
        for (int i = 0; i < size; ++i)
            for (int j = 0; j < matrix[0].size(); ++j)
                resultVec[i] += matrix[i][j] * vec[j];
        return resultVec;
    }
}

//!
//! \brief scalar / dot product.
//!
template<typename VectorT1, typename VectorT2>
float mv(const VectorT1 & a, const VectorT2 & b)
{
    float m = 0;
    for (int i = 0; i < a.size(); ++i)
        m += a[i]*b[i];

    return m;
}

//!
//! \brief scalar * vector.
//!
template<typename ScalarT, typename VectorT>
VectorT msv(const ScalarT & a, const VectorT & v)
{
    VectorT vec; vec.resize(v.size());
    for (int i = 0; i < v.size(); ++i)
        vec[i] = a*v[i];

    return vec;
}

//!
//! \brief vector - vector.
//!
template<typename VectorT1, typename VectorT2>
auto minus(const VectorT1 & a, const VectorT2 & b) -> VectorT1
{
    VectorT1 vec; vec.resize(a.size());
    for (int i = 0; i < b.size(); ++i)
        vec[i] = a[i]-b[i];

    return vec;
}

//!
//! \brief vector + vector.
//!
template<typename VectorT1, typename VectorT2>
auto plus(const VectorT1 & a, const VectorT2 & b) -> VectorT1
{
    VectorT1 vec; vec.resize(a.size());
    for (int i = 0; i < b.size(); ++i)
        vec[i] = a[i]+b[i];

    return vec;
}
