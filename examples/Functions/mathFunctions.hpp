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


//!
//! matrix-vector product split into single scalar operations
//!
/*template<typename MeshT, typename VectorT, typename LoopbodyT, typename BufferT>
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
                result[col]   += ((gc*gr) + val) * _d[row];
            }
        }

        // separate SCATTER (accumulate)
        for (int col = 0; col < ncols; ++col)
            sBuffer[nodeIdSet[col]] += result[col];
    }));

    return;
}*/
