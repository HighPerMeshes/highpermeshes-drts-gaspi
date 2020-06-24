#include "Environment.hpp"

int main(int argc, char **argv)
{

  Environment e { argc, argv, "matrix-vec-omp" };
  auto MeasureKernel = GetMeasureKernel(e.dispatcher  );

  constexpr int dim = Mesh::CellDimension;
  HPM::dataType::Vec<double, 8> d;

  auto buffer = e.hpm.GetBuffer<double>(e.mesh, ::HPM::dof::MakeDofs<1, 0, 0, 0, 0>());

  auto matrix_vec_product =
      HPM::ForEachEntity(
          e.AllCells,
          std::tuple(ReadWrite(Node(buffer))),
          [&](auto const &cell, const auto &iter, auto &lvs) {
            auto &sBuffer = dof::GetDofs<0>(std::get<0>(lvs));

            const int nrows = dim + 1;
            const int ncols = dim + 1;
            const auto gradients = GetGradientsDSL();
            const auto &nodeIdSet = std::array{0, 1, 2, 3};

            auto tmp = cell.GetGeometry().GetJacobian();
            double detJ = tmp.Determinant();
            detJ = std::abs(detJ);
            auto inv = tmp.Invert();
            auto invJT = inv.Transpose();

            double val = 0;

            // Material information is integrate as diffusion tensor D = sigma * I with sigma as random scalar value.
            // For example: sigma = 2
            double sigma = 2;

            for (int col = 0; col < ncols; ++col)
            {
              auto gc = invJT * gradients[col] * sigma * (detJ / 6);
              for (int row = 0; row < nrows; ++row)
              {
                // add mass (matrix) term
                if (col == row)
                  val = detJ / 60;
                else
                  val = detJ / 120;

                auto gr = invJT * gradients[row];
                sBuffer[nodeIdSet[col]][0] += ((gc * gr) + val) * d[nodeIdSet[row]];
              }
            }
          });

  auto matrix_vec_product_openmp =
    HPM::MeshLoop{
        e.AllCells,
        matrix_vec_product.access_definitions,
        HPM::internal::OpenMP_ForEachEntity<3>{},
        matrix_vec_product.loop_body};

  print_time(MeasureKernel(matrix_vec_product_openmp));

  return EXIT_SUCCESS;
}
