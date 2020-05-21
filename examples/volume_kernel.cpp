#include "Environment.hpp"

int main(int argc, char **argv)
{

  Environment e { argc, argv, "volume" };
  auto MeasureKernel = GetMeasureKernel(e.dispatcher);

  auto fieldH = e.hpm.GetBuffer<Vec3D>(e.mesh, Dofs);
  auto fieldE = e.hpm.GetBuffer<Vec3D>(e.mesh, Dofs);
  auto rhsH = e.hpm.GetBuffer<Vec3D>(e.mesh, Dofs);
  auto rhsE = e.hpm.GetBuffer<Vec3D>(e.mesh, Dofs);

  using DG = DgNodes<double, Vec3D, order>;

  std::chrono::nanoseconds in_kernel { 0 };

  auto volume_kernel = HPM::ForEachEntity(
      e.AllCells,
      std::tuple(
          Read(Cell(fieldH)),
          Read(Cell(fieldE)),
          ReadWrite(Cell(rhsH)),
          ReadWrite(Cell(rhsE))),
      [&](const auto &element, const auto &, auto &lvs) {
        
	in_kernel += HPM::auxiliary::MeasureTime(
                                                                     [&]() {


	const Mat3D &D = element.GetGeometry().GetInverseJacobian() * 2.0;
	
        HPM::ForEach(DG::numVolNodes, [&](const std::size_t n) {
          Mat3D derivative_E, derivative_H; //!< derivative of fields w.r.t reference coordinates

          const auto &fieldH = dof::GetDofs<dof::Name::Cell>(std::get<0>(lvs));
          const auto &fieldE = dof::GetDofs<dof::Name::Cell>(std::get<1>(lvs));

          HPM::ForEach(DG::numVolNodes, [&](const std::size_t m) {
            derivative_H += DyadicProduct(DG::derivative[n][m], fieldH[m]);
            derivative_E += DyadicProduct(DG::derivative[n][m], fieldE[m]);
          });

          auto &rhsH = dof::GetDofs<dof::Name::Cell>(std::get<2>(lvs));
          auto &rhsE = dof::GetDofs<dof::Name::Cell>(std::get<3>(lvs));

          rhsH[n] += -Curl(D, derivative_E); //!< first half of right-hand-side of fields
          rhsE[n] += Curl(D, derivative_H);
	
        });

		});
     
});

  print_time(MeasureKernel(volume_kernel));
  std::cout << "In Kernel = " << std::chrono::duration_cast<std::chrono::milliseconds>(in_kernel).count() << " ms\n";

  return EXIT_SUCCESS;
}
