#include "Environment.hpp"

int main(int argc, char **argv)
{

  Environment e { argc, argv, "surface" };
  auto MeasureKernel = GetMeasureKernel(e.dispatcher  );

  auto fieldH = e.hpm.GetBuffer<Vec3D>(e.mesh, Dofs);
  auto fieldE = e.hpm.GetBuffer<Vec3D>(e.mesh, Dofs);
  auto rhsH = e.hpm.GetBuffer<Vec3D>(e.mesh, Dofs);
  auto rhsE = e.hpm.GetBuffer<Vec3D>(e.mesh, Dofs);

  using DG = DgNodes<double, Vec3D, order>;
  HPM::DG::DgNodesMap<DG, Mesh> DgNodeMap(e.mesh);

  auto surface_kernel = HPM::ForEachIncidence<2>(
      e.AllCells,
      std::tuple(
          Read(ContainingMeshElement(fieldH)),
          Read(ContainingMeshElement(fieldE)),
          Read(NeighboringMeshElementOrSelf(fieldH)),
          Read(NeighboringMeshElementOrSelf(fieldE)),
          ReadWrite(ContainingMeshElement(rhsH)),
          ReadWrite(ContainingMeshElement(rhsE))),
      [&](const auto &element, const auto &face, const auto &, auto &lvs) {
        const std::size_t face_index = face.GetTopology().GetLocalIndex();
        const RealType face_normal_scaling_factor = 2.0 / element.GetGeometry().GetAbsJacobianDeterminant();
        const Vec3D &face_normal = face.GetGeometry().GetNormal() * face_normal_scaling_factor; //!< get all normal coordinates for each face of an element
        const RealType Edg = face_normal.Norm() * 0.5;                                          //!< get edge length for each face
        const Vec3D &face_unit_normal = face.GetGeometry().GetUnitNormal();
        const auto &localMap{DgNodeMap.Get(element, face)};

        HPM::ForEach(DG::NumSurfaceNodes, [&](const std::size_t m) {
          const auto &fieldH = std::get<0>(lvs);
          const auto &fieldE = std::get<1>(lvs);

          auto &NeighboringFieldH = std::get<2>(lvs);
          auto &NeighboringFieldE = std::get<3>(lvs);

          const Vec3D &dH = Edg * HPM::DG::Delta(fieldH, NeighboringFieldH, m, localMap); //!< fields differences
          const Vec3D &dE = Edg * HPM::DG::DirectionalDelta(fieldE, NeighboringFieldE, face, m, localMap);

          const Vec3D &flux_H = (dH - (dH * face_unit_normal) * face_unit_normal - CrossProduct(face_unit_normal, dE)); //!< fields fluxes
          const Vec3D &flux_E = (dE - (dE * face_unit_normal) * face_unit_normal + CrossProduct(face_unit_normal, dH));

          auto &rhsH = std::get<4>(lvs);
          auto &rhsE = std::get<5>(lvs);

          HPM::ForEach(DG::numVolNodes, [&](const std::size_t n) {
            rhsH[n] += DG::LIFT[face_index][m][n] * flux_H;
            rhsE[n] += DG::LIFT[face_index][m][n] * flux_E;
          });
        });
      });

  print_time(MeasureKernel(surface_kernel));

  return EXIT_SUCCESS;
}
