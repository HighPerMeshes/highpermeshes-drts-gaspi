// remove all DUNE dependencies from Jakob Schenk's gridIteration.hh implementation
// performance results are no longer hurt
// merge version created from midg_cpp_modified and gridIteration.hh by Ayesha Afzal

#include <array>
#include <vector>
#include <fstream>
#include <sstream>
#include <iostream>
#include <string>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <chrono>
#include <numeric>

#include "data3dN03.hpp" //!< application-dependent discontinuous Galerkin's cubic order node information
#include "RKCoeff.hpp"   //!< application-dependent Runge-Kutta coefficients

#include <HighPerMeshes.hpp>
#include <HighPerMeshes/third_party/metis/Partitioner.hpp>
#include <HighPerMeshesDRTS.hpp>

using CoordinateType = HPM::dataType::Coord3D;
using RealType = HPM::dataType::Real;
using Vec3D = HPM::dataType::Vec3D;
using Mat3D = HPM::dataType::Mat3D;

using namespace HPM;
using namespace HPM::internal;

int main(int argc, char **argv)
{
    //////////////////////////////////////////////////////////////////////////////////////////////////////////
    //                         Initializing the runtime system                                              //
    //////////////////////////////////////////////////////////////////////////////////////////////////////////
    //    HPM::drts::Runtime<HPM::GetDistributedBuffer, HPM::UsingACE, HPM::UsingGaspi> hpm;
    //HPM::drts::Runtime<HPM::GetDistributedBuffer<>, HPM::UsingDistributedDevices> hpm( {}, std::forward_as_tuple(argc, argv));
    HPM::drts::Runtime<HPM::GetBuffer<>, HPM::UsingDistributedDevices> hpm( {}, std::forward_as_tuple(argc, argv));

    //////////////////////////////////////////////////////////////////////////////////////////////////////////
    //                         Reading Configuration, DG and Mesh Files                                     //
    //////////////////////////////////////////////////////////////////////////////////////////////////////////
    auto maint1 = std::chrono::high_resolution_clock::now();

    /** \brief read configuration file */
    HPM::auxiliary::ConfigParser CFG("config.cfg");
    const RealType startTime = CFG.GetValue<RealType>("StartTime"); //!< get the value of a user-specific starting time
    const RealType finalTime = CFG.GetValue<RealType>("FinalTime"); //!< get the value of a user-specific stop time

    /** \brief read mesh file */
    const std::string meshFile = CFG.GetValue<std::string>("MeshFile"); //!< get the name of a user-specific mesh file
    //const Mesh mesh = Mesh::template CreateFromFile<HPM::auxiliary::GambitMeshFileReader>(meshFile, {1, 1});
    //const Mesh mesh = Mesh::template CreateFromFile<HPM::auxiliary::GambitMeshFileReader, HPM::mesh::MetisPartitioner>(meshFile, {hpm.GetL1PartitionNumber(), hpm.GetL2PartitionNumber()});
    HPM::mesh::MetisPartitioner partitioner;
    HPM::auxiliary::GambitMeshFileReader<CoordinateType, std::array<std::size_t, 4>> reader;
    
    using Mesh = HPM::mesh::PartitionedMesh<CoordinateType, HPM::entity::Simplex>;
    const Mesh mesh = Mesh::template CreateFromFile(meshFile, reader, {hpm.GetL1PartitionNumber(), hpm.GetL2PartitionNumber()}, hpm.gaspi_runtime.rank().get(), partitioner);

    /** \brief read application-dependent discontinuous Galerkin's stuff */
    constexpr std::size_t order = 3;
    using DG = DgNodes<RealType, Vec3D, order>;
    HPM::DG::DgNodesMap<DG, Mesh> DgNodeMap(mesh);


    auto AllCells { mesh.GetEntityRange<Mesh::CellDimension>() } ;    

    //////////////////////////////////////////////////////////////////////////////////////////////////////////
    //    All three kernels (Maxwell's Surface Kernel, Maxwell's Volume Kernel, Runge-Kutta kernel)         //
    //////////////////////////////////////////////////////////////////////////////////////////////////////////
    /** \brief load initial conditions for fields */
    constexpr auto Dofs = ::HPM::dof::MakeDofs<0, 0, 0, DG::numVolNodes, 1>();
    //auto Dofs = ::HPM::dof::MakeDofs(0, 0, 0, (argc > 1 ? std::atoi(argv[1]) : 20), 0);


    using Iterator = ::HPM::iterator::Iterator<std::size_t>;

    Iterator zero{0};
    Iterator one{1};

    auto fieldH(hpm.GetBuffer<CoordinateType>(mesh, Dofs));
    auto fieldE(hpm.GetBuffer<CoordinateType>(mesh, Dofs));
    
    HPM::DistributedDispatcher body{hpm.gaspi_context, hpm.gaspi_segment, hpm};
    body.Execute(
        HPM::ForEachEntity(
        AllCells,
        std::tuple(Write(Cell(fieldE))),
        [&](const auto &cell, auto &&, auto& localVector0) {
            HPM::ForEach(DG::numVolNodes, [&](const auto &n) {
                auto &fieldE = dof::GetDofs<dof::Name::Cell>(std::get<0>(localVector0));
                const auto &nodeCoords = DG::LocalToGlobal(DG::referenceCoords[n], cell.GetTopology().GetNodes());
                fieldE[n].y = sin(M_PI * nodeCoords.x) * sin(M_PI * nodeCoords.z); //!< initial conditions for y component of electric field
            });
        })
    );

    /** \brief create storage for intermediate fields*/
    auto resH(hpm.GetBuffer<CoordinateType>(mesh, Dofs));
    auto resE(hpm.GetBuffer<CoordinateType>(mesh, Dofs));
    auto rhsH(hpm.GetBuffer<CoordinateType>(mesh, Dofs));
    auto rhsE(hpm.GetBuffer<CoordinateType>(mesh, Dofs));
    
    auto maint2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> setup_duration = maint2 - maint1;
    std::cout << "Setup time in seconds: " << setup_duration.count() << std::endl;
    double aggregate_time1 = 0.0, aggregate_time2 = 0.0, aggregate_time3 = 0.0;

    /** \brief outer time step loop, Runge-Kutta loop, maxwell's kernels (surface and volume) and Runge-Kutta kernel  */

    /** \brief determine time step size (polynomial order-based and algorithmic-specific) */
    
    HPM::dataType::MinType<double> MinTimeStep;
    body.Execute(
        HPM::ForEachEntity(
        AllCells,
        std::tuple {},
        [&](const auto &cell, auto&&, auto&&) {
            const RealType face_normal_scaling_factor = 2.0 / cell.GetGeometry().GetAbsJacobianDeterminant();

            HPM::ForEachSubEntity(cell, [&](const auto &face) {
                MinTimeStep.Update(1.0 / (face.GetGeometry().GetNormal() * face_normal_scaling_factor).Norm());
            });
    }));

    RealType timeStep = finalTime / floor(finalTime * (order + 1) * (order + 1) / (.5 * MinTimeStep.Get()));
    std::cout << "time step: " << timeStep << std::endl;

    {
        auto t1 = std::chrono::high_resolution_clock::now();

        /** \brief Maxwell's surface kernel */
        auto surfaceKernelLoop = HPM::ForEachIncidence<2>(
            AllCells,
            std::tuple(
                Read(ContainingMeshElement(fieldH)),
                Read(ContainingMeshElement(fieldE)),
                Read(NeighboringMeshElementOrSelf(fieldH)),
                Read(NeighboringMeshElementOrSelf(fieldE)),
                ReadWrite(ContainingMeshElement(rhsH)),
                ReadWrite(ContainingMeshElement(rhsE))),
            [&](const auto &element, const auto &face, const auto&, auto &lvs) {
                const std::size_t face_index = face.GetTopology().GetLocalIndex();
                const RealType face_normal_scaling_factor = 2.0 / element.GetGeometry().GetAbsJacobianDeterminant();
                const Vec3D &face_normal = face.GetGeometry().GetNormal() * face_normal_scaling_factor; //!< get all normal coordinates for each face of an element
                const RealType Edg = face_normal.Norm() * 0.5;                                          //!< get edge length for each face
                const Vec3D &face_unit_normal = face.GetGeometry().GetUnitNormal();
                const auto &localMap{DgNodeMap.Get(element, face)};

                HPM::ForEach(DG::NumSurfaceNodes, [&](const std::size_t m) {
                    const auto &fieldH = dof::GetDofs<dof::Name::Cell>(std::get<0>(lvs));
                    const auto &fieldE = dof::GetDofs<dof::Name::Cell>(std::get<1>(lvs));

                    auto &NeighboringFieldH = dof::GetDofs<dof::Name::Cell>(std::get<2>(lvs));
                    auto &NeighboringFieldE = dof::GetDofs<dof::Name::Cell>(std::get<3>(lvs));

                    const Vec3D &dH = Edg * HPM::DG::Delta(fieldH, NeighboringFieldH, m, localMap); //!< fields differences
                    const Vec3D &dE = Edg * HPM::DG::DirectionalDelta(fieldE, NeighboringFieldE, face, m, localMap);

                    //const Vec3D &flux_H = (dH - DotProduct(dH, face_unit_normal) * face_unit_normal - CrossProduct(face_unit_normal, dE)); //!< fields fluxes
                    //const Vec3D &flux_E = (dE - DotProduct(dE, face_unit_normal) * face_unit_normal + CrossProduct(face_unit_normal, dH));
                    const Vec3D &flux_H = (dH - (dH*face_unit_normal) * face_unit_normal - CrossProduct(face_unit_normal, dE)); //!< fields fluxes
                    const Vec3D &flux_E = (dE - (dE*face_unit_normal) * face_unit_normal + CrossProduct(face_unit_normal, dH));

                    auto &rhsH = dof::GetDofs<dof::Name::Cell>(std::get<4>(lvs));
                    auto &rhsE = dof::GetDofs<dof::Name::Cell>(std::get<5>(lvs));

                    HPM::ForEach(DG::numVolNodes, [&](const std::size_t n) {
                        rhsH[n] += DG::LIFT[face_index][m][n] * flux_H;
                        rhsE[n] += DG::LIFT[face_index][m][n] * flux_E;
                    });
                });
            }, HPM::internal::OpenMP_ForEachIncidence<3, 2> {}
            );

        auto t2 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = t2 - t1;
        aggregate_time1 += duration.count();
        t1 = std::chrono::high_resolution_clock::now();

        /** \brief Maxwell's volume kernel */
        auto volumeKernelLoop = HPM::ForEachEntity(
            AllCells,
            std::tuple(
                Read(Cell(fieldH)),
                Read(Cell(fieldE)),
                ReadWrite(Cell(rhsH)),
                ReadWrite(Cell(rhsE))),
            [&](const auto &element, const auto&, auto &lvs) {
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
            }, HPM::internal::OpenMP_ForEachEntity<3> {}
            );

        t2 = std::chrono::high_resolution_clock::now();
        duration = t2 - t1;
        aggregate_time2 += duration.count();
        t1 = std::chrono::high_resolution_clock::now();

        /** \brief Runge-Kutta integrtion kernel */
        auto rungeKuttaLoop =
            HPM::ForEachEntity(
                AllCells,
                std::tuple(
                    ReadWrite(Cell(fieldH)),
                    ReadWrite(Cell(fieldE)),
                    ReadWrite(Cell(rhsH)),
                    ReadWrite(Cell(rhsE)),
                    ReadWrite(Cell(resH)),
                    ReadWrite(Cell(resE))),
                [&](const auto&, const auto &iter, auto &lvs) {
                    const auto& RKstage = RungeKuttaCoeff<RealType>::rk4[iter % 5];

                    auto &fieldH = dof::GetDofs<dof::Name::Cell>(std::get<0>(lvs));
                    auto &fieldE = dof::GetDofs<dof::Name::Cell>(std::get<1>(lvs));
                    auto &rhsH = dof::GetDofs<dof::Name::Cell>(std::get<2>(lvs));
                    auto &rhsE = dof::GetDofs<dof::Name::Cell>(std::get<3>(lvs));
                    auto &resH = dof::GetDofs<dof::Name::Cell>(std::get<4>(lvs));
                    auto &resE = dof::GetDofs<dof::Name::Cell>(std::get<5>(lvs));

                    HPM::ForEach(DG::numVolNodes, [&](const std::size_t n) {
                        resH[n] = RKstage[0] * resH[n] + timeStep * rhsH[n]; //!< residual fields
                        resE[n] = RKstage[0] * resE[n] + timeStep * rhsE[n];
                        fieldH[n] += RKstage[1] * resH[n]; //!< updated fields
                        fieldE[n] += RKstage[1] * resE[n];
                        rhsH[n] = 0.0; //TODO
                        rhsE[n] = 0.0;
                    });
                }, HPM::internal::OpenMP_ForEachEntity<3> {}
                );

        body.Execute(
            iterator::Range<size_t> { static_cast<std::size_t>(((finalTime - startTime) / timeStep) * 5) }, surfaceKernelLoop, volumeKernelLoop, rungeKuttaLoop);

        t2 = std::chrono::high_resolution_clock::now();
        duration = t2 - t1;
        aggregate_time3 += duration.count();
    }
    std::cout << "Aggregate execution time for Surface kernel       = " << aggregate_time1 * 1000 << " ms" << std::endl;
    std::cout << "Aggregate execution time for Volume kernel        = " << aggregate_time2 * 1000 << " ms" << std::endl;
    std::cout << "Aggregate execution time for RK kernel            = " << aggregate_time3 * 1000 << " ms" << std::endl;
    std::cout << "Aggregate all kernel execution time               = " << (aggregate_time1 + aggregate_time2 + aggregate_time3) * 1000 << " ms" << std::endl;
    std::cout << "Individual Execution time of Surface kernel       = " << (aggregate_time1 * 1000) / (finalTime / timeStep * 5) << " ms" << std::endl;
    std::cout << "Individual Execution time of Volume kernel        = " << (aggregate_time2 * 1000) / (finalTime / timeStep * 5) << " ms" << std::endl;
    std::cout << "Individual Execution time of RK kernel            = " << (aggregate_time3 * 1000) / (finalTime / timeStep * 5) << " ms" << std::endl;
    std::cout << "Individual all kernel execution time              = " << ((aggregate_time1 + aggregate_time2 + aggregate_time3) * 1000) / (finalTime / timeStep * 5) << " ms" << std::endl;

    // /** \brief find maximum & minimum values for Ey*/
    HPM::dataType::MaxType<double> maxErrorEy; // = 0;
    HPM::dataType::MinType<double> minEy;      // = std::numeric_limits<RealType>::max();
    HPM::dataType::MaxType<double> maxEy;      // = std::numeric_limits<RealType>::lowest();
 
    body.Execute(
        HPM::ForEachEntity(
        AllCells,
        std::tuple(Read(Cell(fieldE))),
        [&](const auto &element, const auto&, auto &lvs) {
            const auto &fieldE = dof::GetDofs<dof::Name::Cell>(std::get<0>(lvs));

            HPM::ForEach(DG::numVolNodes, [&](const std::size_t n) {
                const auto &nodeCoords = DG::LocalToGlobal(DG::referenceCoords[n], element.GetTopology().GetNodes());            //!< reference to global nodal coordinates
                const RealType exactEy = sin(M_PI * nodeCoords.x) * sin(M_PI * nodeCoords.z) * cos(sqrt(2.) * M_PI * finalTime); //!< exact analytical electrical field value in y direction
                maxErrorEy.Update(std::abs(exactEy - fieldE[n].y));                                                              //!< maximum error in electrical field value in y direction
                minEy.Update(fieldE[n].y);                                                                                       //!< minimum electric field value in y direction
                maxEy.Update(fieldE[n].y);                                                                                       //!< maximum electric field value in y direction
            });
        })
    );

    std::cout << "\nt=" << finalTime
              << " Ey in [ " << minEy
              << ", " << maxEy
              << " ] with max nodal error " << maxErrorEy
              << std::endl;

    //////////////////////////////////////////////////////////////////////////////////////////////////////////
    //                         Shutdown of the runtime system                                               //
    //////////////////////////////////////////////////////////////////////////////////////////////////////////

    return EXIT_SUCCESS;
}
