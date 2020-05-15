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

using namespace ::HPM;
using namespace ::HPM::dataType;
using namespace ::HPM::internal;

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
    using Mesh = HPM::mesh::PartitionedMesh<CoordinateType, HPM::entity::Simplex>;
    //const Mesh mesh = Mesh::template CreateFromFile<HPM::auxiliary::GambitMeshFileReader>(meshFile, {1, 1});
    //const Mesh mesh = Mesh::template CreateFromFile<HPM::auxiliary::GambitMeshFileReader, HPM::mesh::MetisPartitioner>(meshFile, {hpm.GetL1PartitionNumber(), hpm.GetL2PartitionNumber()});
    HPM::mesh::MetisPartitioner partitioner;
    HPM::auxiliary::GambitMeshFileReader<CoordinateType, std::array<std::size_t, 4>> reader;
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
            [&] (const auto& cell, auto&&, auto& lvs)
            {
                const auto& nodes = cell.GetTopology().GetNodes();

                HPM::ForEach(DG::numVolNodes, [&](const auto n) 
                {  
                    const auto& nodeCoords = DG::LocalToGlobal(DG::referenceCoords[n], nodes);
                    auto& fieldE = dof::GetDofs<dof::Name::Cell>(std::get<0>(lvs));

                    fieldE[n].y = std::sin(M_PI * nodeCoords.x) * std::sin(M_PI * nodeCoords.z); 	//!< initial conditions for y component of electric field
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
            [&](const auto& cell, auto&&, auto&&) 
            {
                const RealType face_normal_scaling_factor = 2.0 / cell.GetGeometry().GetAbsJacobianDeterminant();

                HPM::ForEachSubEntity(cell, [&](const auto& face) 
                {
                    MinTimeStep.Update(1.0 / (face.GetGeometry().GetNormal() * face_normal_scaling_factor).Norm());
                });
            }
        ));

    RealType timeStep = finalTime / floor(finalTime * (order + 1) * (order + 1) / (.5 * MinTimeStep.Get()));
    std::cout << "time step: " << timeStep << std::endl;

    {
        auto t1 = std::chrono::high_resolution_clock::now();

        /** \brief Maxwell's surface kernel */
        Vec<Matrix<RealType, DG::numVolNodes, DG::NumSurfaceNodes>, Mesh::NumFacesPerCell> lift;
        
        for (std::size_t face_index = 0; face_index < Mesh::NumFacesPerCell; ++face_index)
        {
            for (std::size_t m = 0; m < DG::NumSurfaceNodes; ++m)
            {
                for (std::size_t n = 0; n < DG::numVolNodes; ++n)
                {
                    lift[face_index][n][m] = DG::LIFT[face_index][m][n];
                }
            }
        }

        auto surfaceKernelLoop = HPM::ForEachIncidence<2>(
            AllCells,
            std::tuple(
                Read(ContainingMeshElement(fieldH)),
                Read(ContainingMeshElement(fieldE)),
                Read(NeighboringMeshElementOrSelf(fieldH)),
                Read(NeighboringMeshElementOrSelf(fieldE)),
                ReadWrite(ContainingMeshElement(rhsH)),
                ReadWrite(ContainingMeshElement(rhsE))),
            [&](const auto &element, const auto &face, const auto&, auto &lvs) 
            {
                const std::size_t face_index = face.GetTopology().GetLocalIndex();
                const RealType face_normal_scaling_factor = 2.0 / element.GetGeometry().GetAbsJacobianDeterminant();

                const Vec3D &face_normal = face.GetGeometry().GetNormal() * face_normal_scaling_factor; //!< get all normal coordinates for each face of an element
                const RealType Edg = face_normal.Norm() * 0.5;                                          //!< get edge length for each face
                const Vec3D &face_unit_normal = face.GetGeometry().GetUnitNormal();
                const auto &localMap{DgNodeMap.Get(element, face)};

                const auto &fieldH = dof::GetDofs<dof::Name::Cell>(std::get<0>(lvs));
                const auto &fieldE = dof::GetDofs<dof::Name::Cell>(std::get<1>(lvs));
                auto &NeighboringFieldH = dof::GetDofs<dof::Name::Cell>(std::get<2>(lvs));
                auto &NeighboringFieldE = dof::GetDofs<dof::Name::Cell>(std::get<3>(lvs));
                auto &rhsH = dof::GetDofs<dof::Name::Cell>(std::get<4>(lvs));
                auto &rhsE = dof::GetDofs<dof::Name::Cell>(std::get<5>(lvs));
                Matrix<RealType, 3, DG::NumSurfaceNodes> dE, dH, flux_E, flux_H;

                for (std::size_t m = 0; m < DG::NumSurfaceNodes; ++m)
                {
                    const auto& tmp_1 = Edg * HPM::DG::Delta(fieldH, NeighboringFieldH, m, localMap); //!< fields differences
                    const auto& tmp_2 = Edg * HPM::DG::DirectionalDelta(fieldE, NeighboringFieldE, face, m, localMap);

                    dH[0][m] = tmp_1[0];
                    dH[1][m] = tmp_1[1];
                    dH[2][m] = tmp_1[2];
                    dE[0][m] = tmp_2[0];
                    dE[1][m] = tmp_2[1];
                    dE[2][m] = tmp_2[2];   
                }

                #pragma omp simd
                for (std::size_t m = 0; m < DG::NumSurfaceNodes; ++m)
                {
                    const auto sp_1 = dH[0][m] * face_unit_normal[0] + dH[1][m] * face_unit_normal[1] + dH[2][m] * face_unit_normal[2];
                    flux_H[0][m] = (dH[0][m] - sp_1 * face_unit_normal[0] - (face_unit_normal[1] * dE[2][m] - face_unit_normal[2] * dE[1][m]));
                    flux_H[1][m] = (dH[1][m] - sp_1 * face_unit_normal[1] - (face_unit_normal[2] * dE[0][m] - face_unit_normal[0] * dE[2][m]));
                    flux_H[2][m] = (dH[2][m] - sp_1 * face_unit_normal[2] - (face_unit_normal[0] * dE[1][m] - face_unit_normal[1] * dE[0][m]));
                    
                    const auto sp_2 = dE[0][m] * face_unit_normal[0] + dE[1][m] * face_unit_normal[1] + dE[2][m] * face_unit_normal[2];
                    flux_E[0][m] = (dE[0][m] - sp_2 * face_unit_normal[0] + (face_unit_normal[1] * dH[2][m] - face_unit_normal[2] * dH[1][m]));
                    flux_E[1][m] = (dE[1][m] - sp_2 * face_unit_normal[1] + (face_unit_normal[2] * dH[0][m] - face_unit_normal[0] * dH[2][m]));
                    flux_E[2][m] = (dE[2][m] - sp_2 * face_unit_normal[2] + (face_unit_normal[0] * dH[1][m] - face_unit_normal[1] * dH[0][m]));
                }
                
                #pragma omp simd
                for (std::size_t n = 0; n < DG::numVolNodes; ++n)
                {
                    RealType rhsH_0 = rhsH[n][0], rhsH_1 = rhsH[n][1], rhsH_2 = rhsH[n][2];
                    RealType rhsE_0 = rhsE[n][0], rhsE_1 = rhsE[n][1], rhsE_2 = rhsE[n][2];
                    
                    for (std::size_t m = 0; m < DG::NumSurfaceNodes; ++m)
                    {
                        
                        rhsH_0 += lift[face_index][n][m] * flux_H[0][m];
                        rhsH_1 += lift[face_index][n][m] * flux_H[1][m];
                        rhsH_2 += lift[face_index][n][m] * flux_H[2][m];
                        rhsE_0 += lift[face_index][n][m] * flux_E[0][m];
                        rhsE_1 += lift[face_index][n][m] * flux_E[1][m];
                        rhsE_2 += lift[face_index][n][m] * flux_E[2][m];
                    }

                    rhsH[n] = Vec3D(rhsH_0, rhsH_1, rhsH_2);
                    rhsE[n] = Vec3D(rhsE_0, rhsE_1, rhsE_2);
                }
            });

        auto t2 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = t2 - t1;
        aggregate_time1 += duration.count();
        t1 = std::chrono::high_resolution_clock::now();

        /** \brief Maxwell's volume kernel */
        RealType dg_derivative[3][DG::numVolNodes][DG::numVolNodes];

        for (std::size_t i = 0; i < 3; ++i)
        {
            for (std::size_t m = 0; m < DG::numVolNodes; ++m)
            {
                for (std::size_t n = 0; n < DG::numVolNodes; ++n)
                {
                    dg_derivative[i][m][n] = DG::derivative[n][m][i];
                }
            }
        }

        auto volumeKernelLoop = HPM::ForEachEntity(
            AllCells,
            std::tuple(
                Read(Cell(fieldH)),
                Read(Cell(fieldE)),
                ReadWrite(Cell(rhsH)),
                ReadWrite(Cell(rhsE))),
            [&](const auto& element, const auto&, auto& lvs) 
            {
                const Mat3D &D = element.GetGeometry().GetInverseJacobian() * 2.0;
                const auto &fieldH = dof::GetDofs<dof::Name::Cell>(std::get<0>(lvs));
                const auto &fieldE = dof::GetDofs<dof::Name::Cell>(std::get<1>(lvs));
                auto &rhsH = dof::GetDofs<dof::Name::Cell>(std::get<2>(lvs));
                auto &rhsE = dof::GetDofs<dof::Name::Cell>(std::get<3>(lvs));
                Matrix<RealType, 3, DG::numVolNodes> derivative_E, derivative_H;

                for (std::size_t i = 0; i < 3; ++i)
                {
                    for (std::size_t j = 0; j < 3; ++j)
                    {
                        #pragma omp simd
                        for (std::size_t n = 0; n < DG::numVolNodes; ++n)
                        {
                            derivative_E[j][n] = fieldH[0][j] * dg_derivative[i][0][n];
                            derivative_H[j][n] = fieldE[0][j] * dg_derivative[i][0][n];
                        }
                    }

                    for (std::size_t m = 1; m < DG::numVolNodes; ++m)
                    {
                        for (std::size_t j = 0; j < 3; ++j)
                        {
                            #pragma omp simd
                            for (std::size_t n = 0; n < DG::numVolNodes; ++n)
                            {
                                derivative_E[j][n] += fieldH[m][j] * dg_derivative[i][m][n];
                                derivative_H[j][n] += fieldE[m][j] * dg_derivative[i][m][n];
                            }
                        }
                    }
                
                    #pragma omp simd
                    for (std::size_t n = 0; n < DG::numVolNodes; ++n)
                    {
                        rhsE[n] += Vec3D(
                            D[i][1] * derivative_E[2][n] - D[i][2] * derivative_E[1][n],
                            D[i][2] * derivative_E[0][n] - D[i][0] * derivative_E[2][n],
                            D[i][0] * derivative_E[1][n] - D[i][1] * derivative_E[0][n]);

                        rhsH[n] -= Vec3D(
                            D[i][1] * derivative_H[2][n] - D[i][2] * derivative_H[1][n],
                            D[i][2] * derivative_H[0][n] - D[i][0] * derivative_H[2][n],
                            D[i][0] * derivative_H[1][n] - D[i][1] * derivative_H[0][n]);
                    }
                }
            });

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
                });

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
