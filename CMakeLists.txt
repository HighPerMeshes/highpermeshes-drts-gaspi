cmake_minimum_required (VERSION 3.1)
project (HighPerMeshesDRTS)
set(CMAKE_EXPORT_COMPILE_COMMANDS on)

list(INSERT CMAKE_MODULE_PATH 0 ${CMAKE_SOURCE_DIR}/cmake)

set (CMAKE_CXX_STANDARD 17)
set (BUILD_TESTS on)
set (BUILD_EXAMPLES on)

# Create a library called "HighPerMeshesDRTS", also add an alias for the correct namespace
add_library ( HighPerMeshesDRTS INTERFACE )
add_library ( HighPerMeshesDRTS::HighPerMeshesDRTS ALIAS HighPerMeshesDRTS )

# Link the required libraries
target_link_libraries(HighPerMeshesDRTS 
	INTERFACE HighPerMeshes::HighPerMeshes
	INTERFACE METIS::metis
	INTERFACE ACE::ACE
	INTERFACE GaspiCxx::GaspiCxx )

# Define headers for this library. 
target_include_directories(HighPerMeshesDRTS
	INTERFACE 
		$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
		$<INSTALL_INTERFACE:include>
   )

include(GNUInstallDirs)

# Where to install the configuration for the HighPerMeshes package 
set(INSTALL_CONFIGDIR ${CMAKE_INSTALL_LIBDIR}/cmake/HighPerMeshesDRTS)

# How to install targets
install(TARGETS HighPerMeshesDRTS
	EXPORT HighPerMeshesDRTSTargets
	ARCHIVE  DESTINATION ${CMAKE_INSTALL_LIBDIR}
	LIBRARY  DESTINATION ${CMAKE_INSTALL_LIBDIR}
	RUNTIME  DESTINATION ${CMAKE_INSTALL_BINDIR})  # This is for Windows

# How to install headers
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

# Export targets to a script  
install(
	EXPORT HighPerMeshesDRTSTargets
	FILE HighPerMeshesDRTSTargets.cmake
	DESTINATION ${INSTALL_CONFIGDIR}
	NAMESPACE HighPerMeshesDRTS::
)

include(CMakePackageConfigHelpers)
configure_package_config_file(${CMAKE_CURRENT_LIST_DIR}/cmake/HighPerMeshesDRTSConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/HighPerMeshesDRTSConfig.cmake
    INSTALL_DESTINATION ${INSTALL_CONFIGDIR}
)

# Install config and find modules
install(FILES
    ${CMAKE_CURRENT_LIST_DIR}/cmake/FindACE.cmake
	${CMAKE_CURRENT_LIST_DIR}/cmake/FindGaspiCxx.cmake
	${CMAKE_CURRENT_BINARY_DIR}/HighPerMeshesDRTSConfig.cmake
    DESTINATION ${INSTALL_CONFIGDIR}
)

configure_file(
	${CMAKE_CURRENT_LIST_DIR}/cmake/FindGaspiCxx.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/FindGaspiCxx.cmake
    COPYONLY)

configure_file(
	${CMAKE_CURRENT_LIST_DIR}/cmake/FindACE.cmake
	${CMAKE_CURRENT_BINARY_DIR}/FindACE.cmake
	COPYONLY)

export(EXPORT 
	HighPerMeshesDRTSTargets 
	FILE ${CMAKE_CURRENT_BINARY_DIR}/HighPerMeshesDRTSTargets.cmake
	NAMESPACE HighPerMeshesDRTS::
	)

#Register packag
export(PACKAGE HighPerMeshesDRTS)

if(GPI2_USE_MPIRUN)
	add_definitions(-DGPI2_MPI_INTEROP)
endif(GPI2_USE_MPIRUN)

if(BUILD_TESTS OR BUILD_EXAMPLES)
	find_package(HighPerMeshes REQUIRED)
	find_package(ACE REQUIRED)
	find_package(GaspiCxx REQUIRED)
endif()

if (BUILD_TESTS)
  add_subdirectory (tests)
endif (BUILD_TESTS)

if (BUILD_EXAMPLES)
  add_subdirectory (examples)
endif (BUILD_EXAMPLES)
