
configure_file(./MIDG2_DSL/config.cfg config.cfg COPYONLY)
configure_file(./MIDG2_DSL/F072.neu F072.neu COPYONLY)

add_executable( scalabilitySimplePattern ScalabilitySimplePattern.cpp )		

target_link_libraries (scalabilitySimplePattern 
    LINK_PRIVATE 
    HighPerMeshesDRTS::HighPerMeshesDRTS
)

add_executable( scalabilityNeighbors ScalabilityNeighbors.cpp )		

target_link_libraries (scalabilityNeighbors
    LINK_PRIVATE 
    HighPerMeshesDRTS::HighPerMeshesDRTS
)

add_executable( experiments Experiments.cpp )

find_package(OpenMP REQUIRED)

add_executable( midg2
    ./MIDG2_DSL/mainUnfusedKernelsDRTS.cpp )		

target_link_libraries (midg2 
    LINK_PRIVATE 
    HighPerMeshesDRTS::HighPerMeshesDRTS
    OpenMP::OpenMP_CXX
)

target_link_libraries(
    experiments
    LINK_PRIVATE
    HighPerMeshesDRTS::HighPerMeshesDRTS
    OpenMP::OpenMP_CXX
)

target_include_directories(
    experiments
    PRIVATE
    ../tests/util/
)

add_subdirectory(Poisson_DSL)
