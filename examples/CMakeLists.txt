
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

find_package(OpenMP REQUIRED)

add_executable( midg2
    ./MIDG2_DSL/mainUnfusedKernelsDRTS.cpp )		

target_link_libraries (midg2 
    LINK_PRIVATE 
    HighPerMeshesDRTS::HighPerMeshesDRTS
    OpenMP::OpenMP_CXX
)

function( Experiment name )
    
    add_executable( ${name} ${name}.cpp )

    target_link_libraries(
        ${name}
        LINK_PRIVATE
        HighPerMeshesDRTS::HighPerMeshesDRTS
        OpenMP::OpenMP_CXX
    )

    target_include_directories(
        ${name}
        PRIVATE
        ../tests/util/
    )    
endfunction()

Experiment(matrix_vec_product)
Experiment(matrix_vec_product_omp)

Experiment(surface_kernel)
Experiment(surface_kernel_omp)

Experiment(volume_kernel)
Experiment(volume_kernel_omp)


add_subdirectory(Poisson_DSL)
add_subdirectory(Monodomain_DSL)
