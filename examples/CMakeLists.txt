add_executable( midg2
    ./MIDG2_DSL/mainUnfusedKernelsDRTS.cpp )		

target_link_libraries (midg2 
    LINK_PRIVATE 
    HighPerMeshesDRTS::HighPerMeshesDRTS
)

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

add_subdirectory(Poisson_DSL)
add_subdirectory(Monodomain_DSL)
