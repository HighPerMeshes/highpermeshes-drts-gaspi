find_package(GTest REQUIRED)

add_executable( tests
    dsl/data_access/GatherDofs.cpp
    drts/Task.cpp
    drts/comm/BufferTest.cpp 
    drts/comm/CLBufferTest.cpp
    dsl/dispatchers/DistributedDispatcher.cpp
)
              
target_include_directories (tests PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR} )

target_link_libraries( tests LINK_PRIVATE  
    GTest::GTest 
    GTest::Main 
    HighPerMeshesDRTS::HighPerMeshesDRTS )

#configure_file(./run_distributed_tests.sh run_distributed_tests.sh COPYONLY)
