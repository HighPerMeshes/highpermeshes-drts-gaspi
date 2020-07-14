#!/bin/bash
mkdir hpm-repos
cd hpm-repos

# metis
wget http://glaros.dtc.umn.edu/gkhome/fetch/sw/metis/metis-5.1.0.tar.gz
tar -xvf metis-5.1.0.tar.gz
cd metis-5.1.0
make config prefix=..
make install 
METIS_LIBRARY=$(pwd)/build/lib/libmetis.a
METIS_INCLUDE_DIR=$(pwd)/build/include
METIS_CMAKE_FLAGS="-DMETIS_LIBRARY=$METIS_LIBRARY -DMETIS_INCLUDE_DIR=$METIS_INCLUDE_DIR" 
cd ..

# googletest
git clone https://github.com/google/googletest
cd googletest
mkdir build 
cd build
cmake ..
make
GTEST_LIBRARY=$(pwd)/lib/libgtest.a
GTEST_MAIN_LIBRARY=$(pwd)/lib/libgtest_main.a
cd ..
GTEST_INCLUDE_DIR=$(pwd)/googletest/include
GTEST_CMAKE_FLAGS="-DGTEST_LIBRARY=$GTEST_LIBRARY -DGTEST_MAIN_LIBRARY=$GTEST_MAIN_LIBRARY -DGTEST_INCLUDE_DIR=$GTEST_INCLUDE_DIR"
cd ..

# hpm
git clone https://github.com/HighPerMeshes/highpermeshes-dsl
cd highpermeshes-dsl
mkdir build
cd build
cmake .. $METIS_CMAKE_FLAGS $GTEST_CMAKE_FLAGS
make
HighPerMeshes_DIR=$(pwd)
HighPerMeshes_CMAKE_FLAGS="-DHighPerMeshes_DIR=$HighPerMeshes_DIR"
cd ../..

# GaspiCxx
git clone https://github.com/cc-hpc-itwm/GaspiCxx
cd GaspiCxx
mkdir build
cd build
cmake .. $GTEST_CMAKE_FLAGS
make
GaspiCxx_LIBRARY=$(pwd)/src/libGaspiCxx.a
cd ..
GaspiCxx_INCLUDE_DIR=$(pwd)/include
GaspiCxx_CMAKE_FLAGS="-DGaspiCxx_LIBRARY=$GaspiCxx_LIBRARY -DGaspiCxx_INCLUDE_DIR=$GaspiCxx_INCLUDE_DIR"
cd ..

# ACE
git clone https://github.com/cc-hpc-itwm/ACE
cd ACE
git checkout device #Need device branch for ACE
mkdir build
cd build
cmake .. $GTEST_CMAKE_FLAGS
make 
ACE_LIBRARY=$(pwd)/src/libACE.a
cd ..
ACE_INCLUDE_DIR=$(pwd)/include
ACE_CMAKE_FLAGS="-DACE_LIBRARY=$ACE_LIBRARY -DACE_INCLUDE_DIR=$ACE_INCLUDE_DIR"
cd ..

# highpermeshes-drts
git clone https://github.com/HighPerMeshes/highpermeshes-drts-gaspi
cd highpermeshes-drts-gaspi
mkdir build
cd build
cmake .. $ACE_CMAKE_FLAGS $GaspiCxx_CMAKE_FLAGS $HighPerMeshes_CMAKE_FLAGS $GTEST_CMAKE_FLAGS $METIS_CMAKE_FLAGS
make
cd ../..