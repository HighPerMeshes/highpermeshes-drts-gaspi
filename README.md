# highpermeshes-drts-gaspi
Distributed runtime system (DRTS) for HighPerMeshes DSL based on GASPI

# Requirements

HighPerMeshes is a C++ Cmake Project that requires at least g++-8 / clang-6 and Cmake 3.1.
For a script to clone and install the DRTS see `clone_script.sh`.
The script installs all direct dependencies of the DRTS, i.e., not the dependencies of our dependencies.
Most notably, you have to configure GPI-2 (https://github.com/cc-hpc-itwm/GPI-2) to execute the script. 

## GaspiCxx
https://github.com/cc-hpc-itwm/GaspiCxx.git

## ACE
https://github.com/cc-hpc-itwm/ACE.git

For ACE to properly work with HighPerMeshes, the `device` branch must be checked out:
```
  git checkout device
```
## Google Test 
https://github.com/google/googletest

## Metis
http://glaros.dtc.umn.edu/gkhome/metis/metis/overview
