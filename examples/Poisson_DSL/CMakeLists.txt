# Add executable called "poisson_dsl3D" that is built from the source 
# file "poisson3D.cpp". The extensions are automatically found.
add_executable ( poisson_dsl3D
                  ./src/poisson3D.cpp )	

# Link the executable
target_link_libraries (poisson_dsl3D LINK_PRIVATE HighPerMeshesDRTS::HighPerMeshesDRTS)

# config file settings
configure_file(./src/config.cfg config.cfg COPYONLY)

# config file settings: meshfiles
#configure_file(./meshfiles/meshTestFile.neu meshTestFile.neu COPYONLY)
configure_file(./meshfiles/unitcube_amiraFile.am ./meshfiles/unitcube_amiraFile.am COPYONLY)

# config file settings for gaspi_run
#configure_file(./gaspiRunFiles/mfile12 ./gaspiRunFiles/mfile12 COPYONLY)    # machine file
#configure_file(./gaspiRunFiles/devtopo1 ./gaspiRunFiles/devtopo1 COPYONLY)
#configure_file(./gaspiRunFiles/devtopo2 ./gaspiRunFiles/devtopo2 COPYONLY)

