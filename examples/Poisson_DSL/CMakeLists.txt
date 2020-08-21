# Add executable called "poisson_dsl3D" that is built from the source 
# file "poisson3D.cpp". The extensions are automatically found.
add_executable ( poisson_dsl3D ./src/poisson3D.cpp )	

# Link the executable
target_link_libraries (poisson_dsl3D LINK_PRIVATE HighPerMeshesDRTS::HighPerMeshesDRTS)

# config file settings
configure_file(./src/config.cfg config.cfg COPYONLY)

# config file settings: meshfiles
configure_file(./meshfiles/unitcube_amiraFile.am ./meshfiles/unitcube_amiraFile.am COPYONLY)

#---------------------------------------------------------------------------------------------------------------#

# Add executable called "poisson_3DMatFree" that is built from the source 
# file "poisson3DMatFree.cpp". The extensions are automatically found.
add_executable ( poisson_3DMatFree ./src/poisson3DMatFree.cpp )

# Link the executable
target_link_libraries (poisson_3DMatFree LINK_PRIVATE HighPerMeshesDRTS::HighPerMeshesDRTS)

# config file settings
configure_file(./src/configMatFree.cfg configMatFree.cfg COPYONLY)

# config file settings: meshfiles
configure_file(./meshfiles/unitcube_amiraFile.am ./meshfiles/unitcube_amiraFile.am COPYONLY)

#---------------------------------------------------------------------------------------------------------------#

# Add executable called "poisson_3DMatFree" that is built from the source 
# file "poisson2DMatFree.cpp". The extensions are automatically found.
add_executable ( poisson_2DMatFree ./src/poisson2DMatFree.cpp )

# Link the executable
target_link_libraries (poisson_2DMatFree LINK_PRIVATE HighPerMeshesDRTS::HighPerMeshesDRTS)

# config file settings
configure_file(./src/config2DMatFree.cfg config2DMatFree.cfg COPYONLY)

# config file settings: meshfiles
configure_file(./meshfiles/unstructured2DMeshfile.am ./meshfiles/unstructured2DMeshfile.am COPYONLY)

