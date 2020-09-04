# Add executable called "monodomain_dsl2D" that is built from the source 
# file "monodomain2D.cpp". The extensions are automatically found.
add_executable ( monodomain_dsl2D
                  ./src/monodomain2D.cpp )	

# Link the executable
target_link_libraries (monodomain_dsl2D LINK_PRIVATE HighPerMeshesDRTS::HighPerMeshesDRTS)

# config file
configure_file(./src/config.cfg config.cfg COPYONLY)

# config file setting: meshfile
#configure_file(./meshfiles/mesh2D_test5x5.am COPYONLY)
#configure_file(./meshfiles/mesh2D_20x20.am COPYONLY)
configure_file(./meshfiles/mesh2D.am COPYONLY)



# Add executable called "monodomain_dsl3D" that is built from the source 
# file "monodomain3D.cpp". The extensions are automatically found.
add_executable ( monodomain_dsl3D
                  ./src/monodomain3D.cpp )	

# Link the executable
target_link_libraries (monodomain_dsl3D LINK_PRIVATE HighPerMeshesDRTS::HighPerMeshesDRTS)

# config file
configure_file(./src/config3D.cfg config3D.cfg COPYONLY)

# config file setting: meshfile
configure_file(./meshfiles/glennWithFiber.am COPYONLY)
