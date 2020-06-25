# Add executable called "monodomain_dsl2D" that is built from the source 
# file "monodomain2D.cpp". The extensions are automatically found.
add_executable ( monodomain_dsl2D
                  ./src/monodomain2D.cpp )	

# Link the executable
target_link_libraries (monodomain_dsl2D LINK_PRIVATE HighPerMeshesDRTS::HighPerMeshesDRTS)

# config file
configure_file(./src/config.cfg config.cfg COPYONLY)

# config file setting: meshfile
#configure_file(./meshfiles/mesh2D.am COPYONLY)
configure_file(./meshfiles/mesh2D_test5x5.am COPYONLY)
