# FindGPI2.cmake
# 
# Finds the GPI2 library
#
# This will define the following variables
#
#   GPI2_FOUND
#   GPI2_INCLUDE_DIRS
#   GPI2_LIBARIES
#
# and the following imported targets
#
#   GPI2::GPI2
#

find_package(PkgConfig QUIET)
PKG_CHECK_MODULES(GPI2 GPI2>=1.3.0 QUIET IMPORTED_TARGET GPI2)

#Try pkg config first
if(TARGET PkgConfig::GPI2 AND NOT TARGET GPI2::GPI2) 
  
  #Standardise output to GPI2::GPI2
  add_library (GPI2 INTERFACE)
  add_library (GPI2::GPI2 ALIAS GPI2)
  target_link_libraries (GPI2 INTERFACE PkgConfig::GPI2)

else() 

  # Gets the environment variable GPI2_DIR if possible
  if( DEFINED ENV{GPI2_DIR} )
    set( GPI2_DIR "$ENV{GPI2_DIR}" )
  endif()

  # Tries to find the appropriate include path
  find_path(
    GPI2_INCLUDE_DIR
      GPI2.h
    PATH_SUFFIXES
      include
      src
    HINTS
      ${GPI2_DIR}
  )

  # Tries to find the appropriate library
  find_library( GPI2_LIBRARY
    NAMES 
      GPI2
    PATH_SUFFIXES
      lib
      lib32
      lib64
    HINTS 
      ${GPI2_DIR}
  )

  # Handles standard argument of the find call such as REUIRED
  include( FindPackageHandleStandardArgs )
  find_package_handle_standard_args( 
      GPI2
      DEFAULT_MSG
      GPI2_INCLUDE_DIR
      GPI2_LIBRARY
  )

  # This is for the cmake-gui to not clutter options
  mark_as_advanced(
    GPI2_LIBRARY
    GPI2_INCLUDE_DIR
    GPI2_DIR
  )

  if( GPI2_FOUND )

    # It is convention to pluralize the include and library variables
    set( GPI2_INCLUDE_DIRS ${GPI2_INCLUDE_DIR} )
    set( GPI2_LIBRARIES ${GPI2_LIBRARY} )

    # Add GPI2 as an imported target
    if(NOT TARGET GPI2::GPI2)
      add_library(GPI2::GPI2 UNKNOWN IMPORTED)
      set_target_properties(GPI2::GPI2 PROPERTIES IMPORTED_LOCATION "${GPI2_LIBRARIES}")
      set_target_properties(GPI2::GPI2 PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${GPI2_INCLUDE_DIRS}")
    endif()
    
  endif()

endif()