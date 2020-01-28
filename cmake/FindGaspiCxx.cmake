# FindGaspiCxx.cmake
# 
# Finds the GaspiCxx library
#
# This will define the following variables
#
#   GaspiCxx_FOUND
#   GaspiCxx_INCLUDE_DIRS
#   GaspiCxx_LIBARIES
#
# and the following imported targets
#
#   GaspiCxx::GaspiCxx
#

# Gets the environment variable GaspiCxx_DIR if possible
if( DEFINED ENV{GaspiCxx_DIR} )
  set( GaspiCxx_DIR "$ENV{GaspiCxx_DIR}" )
endif()

# Tries to find the appropriate include path
find_path(
  GaspiCxx_INCLUDE_DIR
    GaspiCxx.h
  HINTS
    ${GaspiCxx_DIR}
)

# Tries to find the appropriate library
find_library( GaspiCxx_LIBRARY
  NAMES 
    GaspiCxx
  HINTS 
    ${GaspiCxx_DIR}
)

# Handles standard argument of the find call such as REUIRED
include( FindPackageHandleStandardArgs )
find_package_handle_standard_args( 
    GaspiCxx 
    DEFAULT_MSG
    GaspiCxx_INCLUDE_DIR
    GaspiCxx_LIBRARY
)

# This is for the cmake-gui to not clutter options
mark_as_advanced(
  GaspiCxx_LIBRARY
  GaspiCxx_INCLUDE_DIR
  GaspiCxx_DIR
)

if( GaspiCxx_FOUND )

  # It is convention to pluralize the include and library variables
  set( GaspiCxx_INCLUDE_DIRS ${GaspiCxx_INCLUDE_DIR} )
  set( GaspiCxx_LIBRARIES ${GaspiCxx_LIBRARY} )

  find_package(GPI2 1.3.0 REQUIRED)

  # Add GaspiCxx as an imported target
  if(NOT TARGET GaspiCxx::GaspiCxx)
    
    add_library(GaspiCxx::GaspiCxx UNKNOWN IMPORTED)
    set_target_properties(GaspiCxx::GaspiCxx PROPERTIES IMPORTED_LOCATION "${GaspiCxx_LIBRARIES}")
    set_target_properties(GaspiCxx::GaspiCxx PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${GaspiCxx_INCLUDE_DIRS};${GPI2_INCLUDE_DIRS}")
    set_target_properties(GaspiCxx::GaspiCxx PROPERTIES INTERFACE_LINK_LIBRARIES GPI2::GPI2)

  endif()
  
endif()