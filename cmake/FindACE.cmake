# FindACE.cmake
# 
# Finds the ACE library
#
# This will define the following variables
#
#   ACE_FOUND
#   ACE_INCLUDE_DIRS
#   ACE_LIBARIES
#
# and the following imported targets
#
#   ACE::ACE
#

# Gets the environment variable ACE_DIR if possible
if( DEFINED ENV{ACE_DIR} )
  set( ACE_DIR "$ENV{ACE_DIR}" )
endif()

# Tries to find the appropriate include path
find_path(
  ACE_INCLUDE_DIR
    Timer.hpp
  PATH_SUFFIXES
    include/ACE
  HINTS
    ${ACE_DIR}
)

# Tries to find the appropriate library
find_library(ACE_LIBRARY
  NAMES 
    ACE
  HINTS 
    ${ACE_DIR}
)

# Handles standard argument of the find call such as REUIRED
include( FindPackageHandleStandardArgs )
find_package_handle_standard_args( 
    ACE 
    DEFAULT_MSG
    ACE_INCLUDE_DIR
    ACE_LIBRARY
)

# This is for the cmake-gui to not clutter options
mark_as_advanced(
  ACE_LIBRARY
  ACE_INCLUDE_DIR
  ACE_DIR
)

if( ACE_FOUND )

  # It is convention to pluralize the include and library variables
  set( ACE_INCLUDE_DIRS ${ACE_INCLUDE_DIR} )
  set( ACE_LIBRARIES ${ACE_LIBRARY} )

  find_package(OpenCL REQUIRED)

  # Add ACE as an imported target
  if(NOT TARGET ACE::ACE)
    add_library(ACE::ACE UNKNOWN IMPORTED)
    set_target_properties(ACE::ACE PROPERTIES IMPORTED_LOCATION "${ACE_LIBRARIES}")
    set_target_properties(ACE::ACE PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${ACE_INCLUDE_DIRS}")
    set_target_properties(ACE::ACE PROPERTIES INTERFACE_LINK_LIBRARIES "numa;pthread;OpenCL::OpenCL")
  endif()
  
endif()