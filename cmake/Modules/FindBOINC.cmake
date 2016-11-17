# - Find BOINC 
# Find the native BOINC includes and libraries
#
#  BOINC_INCLUDE_DIR        - where to find boinc.h, etc.
#  BOINC_SERVER_FOUND       - true if libraries required for compiling boinc server code are found
#  BOINC_SERVER_LIBRARIES   - all the libraries required for compiling boinc server code
#  BOINC_APP_FOUND          - true if libraries required for compiling boinc apps are found 
#  BOINC_APP_LIBRARIES      - all the libraries required for compiling boinc apps

IF (BOINC_INCLUDE_DIR)
    # Already in cache, be silent
    SET(BOINC_FIND_QUIETLY TRUE)
ENDIF (BOINC_INCLUDE_DIR)

FIND_PATH(BOINC_INCLUDE_DIR api/boinc_api.h
    /boinc/src/boinc
    /home/tdesell/boinc
    /Users/deselt/Software/boinc
    $ENV{BOINC_SOURCE}
)
MESSAGE(STATUS "BOINC include directory: ${BOINC_INCLUDE_DIR}")

FIND_LIBRARY(BOINC_LIBRARY
    NAMES boinc
    PATHS /boinc/src/boinc /home/tdesell/boinc /Users/deselt/Software/boinc/mac_build/build/Deployment/ $ENV{BOINC_SOURCE}
    PATH_SUFFIXES lib
)
#MESSAGE(STATUS "BOINC library: ${BOINC_LIBRARY}")

FIND_LIBRARY(BOINC_CRYPT_LIBRARY
    NAMES boinc_crypt
    PATHS /boinc/src/boinc /home/tdesell/boinc /Users/Deselt/Software/boinc/mac_build/build/Deployment/ $ENV{BOINC_SOURCE}
    PATH_SUFFIXES lib
)
#MESSAGE(STATUS "BOINC crypt library: ${BOINC_CRYPT_LIBRARY}")

FIND_LIBRARY(BOINC_API_LIBRARY
    NAMES boinc_api
    PATHS /boinc/src/boinc /home/tdesell/boinc /Users/Deselt/Software/boinc/mac_build/build/Deployment/ $ENV{BOINC_SOURCE}
    PATH_SUFFIXES api
)
#MESSAGE(STATUS "BOINC api library: ${BOINC_API_LIBRARY}")

FIND_LIBRARY(BOINC_OPENCL_LIBRARY
    NAMES boinc_opencl
    PATHS /boinc/src/boinc /home/tdesell/boinc /Users/Deselt/Software/boinc/mac_build/build/Deployment/ $ENV{BOINC_SOURCE}
    PATH_SUFFIXES lib
)
MESSAGE(STATUS "BOINC opencl library: ${BOINC_OPENCL_LIBRARY}")

FIND_LIBRARY(BOINC_SCHED_LIBRARY
    NAMES sched
    PATHS /boinc/src/boinc /home/tdesell/boinc /Users/Deselt/Software/boinc/mac_build/build/Deployment/ $ENV{BOINC_SOURCE}
    PATH_SUFFIXES sched
)
MESSAGE(STATUS "BOINC sched library: ${BOINC_SCHED_LIBRARY}")

IF (BOINC_INCLUDE_DIR AND BOINC_LIBRARY AND BOINC_API_LIBRARY AND BOINC_OPENCL_LIBRARY)
    add_definitions( -D_BOINC_APP_ )
    SET (BOINC_APP_FOUND TRUE)
    SET (BOINC_APP_LIBRARIES ${BOINC_API_LIBRARY} ${BOINC_LIBRARY} ${BOINC_OPENCL_LIBRARY})

    MESSAGE(STATUS "Found BOINC_APP_LIBRARIES: ${BOINC_APP_LIBRARIES}")
    MESSAGE(STATUS "BOINC include directory: ${BOINC_INCLUDE_DIR}")
ELSE (BOINC_INCLUDE_DIR AND BOINC_LIBRARY AND BOINC_API_LIBRARY AND BOINC_OPENCL_LIBRARY)
    SET (BOINC_APP_FOUND FALSE)
    SET (BOINC_APP_LIBRARIES )
ENDIF (BOINC_INCLUDE_DIR AND BOINC_LIBRARY AND BOINC_API_LIBRARY AND BOINC_OPENCL_LIBRARY)

IF (BOINC_INCLUDE_DIR AND BOINC_LIBRARY AND BOINC_API_LIBRARY AND BOINC_SCHED_LIBRARY AND BOINC_CRYPT_LIBRARY)
    add_definitions( -D_BOINC_SERVER_ )
    SET(BOINC_SERVER_FOUND TRUE)
    SET( BOINC_SERVER_LIBRARIES ${BOINC_SCHED_LIBRARY} ${BOINC_LIBRARY} ${BOINC_API_LIBRARY} ${BOINC_CRYPT_LIBRARY})

    MESSAGE(STATUS "Found BOINC_SERVER_LIBRARIES: ${BOINC_SERVER_LIBRARIES}")
    MESSAGE(STATUS "BOINC include directory: ${BOINC_INCLUDE_DIR}")
ELSE (BOINC_INCLUDE_DIR AND BOINC_LIBRARY AND BOINC_API_LIBRARY AND BOINC_SCHED_LIBRARY AND BOINC_CRYPT_LIBRARY)
    SET(BOINC_FOUND FALSE)
    SET( BOINC_LIBRARIES )
ENDIF (BOINC_INCLUDE_DIR AND BOINC_LIBRARY AND BOINC_API_LIBRARY AND BOINC_SCHED_LIBRARY AND BOINC_CRYPT_LIBRARY)

MARK_AS_ADVANCED(
    BOINC_APP_LIBRARIES
    BOINC_SCHED_LIBRARIES
    BOINC_INCLUDE_DIR
    )
