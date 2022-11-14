include(FindPackageHandleStandardArgs)

if (oneCCL_DIR)
  list(PREPEND CMAKE_PREFIX_PATH "${oneCCL_DIR}")
endif()

if (EXISTS "${CCL_CONFIGURATION}")
  set(ONECCL_SUBDIR_ "${CCL_CONFIGURATION}")
else()
  set(ONECCL_SUBDIR_ "cpu_gpu_dpcpp")
endif()

if (ONECCL_SUBDIR_ EQUAL "cpu")
    include(CheckCXXCompilerFlag)
    check_cxx_compiler_flag("-fsycl" FSYCL_OPTION_)
    if (FSYCL_OPTION_)
        message(STATUS "STATUS: -fsycl not supported for CCL_CONFIGURATION=cpu")
    endif()
endif()

find_path(oneCCL_INCLUDE_DIRS oneapi/ccl.hpp PATH_SUFFIXES "include/${ONECCL_SUBDIR_}")
find_library(oneCCL_LIBRARIES ccl PATH_SUFFIXES "lib/${ONECCL_SUBDIR_}")
if(oneCCL_INCLUDE_DIRS AND oneCCL_LIBRARIES)
    add_library(oneCCL UNKNOWN IMPORTED)
    set_target_properties(
        oneCCL
        PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES "${oneCCL_INCLUDE_DIRS}"
            IMPORTED_LINK_INTERFACE_LANGUAGES "C++"
            IMPORTED_LOCATION "${oneCCL_LIBRARIES}"
    )

    find_package(MPI QUIET COMPONENTS CXX)
    if (NOT MPI_FOUND)
        message(STATUS "oneCCL: MPI is not found")
    else()
        set_target_properties(oneCCL PROPERTIES INTERFACE_LINK_LIBRARIES MPI::MPI_CXX)
        message(STATUS "oneCCL: MPI found")
    endif()
endif()
find_package_handle_standard_args(
    oneCCL
    REQUIRED_VARS oneCCL_INCLUDE_DIRS oneCCL_LIBRARIES
)
