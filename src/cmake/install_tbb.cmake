# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

include(cmake/ie_parallel.cmake)

# pre-find TBB: need to provide TBB_IMPORTED_TARGETS used for installation
ov_find_package_tbb()

if(TBB_FOUND AND TBB_VERSION VERSION_GREATER_EQUAL 2021)
    message(STATUS "Static tbbbind_2_5 package usage is disabled, since oneTBB (ver. ${TBB_VERSION}) is used")
    set(ENABLE_TBBBIND_2_5 OFF)
elseif(ENABLE_TBBBIND_2_5)
    # download and find a prebuilt version of TBBBind_2_5
    ov_download_tbbbind_2_5()
    find_package(TBBBIND_2_5 QUIET)

    if(TBBBIND_2_5_FOUND)
        message(STATUS "Static tbbbind_2_5 package is found")
        set_target_properties(${TBBBIND_2_5_IMPORTED_TARGETS} PROPERTIES
            INTERFACE_COMPILE_DEFINITIONS TBBBIND_2_5_AVAILABLE)
        if(NOT BUILD_SHARED_LIBS)
            set(install_tbbbind ON)
        endif()
    endif()
endif()

# install TBB

# define variables for OpenVINOConfig.cmake
if(THREADING MATCHES "^(TBB|TBB_AUTO)$")
    set(IE_TBB_DIR "${TBB_DIR}")
    list(APPEND PATH_VARS "IE_TBB_DIR")
endif()

if(install_tbbbind)
    set(IE_TBBBIND_DIR "${TBBBIND_2_5_DIR}")
    list(APPEND PATH_VARS "IE_TBBBIND_DIR")
endif()

# install only downloaded | custom TBB, system one is not installed
# - downloaded TBB should be a part of all packages
# - custom TBB provided by users, needs to be a part of wheel packages
# - TODO: system TBB also needs to be a part of wheel packages
if(THREADING MATCHES "^(TBB|TBB_AUTO)$" AND
       ( (DEFINED TBB AND TBB MATCHES ${TEMP}) OR
         (DEFINED TBBROOT OR DEFINED TBB_DIR OR DEFINED ENV{TBBROOT} OR
          DEFINED ENV{TBB_DIR}) OR ENABLE_SYSTEM_TBB ) )
    ie_cpack_add_component(tbb REQUIRED)
    list(APPEND core_components tbb)

    if(TBB MATCHES ${TEMP})
        set(tbb_downloaded ON)
    elseif(DEFINED ENV{TBBROOT} OR DEFINED ENV{TBB_DIR} OR
           DEFINED TBBROOT OR DEFINED TBB_DIR)
        set(tbb_custom ON)
    endif()

    if(CPACK_GENERATOR STREQUAL "DEB" AND NOT ENABLE_SYSTEM_TBB)
        message(FATAL_ERROR "Debian packages can be built only with system TBB. Use -DENABLE_SYSTEM_TBB=ON")
    endif()

    if(ENABLE_SYSTEM_TBB)
        # for system libraries we still need to install TBB libraries
        # so, need to take locations of actual libraries and install them
        foreach(tbb_lib IN LISTS TBB_IMPORTED_TARGETS)
            get_target_property(tbb_loc ${tbb_lib} IMPORTED_LOCATION_RELEASE)
            # depending on the TBB, tbb_loc can be in form:
            # - libtbb.so.x.y
            # - libtbb.so.x
            # - libtbb.so
            # We need to install such files
            get_filename_component(name_we "${tbb_loc}" NAME_WE)
            get_filename_component(dir "${tbb_loc}" DIRECTORY)
            # grab all tbb files matching pattern
            file(GLOB tbb_files "${dir}/${name_we}.*")
            foreach(tbb_file IN LISTS tbb_files)
                if(tbb_file MATCHES "^.*\.${CMAKE_SHARED_LIBRARY_SUFFIX}(\.[0-9]+)*$")
                    # since the setup.py for pip installs tbb component
                    # explicitly, it's OK to put EXCLUDE_FROM_ALL to such component
                    # to ignore from IRC / apt / yum distribution;
                    # but they will be present in .wheel
                    install(FILES "${tbb_file}"
                            DESTINATION runtime/3rdparty/tbb/lib
                            COMPONENT tbb EXCLUDE_FROM_ALL)
                endif()
            endforeach()
        endforeach()
    elseif(tbb_custom)
        # for custom TBB we need to install it to our package
        # to simplify life for our customers
        set(IE_TBBROOT_INSTALL "runtime/3rdparty/tbb")

        # TBBROOT is not defined if ENV{TBBROOT} is not found
        # so, we have to deduce this value outselves
        if(NOT DEFINED TBBROOT AND DEFINED ENV{TBBROOT})
            file(TO_CMAKE_PATH $ENV{TBBROOT} TBBROOT)
        endif()
        if(NOT DEFINED TBBROOT)
            get_target_property(_tbb_include_dir TBB::tbb INTERFACE_INCLUDE_DIRECTORIES)
            get_filename_component(TBBROOT ${_tbb_include_dir} PATH)
        endif()
        if(DEFINED TBBROOT)
            set(TBBROOT "${TBBROOT}" CACHE PATH "TBBROOT path" FORCE)
        else()
            message(FATAL_ERROR "Failed to deduce TBBROOT, please define env var TBBROOT")
        endif()

        file(RELATIVE_PATH IE_TBB_DIR_INSTALL "${TBBROOT}" "${TBB_DIR}")
        set(IE_TBB_DIR_INSTALL "${IE_TBBROOT_INSTALL}/${IE_TBB_DIR_INSTALL}")

        install(DIRECTORY "${TBBROOT}/"
                DESTINATION "${IE_TBBROOT_INSTALL}"
                COMPONENT tbb)
    elseif(tbb_downloaded)
        if(WIN32)
            install(DIRECTORY "${TBB}/bin"
                    DESTINATION runtime/3rdparty/tbb
                    COMPONENT tbb)
        else()
            install(DIRECTORY "${TBB}/lib"
                    DESTINATION runtime/3rdparty/tbb
                    COMPONENT tbb)
        endif()

        install(FILES "${TBB}/LICENSE"
                DESTINATION runtime/3rdparty/tbb
                COMPONENT tbb)

        # install development files

        ie_cpack_add_component(tbb_dev REQUIRED)
        list(APPEND core_dev_components tbb_dev)

        set(IE_TBB_DIR_INSTALL "3rdparty/tbb/cmake")
        install(FILES "${TBB}/cmake/TBBConfig.cmake"
                    "${TBB}/cmake/TBBConfigVersion.cmake"
                DESTINATION runtime/${IE_TBB_DIR_INSTALL}
                COMPONENT tbb_dev)
        install(DIRECTORY "${TBB}/include"
                DESTINATION runtime/3rdparty/tbb
                COMPONENT tbb_dev)

        if(WIN32)
            # .lib files are needed only for Windows
            install(DIRECTORY "${TBB}/lib"
                    DESTINATION runtime/3rdparty/tbb
                    COMPONENT tbb_dev)
        endif()
    else()
        message(WARNING "TBB of unknown origin. TBB files are not installed")
    endif()

    unset(tbb_downloaded)
    unset(tbb_custom)
endif()

# install tbbbind for static OpenVINO case
if(install_tbbbind)
    install(DIRECTORY "${TBBBIND_2_5}/lib"
            DESTINATION runtime/3rdparty/tbb_bind_2_5
            COMPONENT tbb)
    install(FILES "${TBBBIND_2_5}/LICENSE"
            DESTINATION runtime/3rdparty/tbb_bind_2_5
            COMPONENT tbb)

    set(IE_TBBBIND_DIR_INSTALL "3rdparty/tbb_bind_2_5/cmake")
    install(FILES "${TBBBIND_2_5}/cmake/TBBBIND_2_5Config.cmake"
            DESTINATION runtime/${IE_TBBBIND_DIR_INSTALL}
            COMPONENT tbb_dev)
endif()
