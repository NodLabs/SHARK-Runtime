# Copyright 2020 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

if(NOT IREE_HAL_DRIVER_VULKAN OR NOT IREE_BUILD_SAMPLES)
  return()
endif()

message("In vulkan samples")
# This target statically links against Vulkan.
# One way to achieve this is by installing the Vulkan SDK from
# https://vulkan.lunarg.com/.
# If Vulkan is not found, omit the target.
include(FindVulkan)
if(NOT Vulkan_FOUND)
  message("Could not find Vulkan, disabling Vulkan GUI programs")
  return()
endif()
message("Found Vulkan, enabling Vulkan GUI programs")

# vcpkg install imgui[vulkan-binding,sdl2-binding]
# Dear ImGui
set(IMGUI_DIR ../../../../../third_party/imgui)
message("Looking for Imgui in ${CMAKE_CURRENT_SOURCE_DIR}/${IMGUI_DIR}")
include_directories(${IMGUI_DIR} ${IMGUI_DIR}/backends ..)

# vcpkg install sdl2[vulkan]
find_package(SDL2 CONFIG QUIET)
if(NOT SDL2_FOUND)
  message("Could not find SDL2, disabling Vulkan GUI programs")
  return()
endif()
message("Found SDL2, enabling Vulkan GUI programs")

iree_cc_library(
  NAME
    vulkan_gui_util
  HDRS
    "vulkan_gui_util.h"
  SRCS
    "vulkan_gui_util.cc"
    "${IMGUI_DIR}/backends/imgui_impl_sdl.cpp"
    "${IMGUI_DIR}/backends/imgui_impl_vulkan.cpp"
    "${IMGUI_DIR}/imgui.cpp"
    "${IMGUI_DIR}/imgui_draw.cpp"
    "${IMGUI_DIR}/imgui_demo.cpp"
    "${IMGUI_DIR}/imgui_tables.cpp"
    "${IMGUI_DIR}/imgui_widgets.cpp"
  DEPS
    iree::base
    SDL2::SDL2
    Vulkan::Vulkan
)

iree_cc_binary(
  NAME
    iree-run-module-vulkan-gui
  SRCS
    "iree-run-module-vulkan-gui-main.cc"
  DEPS
    ::vulkan_gui_util
    iree::base::cc
    iree::base::internal::file_io
    iree::base::internal::flags
    iree::base::internal::main
    iree::base::tracing
    iree::hal::drivers::vulkan::registration
    iree::modules::hal
    iree
    iree::vm
    iree::vm::bytecode_module
  LINKOPTS
    "${IREE_TARGET_GUI_LINKOPTS}"
)
