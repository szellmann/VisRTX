# Copyright (c) 2019-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

if (TARGET MDL_SDK::MDL_SDK)
  return()
endif()

message(STATUS "find_path(MDL_SDK_ROOT NAMES mdl_sdk.h PATHS ${MDL_SDK_PATH} ENV MDL_SDK_PATH PATH_SUFFIXES include/mi/)")
set(CMAKE_FIND_DEBUG_MODE TRUE)
find_path(MDL_SDK_ROOT NAMES "include/mi/mdl_sdk.h" PATHS ${MDL_SDK_PATH} ENV MDL_SDK_PATH)
set(CMAKE_FIND_DEBUG_MODE FALSE)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(MDL_SDK DEFAULT_MSG MDL_SDK_ROOT)


set(MDL_SDK_INCLUDE_DIR ${MDL_SDK_ROOT}/include)
set(MDL_SDK_INCLUDE_DIRS ${MDL_SDK_ROOT}/include)
mark_as_advanced(MDL_SDK_INCLUDE_DIR MDL_SDK_INCLUDE_DIRS)

add_library(MDL_SDK::MDL_SDK INTERFACE IMPORTED)
target_include_directories(MDL_SDK::MDL_SDK INTERFACE ${MDL_SDK_INCLUDE_DIR})
