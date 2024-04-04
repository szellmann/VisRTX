// Copyright (c) 2019-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
// this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation
// and/or other materials provided with the distribution.
//
// 3. Neither the name of the copyright holder nor the names of its
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
// This file was generated by generate_device_frontend.py
// Don't make changes to this directly

#pragma once
namespace visgl{
#define STRING_ENUM_unknown -1
#define STRING_ENUM_ASTC_10x10 0
#define STRING_ENUM_ASTC_10x10_SRGB 1
#define STRING_ENUM_ASTC_10x5 2
#define STRING_ENUM_ASTC_10x5_SRGB 3
#define STRING_ENUM_ASTC_10x6 4
#define STRING_ENUM_ASTC_10x6_SRGB 5
#define STRING_ENUM_ASTC_10x8 6
#define STRING_ENUM_ASTC_10x8_SRGB 7
#define STRING_ENUM_ASTC_12x10 8
#define STRING_ENUM_ASTC_12x10_SRGB 9
#define STRING_ENUM_ASTC_12x12 10
#define STRING_ENUM_ASTC_12x12_SRGB 11
#define STRING_ENUM_ASTC_4x4 12
#define STRING_ENUM_ASTC_4x4_SRGB 13
#define STRING_ENUM_ASTC_5x4 14
#define STRING_ENUM_ASTC_5x4_SRGB 15
#define STRING_ENUM_ASTC_5x5 16
#define STRING_ENUM_ASTC_5x5_SRGB 17
#define STRING_ENUM_ASTC_6x5 18
#define STRING_ENUM_ASTC_6x5_SRGB 19
#define STRING_ENUM_ASTC_6x6 20
#define STRING_ENUM_ASTC_6x6_SRGB 21
#define STRING_ENUM_ASTC_8x5 22
#define STRING_ENUM_ASTC_8x5_SRGB 23
#define STRING_ENUM_ASTC_8x6 24
#define STRING_ENUM_ASTC_8x6_SRGB 25
#define STRING_ENUM_ASTC_8x8 26
#define STRING_ENUM_ASTC_8x8_SRGB 27
#define STRING_ENUM_BC1_RGB 28
#define STRING_ENUM_BC1_RGBA 29
#define STRING_ENUM_BC1_RGBA_SRGB 30
#define STRING_ENUM_BC1_RGB_SRGB 31
#define STRING_ENUM_BC2 32
#define STRING_ENUM_BC2_SRGB 33
#define STRING_ENUM_BC3 34
#define STRING_ENUM_BC3_SRGB 35
#define STRING_ENUM_BC4 36
#define STRING_ENUM_BC4_SNORM 37
#define STRING_ENUM_BC5 38
#define STRING_ENUM_BC5_SNORM 39
#define STRING_ENUM_BC6H_SFLOAT 40
#define STRING_ENUM_BC6H_UFLOAT 41
#define STRING_ENUM_BC7 42
#define STRING_ENUM_BC7_SRGB 43
#define STRING_ENUM_OpenGL 44
#define STRING_ENUM_OpenGL_ES 45
#define STRING_ENUM_attribute0 46
#define STRING_ENUM_attribute1 47
#define STRING_ENUM_attribute2 48
#define STRING_ENUM_attribute3 49
#define STRING_ENUM_blend 50
#define STRING_ENUM_both 51
#define STRING_ENUM_clampToEdge 52
#define STRING_ENUM_color 53
#define STRING_ENUM_device 54
#define STRING_ENUM_exact 55
#define STRING_ENUM_first 56
#define STRING_ENUM_firstFrame 57
#define STRING_ENUM_incremental 58
#define STRING_ENUM_linear 59
#define STRING_ENUM_mask 60
#define STRING_ENUM_mirrorRepeat 61
#define STRING_ENUM_nearest 62
#define STRING_ENUM_none 63
#define STRING_ENUM_objectNormal 64
#define STRING_ENUM_objectPosition 65
#define STRING_ENUM_opaque 66
#define STRING_ENUM_primitiveId 67
#define STRING_ENUM_repeat 68
#define STRING_ENUM_second 69
#define STRING_ENUM_tessellate 70
#define STRING_ENUM_worldNormal 71
#define STRING_ENUM_worldPosition 72
extern const char *param_strings[];
int parameter_string_hash(const char *str);
} //namespace visgl
