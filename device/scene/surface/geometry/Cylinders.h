/*
 * Copyright (c) 2019-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#pragma once

#include "array/Array.h"
#include "Geometry.h"
#include "utility/HostDeviceArray.h"
// anari
#include "anari/detail/Optional.h"

namespace visrtx {

struct Cylinders : public Geometry
{
  Cylinders() = default;
  ~Cylinders() override;

  void commit() override;

  void populateBuildInput(OptixBuildInput &) const override;

  int optixGeometryType() const override;

 private:
  GeometryGPUData gpuData() const override;
  void cleanup();

  anari::IntrusivePtr<Array1D> m_index;
  anari::IntrusivePtr<Array1D> m_radius;

  anari::IntrusivePtr<Array1D> m_vertex;
  anari::IntrusivePtr<Array1D> m_vertexColor;
  anari::IntrusivePtr<Array1D> m_vertexAttribute0;
  anari::IntrusivePtr<Array1D> m_vertexAttribute1;
  anari::IntrusivePtr<Array1D> m_vertexAttribute2;
  anari::IntrusivePtr<Array1D> m_vertexAttribute3;

  HostDeviceArray<uint32_t> m_generatedIndices;

  HostDeviceArray<box3> m_aabbs;
  CUdeviceptr m_aabbsBufferPtr{};

  anari::Optional<float> m_globalRadius;

  bool m_caps{false};
};

} // namespace visrtx
