/*
 * Copyright (c) 2019-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "UnstructuredField.h"
// cuda
#include <math_constants.h>
// thrust
#include <thrust/for_each.h>
#include <thrust/tuple.h>
#include <thrust/iterator/counting_iterator.h>

namespace visrtx {

// Helper functions ///////////////////////////////////////////////////////////

// UnstructuredField definitions //////////////////////////////////////////////

UnstructuredField::UnstructuredField(DeviceGlobalState *d)
    : SpatialField(d)
{}

UnstructuredField::~UnstructuredField()
{
  cleanup();
}

void UnstructuredField::commit()
{
  cleanup();

  m_params.vertexPosition = getParamObject<Array1D>("vertex.position");
  m_params.vertexData = getParamObject<Array1D>("vertex.data");
  m_params.index = getParamObject<Array1D>("index");
  m_params.cellIndex = getParamObject<Array1D>("cell.index");

  if (!m_params.vertexPosition) {
    reportMessage(ANARI_SEVERITY_WARNING,
        "missing required parameter 'vertex.position' on unstructured spatial field");
    return;
  }

  if (!m_params.vertexData) { // currently vertex data only!
    reportMessage(ANARI_SEVERITY_WARNING,
        "missing required parameter 'vertex.data' on unstructured spatial field");
    return;
  }

  if (!m_params.index) {
    reportMessage(ANARI_SEVERITY_WARNING,
        "missing required parameter 'index' on unstructured spatial field");
    return;
  }

  if (!m_params.cellIndex) {
    reportMessage(ANARI_SEVERITY_WARNING,
        "missing required parameter 'cell.index' on unstructured spatial field");
    return;
  }

  // TODO: check data type/index type validity!
  // cf. stagingBuffer in SR field?

  // Calculate bounds //

  size_t numCells = m_params.cellIndex->size();
  m_aabbs.resize(numCells);

  auto begin = thrust::counting_iterator<uint64_t>(0);
  auto end = begin + numCells;

  auto *vertexPosition = m_params.vertexPosition->beginAs<vec3>(AddressSpace::GPU);
  auto *index = m_params.index->beginAs<uint64_t>(AddressSpace::GPU);
  auto *cellIndex = m_params.cellIndex->beginAs<uint64_t>(AddressSpace::GPU);

  size_t numIndices = m_params.index->endAs<uint64_t>(AddressSpace::GPU)-index;

  auto &state = *deviceState();

  thrust::transform(thrust::cuda::par.on(state.stream),
      begin,
      end,
      m_aabbs.begin(),
      [=] __device__(uint64_t cellID) {
        uint64_t firstIndex = cellIndex[cellID];
        uint64_t lastIndex = cellID < numCells-1 ? cellIndex[cellID+1] : numIndices;

        box3 result(vec3(CUDART_INF_F), vec3(-CUDART_INF_F));
        for (uint64_t i = firstIndex; i < lastIndex; ++i) {
          uint64_t idx = index[i];
          result.extend(vertexPosition[idx]);
        }
        return result;
      });

  m_aabbsBufferPtr = (CUdeviceptr)thrust::raw_pointer_cast(m_aabbs.data());

  std::vector<OptixBuildInput> obi(1);
  obi[0] = buildInput();

  reportMessage(ANARI_SEVERITY_DEBUG, "visrtx::UnstructuredField building cell BVH");
  buildOptixBVH(obi,
      m_bvhCells,
      m_traversableCells,
      m_cellBounds,
      this);

  buildGrid();

  upload();
}

box3 UnstructuredField::bounds() const
{
  return m_cellBounds;
}

float UnstructuredField::stepSize() const
{
  return 0.005f; // TODO!!
}

OptixBuildInput UnstructuredField::buildInput() const
{
  OptixBuildInput obi = {};

  obi.type = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;

  obi.customPrimitiveArray.aabbBuffers = &m_aabbsBufferPtr;
  obi.customPrimitiveArray.numPrimitives = m_aabbs.size();

  static uint32_t buildInputFlags[1] = {OPTIX_GEOMETRY_FLAG_NONE};

  obi.customPrimitiveArray.flags = buildInputFlags;
  obi.customPrimitiveArray.numSbtRecords = 1;

  return obi;
}

bool UnstructuredField::isValid() const
{
  return true;
}

SpatialFieldGPUData UnstructuredField::gpuData() const
{
  SpatialFieldGPUData sf;
  sf.type = SpatialFieldType::UNSTRUCTURED;
  sf.data.unstructured.vertexData
      = m_params.vertexData->beginAs<float>(AddressSpace::GPU);
  sf.data.unstructured.cellsTraversable = m_traversableCells;
  sf.grid = m_uniformGrid.gpuData();
  return sf;
}

void UnstructuredField::cleanup()
{
  m_uniformGrid.cleanup();
}

void UnstructuredField::buildGrid()
{
}

} // namespace visrtx
