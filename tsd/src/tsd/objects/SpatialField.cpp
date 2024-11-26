// Copyright 2024 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tsd/objects/SpatialField.hpp"
#include "tsd/algorithms/computeScalarRange.hpp"
#include "tsd/core/Context.hpp"
#include "tsd/core/Logging.hpp"
// std
#include <optional>

namespace tsd {

SpatialField::SpatialField(Token stype) : Object(ANARI_SPATIAL_FIELD, stype)
{
  if (stype == tokens::spatial_field::structuredRegular) {
    addParameter("origin")
        .setValue(float3(0.f, 0.f, 0.f))
        .setDescription("bottom-left corner of the field");
    addParameter("spacing")
        .setValue(float3(1.f, 1.f, 1.f))
        .setMin(float3(0.f, 0.f, 0.f))
        .setDescription("voxel size in object-space units");
    addParameter("data")
        .setValue({ANARI_ARRAY3D, INVALID_INDEX})
        .setDescription("vertex-centered voxel data");
  } else if (stype == tokens::spatial_field::amr) {
    addParameter("gridOrigin")
        .setValue(float3(0.f, 0.f, 0.f))
        .setDescription("bottom-left corner of the field");
    addParameter("gridSpacing")
        .setValue(float3(1.f, 1.f, 1.f))
        .setMin(float3(0.f, 0.f, 0.f))
        .setDescription("voxel size in object-space units");
    addParameter("cellWidth")
        .setValue({ANARI_ARRAY1D, INVALID_INDEX})
        .setDescription("array of each level's cell width");
    addParameter("block.bounds")
        .setValue({ANARI_ARRAY1D, INVALID_INDEX})
        .setDescription("array of grid sizes, in voxels, for each AMR block");
    addParameter("block.level")
        .setValue({ANARI_ARRAY1D, INVALID_INDEX})
        .setDescription("array of each block's refinement level");
    addParameter("block.data")
        .setValue({ANARI_ARRAY1D, INVALID_INDEX})
        .setDescription("array of 3D arrays containing the scalar voxel data");
  }
}

anari::Object SpatialField::makeANARIObject(anari::Device d) const
{
  return anari::newObject<anari::SpatialField>(d, subtype().c_str());
}

float2 SpatialField::computeValueRange()
{
  float2 retval{0.f, 1.f};
  auto *ctx = this->context();
  if (!ctx)
    return retval;

  auto getDataRangeFromParameter = [&](Parameter *p) -> std::optional<float2> {
    if (!p || !anari::isArray(p->value().type()))
      return {};
    else if (auto a = ctx->getObject<Array>(p->value().getAsObjectIndex()); a)
      return algorithm::computeScalarRange(*a);
    else
      return {};
  };

  if (subtype() == tokens::spatial_field::structuredRegular) {
    if (auto range = getDataRangeFromParameter(parameter("data")); range)
      retval = *range;
  } else if (subtype() == tokens::spatial_field::amr) {
    if (auto range = getDataRangeFromParameter(parameter("block.data")); range)
      retval = *range;
  } else {
    logWarning(
        "implementation not yet provided for computing the data "
        "range of '%s' spatial fields",
        subtype().c_str());
  }

  return retval;
}

namespace tokens::spatial_field {

const Token structuredRegular = "structuredRegular";
const Token amr = "amr";

} // namespace tokens::spatial_field

} // namespace tsd
