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
  }
}

anari::Object SpatialField::makeANARIObject(anari::Device d) const
{
  return anari::newObject<anari::SpatialField>(d, subtype().c_str());
}

float2 SpatialField::computeValueRange(const Context *ctx)
{
  auto getDataRangeFromParameter = [&](Parameter *p) -> std::optional<float2> {
    if (!p || !anari::isArray(p->value().type()))
      return {};
    else if (auto a = ctx->getObject<Array>(p->value().getAsObjectIndex()); a)
      return algorithm::computeScalarRange(*a);
    else
      return {};
  };

  if (subtype() == tokens::spatial_field::structuredRegular) {
    auto range = getDataRangeFromParameter(parameter("data"));
    return range ? *range : float2{0.f, 1.f};
  } else if (subtype() == tokens::spatial_field::amr) {
    // TODO
    return {0.f, 1.f};
  } else {
    logWarning(
        "implementation not yet provided for computing the data "
        "range of '%s' spatial fields",
        subtype().c_str());
    return {0.f, 1.f};
  }
}

namespace tokens::spatial_field {

const Token structuredRegular = "structuredRegular";
const Token amr = "amr";

} // namespace tokens::spatial_field

} // namespace tsd
