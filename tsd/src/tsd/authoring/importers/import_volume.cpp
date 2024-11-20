// Copyright 2024 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tsd/authoring/importers.hpp"
#include "tsd/authoring/importers/detail/importer_common.hpp"
// std
#include <cstdio>

namespace tsd {

VolumeRef import_volume(Context &ctx,
    const char *filepath,
    ArrayRef colorArray,
    ArrayRef opacityArray)
{
  SpatialFieldRef field;

  auto ext = extensionOf(filepath);
  if (ext == ".raw")
    field = import_RAW(ctx, filepath);
  else if (ext == ".flash")
    field = import_FLASH(ctx, filepath);

  float2 valueRange{0.f, 1.f};
  if (field)
    valueRange = field->computeValueRange();

  auto volume = ctx.createObject<Volume>(tokens::volume::transferFunction1D);
  volume->setName(fileOf(filepath).c_str());
  volume->setParameterObject("value", *field);
  volume->setParameterObject("color", *colorArray);
  volume->setParameterObject("opacity", *opacityArray);
  volume->setParameter("densityScale", 0.1f);
  volume->setParameter("valueRange", ANARI_FLOAT32_BOX1, &valueRange);

  ctx.tree.insert_last_child(
      ctx.tree.root(), utility::Any(ANARI_VOLUME, volume.index()));

  return volume;
}

} // namespace tsd
