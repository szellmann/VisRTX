// Copyright 2024 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include <fstream>
#include "tsd/authoring/importers.hpp"
#include "tsd/authoring/importers/detail/importer_common.hpp"
#include "tsd/core/Logging.hpp"

namespace tsd {

struct ExaBrick {
  int lower[3];
  int size[3];
  int level;
  uint32_t begin;
};

SpatialFieldRef import_EXA(Context &ctx, const char *filepath)
{
  auto ext = extensionOf(filepath);

  auto file = fileOf(filepath);

  std::string bricksFilepath, scalarFilepath;
  if (ext == ".bricks") {
    bricksFilepath = filepath;
    auto baseName = std::string(filepath).substr(
        0, std::string(filepath).length() - std::string(".bricks").length());
    scalarFilepath = baseName+std::string(".scalar");
  } else if (ext == ".scalar") {
    scalarFilepath = filepath;
    auto baseName = std::string(filepath).substr(
        0, std::string(filepath).length() - std::string(".scalar").length());
    bricksFilepath = baseName+std::string(".bricks");
  }

  std::cout << "BRICKS FILE: " << bricksFilepath << '\n';
  std::cout << "SCALAR FILE: " << scalarFilepath << '\n';

  // Indices/scalars are later flattened
  std::vector<float> scalars, orderedScalars;
  std::vector<int> indices;
  std::vector<ExaBrick> bricks;

  std::ifstream scalarFile(scalarFilepath, std::ios::binary | std::ios::ate);
  if (scalarFile.good()) {
    size_t numBytes = scalarFile.tellg();
    scalarFile.close();
    scalarFile.open(scalarFilepath, std::ios::binary);
    if (scalarFile.good()) {
      orderedScalars.resize(numBytes/sizeof(float));
      scalarFile.read((char *)orderedScalars.data(),orderedScalars.size()*sizeof(float));
    }
  }

  // -------------------------------------------------------
  // create brick and index buffers
  // -------------------------------------------------------

  std::ifstream in(bricksFilepath, std::ios::binary);
  if (!in.good()) return {};
  while (!in.eof()) {
    ExaBrick brick;
    in.read((char*)&brick.size,sizeof(brick.size));
    in.read((char*)&brick.lower,sizeof(brick.lower));
    in.read((char*)&brick.level,sizeof(brick.level));
    brick.begin = (int)indices.size();
    if (!in.good())
      break;
    std::vector<int> cellIDs(brick.size[0]*brick.size[1]*brick.size[2]);
    in.read((char*)cellIDs.data(),cellIDs.size()*sizeof(cellIDs[0]));
    indices.insert(indices.end(),cellIDs.begin(),cellIDs.end());
    bricks.push_back(brick);
  }
  std::cout << "#exa: done loading exabricks, found "
            << /*owl::prettyDouble*/((double)bricks.size()) << " bricks with "
            << /*owl::prettyDouble*/((double)indices.size()) << " cells" << std::endl;

  // -------------------------------------------------------
  // flatten cellIDs
  // -------------------------------------------------------

  scalars.resize(orderedScalars.size());
  for (size_t i=0;i<indices.size();i++) {
    if (indices[i] < 0) {
      throw std::runtime_error("overflow in index vector...");
    } else {
      int cellID = indices[i];
      if (cellID < 0)
        throw std::runtime_error("negative cell ID");
      if (cellID >= orderedScalars.size())
        throw std::runtime_error("invalid cell ID");
      scalars[i] = orderedScalars[cellID];
    }
  }
  
  auto blockData = ctx.createArray(ANARI_ARRAY3D, bricks.size());
  {
    auto *dst = (size_t *)blockData->map();
    std::transform(
        bricks.begin(), bricks.end(), dst, [&](const auto &brick) {
          auto block = ctx.createArray(
              ANARI_FLOAT32, brick.size[0], brick.size[1], brick.size[2]);
          block->setData(scalars.data()+brick.begin);
          return block.index();
        });
    blockData->unmap();
  }

  auto blockBounds = ctx.createArray(ANARI_INT32_BOX3, bricks.size());
  {
    auto *dst = (int32_t *)blockBounds->map();
    for (size_t i=0; i<bricks.size(); ++i) {
      int32_t *block = dst+i*6;

      int level = bricks[i].level;

      int lower[3] = {
        bricks[i].lower[0] / (1<<level),
        bricks[i].lower[1] / (1<<level),
        bricks[i].lower[2] / (1<<level)
      };

      block[0] = lower[0];
      block[1] = lower[1];
      block[2] = lower[2];
      block[3] = lower[0] + bricks[i].size[0] - 1;
      block[4] = lower[1] + bricks[i].size[1] - 1;
      block[5] = lower[2] + bricks[i].size[2] - 1;
    }
    blockData->unmap();
  }

  int max_level = -1;
  auto blockLevel = ctx.createArray(ANARI_INT32, bricks.size());
  {
    auto *dst = (int32_t *)blockLevel->map();
    for (size_t i=0; i<bricks.size(); ++i) {
      dst[i] = bricks[i].level;
      max_level = std::max(max_level, dst[i]);
    }
    blockLevel->unmap();
  }

  // --- cellWidth
  std::vector<float> cw;
  for (int l = 0; l <= max_level; ++l) {
    cw.push_back(1 << l);
  }

  auto cellWidth = ctx.createArray(ANARI_FLOAT32, cw.size());
  cellWidth->setData(cw);

  auto field = ctx.createObject<SpatialField>(tokens::spatial_field::amr);
  field->setName(file.c_str());

  field->setParameterObject("cellWidth", *cellWidth);
  field->setParameterObject("block.bounds", *blockBounds);
  field->setParameterObject("block.level", *blockLevel);
  field->setParameterObject("block.data", *blockData);

  logStatus("[import_EXA] ...done!");

  return field;
}

} // namespace tsd
