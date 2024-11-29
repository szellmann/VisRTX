// Copyright 2024 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tsd/authoring/importers.hpp"
#include "tsd/authoring/importers/detail/importer_common.hpp"
// std
#include <fstream>
#include <vector>

namespace tsd {

void import_TRI(Context &ctx, const char *filename)
{
  std::ifstream in(filename, std::ios::binary);

  std::vector<float3> vertex;
  std::vector<int3> index;

  for (;;) {
    int numVerts;
    in.read((char *)&numVerts,sizeof(numVerts));
    if (!in.good() || in.eof())
      break;

    size_t vertOffset = vertex.size();
    vertex.resize(vertex.size()+numVerts);
    in.read((char *)(vertex.data()+vertOffset),numVerts*sizeof(float3));

    int numTris;
    in.read((char *)&numTris,sizeof(numTris));

    size_t triOffset = index.size();
    index.resize(index.size()+numTris);
    in.read((char *)(index.data()+triOffset),numTris*sizeof(int3));
    for (size_t i=triOffset; i<index.size(); ++i) {
      index[i].x += vertOffset;
      index[i].y += vertOffset;
      index[i].z += vertOffset;
    }

    if (numTris >1000000) break;
  }

  auto objectName = fileOf(std::string(filename)) + " (TRI file)";

  std::cout << objectName << " has " << index.size()
      << " tris and " << vertex.size() << " verts\n";

  auto ply_root =
      ctx.tree.insert_last_child(ctx.tree.root(), fileOf(filename).c_str());
  auto mesh = ctx.createObject<Geometry>(tokens::geometry::triangle);

  mesh->setName((objectName + "_mesh").c_str());

  auto makeArray1DForMesh = [&](Token parameterName,
                                anari::DataType type,
                                const void *ptr,
                                size_t size) {
    auto arr = ctx.createArray(type, size);
    arr->setData(ptr);
    mesh->setParameterObject(parameterName, *arr);
  };

  makeArray1DForMesh("vertex.position"_t,
      ANARI_FLOAT32_VEC3,
      vertex.data(),
      vertex.size());

  makeArray1DForMesh("primitive.index"_t,
      ANARI_UINT32_VEC3,
      /*(const uint3 *)*/index.data(),
      index.size());

  auto mat = ctx.createObject<Material>(tokens::material::matte);
  mat->setParameter("color"_t, float3(0.8f));
  mat->setParameter("opacity"_t, 1.f);
  mat->setParameter("alphaMode"_t, "opaque");
  mat->parameter("alphaMode"_t)->setStringSelection(0);
  mat->setName((objectName + " material").c_str());

  auto surface = ctx.createSurface(objectName.c_str(), mesh, mat);
  ctx.tree.insert_last_child(
      ply_root, utility::Any(ANARI_SURFACE, surface.index()));

  if (1) {
    tsd::mat4 mat = math::identity;
    mat[1][1] = -1;

    auto tr = ctx.tree.insert_last_child(
        ctx.tree.root(), {mat, "mirror"});

    ctx.tree.insert_last_child(tr, utility::Any(ANARI_SURFACE, surface.index()));
  }

}

} // namespace tsd
