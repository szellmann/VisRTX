// Copyright 2024 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "AppCore.h"
// anari_viewer
#include "anari_viewer/windows/Window.h"
// std
#include <functional>

namespace tsd_viewer {

using JacobiUpdateCallback = std::function<void()>;

struct SolverControls : public anari_viewer::windows::Window
{
  SolverControls(AppCore *state, const char *name = "Solver Controls");

  void buildUI() override;
  void setDataArray(tsd::Array *a);
  void setUpdateCallback(JacobiUpdateCallback cb);

 private:
  void resetSolver();
  void iterateSolver();
  void exportRAW();

  AppCore *m_core{nullptr};
  tsd::Array *m_data{nullptr};
  int m_iterationsPerCycle{1};
  int m_totalIterations{0};
  JacobiUpdateCallback m_cb;
  bool m_playing{false};
};

} // namespace tsd_viewer
