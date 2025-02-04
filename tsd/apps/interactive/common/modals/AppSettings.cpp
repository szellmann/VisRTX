// Copyright 2024-2025 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "AppSettings.h"
// glfw
#include <GLFW/glfw3.h>

namespace tsd_viewer {

AppSettings::AppSettings() : Modal("AppSettings")
{
  ImGuiIO &io = ImGui::GetIO();
  m_fontScale = io.FontGlobalScale;
}

void AppSettings::buildUI()
{
  bool doUpdate = false;

  doUpdate |= ImGui::DragFloat("font size", &m_fontScale, 0.01f, 0.5f, 4.f);

  if (doUpdate)
    update();

  ImGui::NewLine();

  ImGuiIO &io = ImGui::GetIO();
  if(ImGui::Button("close") || io.KeysDown[GLFW_KEY_ESCAPE])
    this->hide();
}

void AppSettings::update()
{
  ImGuiIO &io = ImGui::GetIO();
  io.FontGlobalScale = m_fontScale;
}

} // namespace tsd_viewer
