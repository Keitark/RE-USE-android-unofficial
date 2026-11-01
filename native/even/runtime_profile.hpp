// Original model work: Copyright (c) 2025 NVIDIA CORPORATION. All rights reserved.
// Model-specific adaptation: NVIDIA NSCLv1; see licenses/NVIDIA-NSCLv1.txt.
// Explicit, bounded per-session settings for the existing app runtime.
// This does not select another provider or change device/system scheduling.
#pragma once
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>

namespace even_reuse {
inline std::map<std::string,std::string> runtime_profile_options(
    const std::string& profile,const std::string& version) {
  if(profile=="default")return {};
  if(profile=="no_spin")return {{"session.intra_op.allow_spinning","0"}};
  if(profile=="dynamic4")return {{"session.dynamic_block_base","4"}};
  if(profile=="bounded_spin") {
    int major=0,minor=0;char separator=0;std::istringstream parsed(version);
    if(!(parsed>>major>>separator>>minor)||separator!='.'||major<1||(major==1&&minor<27))
      throw std::runtime_error("bounded_spin requires ONNX Runtime 1.27 or later");
    return {{"session.intra_op.allow_spinning","1"},
            {"session.intra_op.spin_duration_us","1000"},
            {"session.intra_op.spin_backoff_max","4"},
            {"session.force_spinning_stop","1"}};
  }
  throw std::runtime_error("Unsupported runtime_profile; expected default/no_spin/bounded_spin/dynamic4");
}
} // namespace even_reuse
