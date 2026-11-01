// Original model work: Copyright (c) 2025 NVIDIA CORPORATION. All rights reserved.
// Model-specific adaptation: NVIDIA NSCLv1; see licenses/NVIDIA-NSCLv1.txt.
// Fixed-capacity storage for owned PCM frames. No allocation in push/pop/reset.
#pragma once
#include <array>
#include <cstddef>
#include <stdexcept>
#include <utility>

namespace even_reuse {
template<class Frame,size_t Capacity=64> class FrameQueue {
  static_assert(Capacity>0,"Frame queue capacity must be positive");
 public:
  size_t size()const{return size_;}
  bool empty()const{return size_==0;}
  Frame& front(){if(empty())throw std::runtime_error("Empty frame queue");return frames_[head_];}
  const Frame& front()const{if(empty())throw std::runtime_error("Empty frame queue");return frames_[head_];}
  template<class Value> void push_back(Value&& value){
    if(size_==Capacity)throw std::runtime_error("Frame queue capacity exceeded");
    frames_[(head_+size_)%Capacity]=std::forward<Value>(value);++size_;
  }
  void pop_front(){if(empty())throw std::runtime_error("Empty frame queue");head_=(head_+1)%Capacity;--size_;}
  void clear(){head_=0;size_=0;}
 private:
  std::array<Frame,Capacity>frames_{};size_t head_=0,size_=0;
};
} // namespace even_reuse
