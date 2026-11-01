// Original model work: Copyright (c) 2025 NVIDIA CORPORATION. All rights reserved.
// Model-specific adaptation: NVIDIA NSCLv1; see licenses/NVIDIA-NSCLv1.txt.
// Optional Android API33+ scheduling feedback. Older/unsupported devices keep
// the ordinary CPU path. No affinity, core-online or system setting is changed.
#pragma once
#include <android/performance_hint.h>
#include <algorithm>
#include <cstdint>
#include <dlfcn.h>
#include <vector>

namespace even_reuse {
class PerformanceHint {
  // Exact API33 C signatures, resolved dynamically so the APK can still load
  // on API26-32 without a hard dependency on unavailable symbols.
  using Get=APerformanceHintManager*(*)();
  using Create=APerformanceHintSession*(*)(APerformanceHintManager*,const int32_t*,size_t,int64_t);
  using Report=int(*)(APerformanceHintSession*,int64_t);
  using Close=void(*)(APerformanceHintSession*);
 public:
  explicit PerformanceHint(const std::vector<int32_t>& threads,int64_t target_ns=20000000) {
    library_=dlopen("libandroid.so",RTLD_NOW|RTLD_LOCAL);if(!library_)return;
    auto get=reinterpret_cast<Get>(dlsym(library_,"APerformanceHint_getManager"));
    auto create=reinterpret_cast<Create>(dlsym(library_,"APerformanceHint_createSession"));
    report_=reinterpret_cast<Report>(dlsym(library_,"APerformanceHint_reportActualWorkDuration"));
    close_=reinterpret_cast<Close>(dlsym(library_,"APerformanceHint_closeSession"));
    available_=get&&create&&report_&&close_;if(!available_)return;
    if(auto* manager=get())session_=create(manager,threads.data(),threads.size(),target_ns);
  }
  ~PerformanceHint(){if(session_)close_(session_);if(library_)dlclose(library_);}
  PerformanceHint(const PerformanceHint&)=delete;PerformanceHint& operator=(const PerformanceHint&)=delete;
  bool available()const{return available_;}bool active()const{return session_!=nullptr;}
  int report(int64_t nanoseconds){return session_?report_(session_,std::max<int64_t>(1,nanoseconds)):0;}
 private:
  void* library_=nullptr;APerformanceHintSession* session_=nullptr;bool available_=false;
  Report report_=nullptr;Close close_=nullptr;
};
} // namespace even_reuse
