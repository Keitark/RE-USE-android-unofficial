// Original model work: Copyright (c) 2025 NVIDIA CORPORATION. All rights reserved.
// Model-specific adaptation: NVIDIA NSCLv1; see licenses/NVIDIA-NSCLv1.txt.
// Minimal bounded JSON reader for the generated, app-private model manifest.
#pragma once
#include <cctype>
#include <cstdlib>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace even_reuse {
struct Json {
  enum Kind { Object, Array, String, Number, Boolean, Null } kind = Null;
  std::map<std::string, Json> object;
  std::vector<Json> array;
  std::string text;
  double number = 0;
  bool boolean = false;
  const Json& at(const std::string& key) const {
    auto found = object.find(key);
    if (kind != Object || found == object.end()) throw std::runtime_error("Manifest missing key: " + key);
    return found->second;
  }
  int integer() const {
    if (kind != Number || number < -2147483648. || number > 2147483647. || number != static_cast<int>(number))
      throw std::runtime_error("Manifest expected bounded integer");
    return static_cast<int>(number);
  }
  std::string string() const {
    if (kind != String) throw std::runtime_error("Manifest expected string");
    return text;
  }
};
class JsonReader {
 public:
  explicit JsonReader(const std::string& text) : text_(text) {
    if (text.size() > 2 * 1024 * 1024) throw std::runtime_error("Manifest too large");
  }
  Json parse() {
    Json value = read(0);
    space();
    if (position_ != text_.size()) fail();
    return value;
  }
 private:
  const std::string& text_;
  size_t position_ = 0;
  [[noreturn]] void fail() const { throw std::runtime_error("Invalid JSON manifest at byte " + std::to_string(position_)); }
  void space() { while (position_ < text_.size() && std::isspace(static_cast<unsigned char>(text_[position_]))) ++position_; }
  bool take(char c) { space(); if (position_ < text_.size() && text_[position_] == c) { ++position_; return true; } return false; }
  unsigned hex4() {
    unsigned value = 0;
    for (int i=0; i<4; ++i) {
      if (position_ == text_.size()) fail();
      const char c=text_[position_++]; unsigned digit;
      if (c>='0' && c<='9') digit=c-'0'; else if(c>='a'&&c<='f') digit=c-'a'+10;
      else if(c>='A'&&c<='F') digit=c-'A'+10; else fail();
      value=(value<<4)|digit;
    }
    return value;
  }
  std::string string() {
    if (!take('"')) fail();
    std::string result;
    while (position_ < text_.size()) {
      char c=text_[position_++];
      if(c=='"') return result;
      if(static_cast<unsigned char>(c)<32) fail();
      if(c!='\\') { result+=c; continue; }
      if(position_==text_.size()) fail();
      c=text_[position_++];
      if(c=='"'||c=='\\'||c=='/') result+=c;
      else if(c=='n') result+='\n'; else if(c=='r') result+='\r'; else if(c=='t') result+='\t';
      else if(c=='b') result+='\b'; else if(c=='f') result+='\f';
      else if(c=='u') {
        unsigned code=hex4();
        if(code>=0xd800&&code<=0xdbff) {
          if(position_+2>text_.size()||text_[position_]!='\\'||text_[position_+1]!='u') fail();
          position_+=2; unsigned lower=hex4(); if(lower<0xdc00||lower>0xdfff) fail();
          code=0x10000+((code-0xd800)<<10)+(lower-0xdc00);
        } else if(code>=0xdc00&&code<=0xdfff) fail();
        if(code<0x80) result+=static_cast<char>(code);
        else if(code<0x800) { result+=static_cast<char>(0xc0|(code>>6)); result+=static_cast<char>(0x80|(code&63)); }
        else if(code<0x10000) { result+=static_cast<char>(0xe0|(code>>12)); result+=static_cast<char>(0x80|((code>>6)&63)); result+=static_cast<char>(0x80|(code&63)); }
        else { result+=static_cast<char>(0xf0|(code>>18)); result+=static_cast<char>(0x80|((code>>12)&63)); result+=static_cast<char>(0x80|((code>>6)&63)); result+=static_cast<char>(0x80|(code&63)); }
      } else fail();
    }
    fail();
  }
  Json read(int depth) {
    if(depth>64) fail();
    space(); if(position_==text_.size()) fail();
    Json out;
    if(take('{')) {
      out.kind=Json::Object; if(take('}')) return out;
      do { auto key=string(); if(!take(':')) fail(); auto value=read(depth+1);
        if(!out.object.emplace(key,std::move(value)).second) fail();
      } while(take(','));
      if(!take('}')) fail(); return out;
    }
    if(take('[')) {
      out.kind=Json::Array; if(take(']')) return out;
      do { out.array.push_back(read(depth+1)); if(out.array.size()>100000) fail(); } while(take(','));
      if(!take(']')) fail(); return out;
    }
    if(text_[position_]=='"') { out.kind=Json::String; out.text=string(); return out; }
    if(text_.compare(position_,4,"true")==0) { position_+=4; out.kind=Json::Boolean; out.boolean=true; return out; }
    if(text_.compare(position_,5,"false")==0) { position_+=5; out.kind=Json::Boolean; return out; }
    if(text_.compare(position_,4,"null")==0) { position_+=4; return out; }
    const size_t begin=position_;
    if(text_[position_]=='-') ++position_;
    if(position_==text_.size() || !std::isdigit(static_cast<unsigned char>(text_[position_]))) fail();
    if(text_[position_]=='0') ++position_;
    else while(position_<text_.size() && std::isdigit(static_cast<unsigned char>(text_[position_]))) ++position_;
    if(position_<text_.size() && text_[position_]=='.') {
      ++position_; const size_t digits=position_;
      while(position_<text_.size()&&std::isdigit(static_cast<unsigned char>(text_[position_]))) ++position_;
      if(digits==position_) fail();
    }
    if(position_<text_.size() && (text_[position_]=='e'||text_[position_]=='E')) {
      ++position_; if(position_<text_.size()&&(text_[position_]=='+'||text_[position_]=='-')) ++position_;
      const size_t digits=position_; while(position_<text_.size()&&std::isdigit(static_cast<unsigned char>(text_[position_]))) ++position_;
      if(digits==position_) fail();
    }
    out.kind=Json::Number; out.number=std::strtod(text_.substr(begin,position_-begin).c_str(),nullptr); return out;
  }
};
inline std::string json_quote(const std::string& value) {
  std::string out="\"";
  for(unsigned char c:value) {
    if(c=='"'||c=='\\') { out+='\\'; out+=static_cast<char>(c); }
    else if(c=='\n') out+="\\n"; else if(c=='\r') out+="\\r"; else if(c=='\t') out+="\\t";
    else if(c<32) out+='?'; else out+=static_cast<char>(c);
  }
  return out+'"';
}
}  // namespace even_reuse
