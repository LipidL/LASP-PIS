#pragma once
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace checkpoint {
// Data-only representation. Serialized globals are labels, never executable
// code.
struct Value;
using Ptr = std::shared_ptr<Value>;
struct Tensor {
  std::vector<int64_t> shape, stride;
  std::shared_ptr<std::vector<char>> storage;
  std::string dtype;
  int64_t offset = 0;
  std::vector<double> doubles() const;
};
struct Value {
  enum Kind {
    Null,
    Number,
    String,
    Sequence,
    Mapping,
    Global,
    Object,
    TensorValue
  } kind = Null;
  double number = 0;
  std::string text;
  std::vector<Ptr> items;
  std::map<std::string, Ptr> fields;
  Tensor tensor;
  Ptr get(const std::string &key) const;
  Ptr at(std::size_t i) const;
  int64_t integer() const;
};
class Archive {
  struct Impl;
  std::shared_ptr<Impl> impl_;

public:
  explicit Archive(const std::string &path);
  explicit Archive(const std::vector<char> &bytes);
  std::vector<char> record(const std::string &name) const;
  std::vector<std::string> records() const;
  Ptr root() const;
};
Ptr parse(const std::vector<char> &pickle, const Archive &archive);
// Python modules keep attributes, parameters, buffers, and children separately.
Ptr attribute(const Ptr &object, const std::string &name);
Ptr module(const Ptr &object, const std::string &dotted_path);
std::map<std::string, Tensor> state_dict(const Ptr &object);
bool is_zip(const std::string &path);
} // namespace checkpoint
