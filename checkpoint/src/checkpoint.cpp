#include <algorithm>
#include <checkpoint/checkpoint.hpp>
#include <cmath>
#include <cstring>
#include <fstream>
#include <functional>
#include <limits>
#include <set>
#include <sstream>
#include <stdexcept>
#include <zlib.h>

namespace checkpoint {
namespace {
[[noreturn]] void fail(const std::string &s) {
  throw std::runtime_error("checkpoint: " + s);
}
Ptr value(Value::Kind kind) {
  auto p = std::make_shared<Value>();
  p->kind = kind;
  return p;
}
Ptr numeric(double n) {
  auto p = value(Value::Number);
  p->number = n;
  return p;
}
std::string key(const Ptr &p) {
  if (p->kind == Value::String)
    return p->text;
  if (p->kind == Value::Number)
    return std::to_string(p->integer());
  fail("unsupported dictionary key");
}
std::vector<int64_t> integers(const Ptr &p) {
  std::vector<int64_t> result;
  for (auto &x : p->items)
    result.push_back(x->integer());
  return result;
}
} // namespace
Ptr Value::get(const std::string &k) const {
  auto it = fields.find(k);
  if (it == fields.end())
    fail("missing attribute " + k + " on " + text);
  return it->second;
}
Ptr Value::at(std::size_t i) const {
  if (i >= items.size())
    fail("sequence index out of range on " + text);
  return items[i];
}
int64_t Value::integer() const {
  if (kind != Number || !std::isfinite(number) ||
      number != std::trunc(number) || std::abs(number) > 9007199254740991.)
    fail("invalid integer");
  return int64_t(number);
}

struct Archive::Impl {
  struct Entry {
    uint16_t method;
    uint32_t crc;
    uint64_t compressed, size, offset;
  };
  std::vector<char> bytes;
  std::map<std::string, Entry> entries;
  uint64_t number(uint64_t offset, unsigned width) const {
    if (offset > bytes.size() || width > bytes.size() - offset)
      fail("truncated ZIP metadata");
    uint64_t n = 0;
    for (unsigned k = 0; k < width; ++k)
      n |= uint64_t(static_cast<unsigned char>(bytes[offset + k])) << (k * 8);
    return n;
  }
  explicit Impl(const std::string &path) {
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in)
      fail("cannot open " + path);
    auto n = in.tellg();
    if (n < 0 || uint64_t(n) > (uint64_t(1) << 31))
      fail("archive exceeds 2 GiB limit");
    bytes.resize(size_t(n));
    in.seekg(0);
    if (!in.read(bytes.data(), bytes.size()))
      fail("cannot read " + path);
    index();
  }
  explicit Impl(const std::vector<char> &b) : bytes(b) { index(); }
  void index() {
    if (bytes.size() < 22 || bytes.size() > (uint64_t(1) << 31))
      fail("invalid ZIP size");
    uint64_t end = bytes.size() - 22, minimum = end > 65535 ? end - 65535 : 0;
    while (number(end, 4) != 0x06054b50 ||
           end + 22 + number(end + 20, 2) != bytes.size()) {
      if (end == minimum)
        fail("missing ZIP end record");
      --end;
    }
    if (number(end + 4, 2) || number(end + 6, 2))
      fail("split ZIP archives are unsupported");
    uint64_t count = number(end + 10, 2), size = number(end + 12, 4),
             offset = number(end + 16, 4);
    if (count == 65535 || size == 0xffffffff || offset == 0xffffffff) {
      if (end < 20 || number(end - 20, 4) != 0x07064b50 ||
          number(end - 16, 4) != 0 || number(end - 4, 4) != 1)
        fail("invalid ZIP64 locator");
      uint64_t z = number(end - 12, 8);
      if (number(z, 4) != 0x06064b50 || number(z + 4, 8) < 44 ||
          number(z + 16, 4) || number(z + 20, 4))
        fail("invalid ZIP64 end record");
      count = number(z + 32, 8);
      size = number(z + 40, 8);
      offset = number(z + 48, 8);
      if (number(z + 24, 8) != count)
        fail("inconsistent ZIP64 entry counts");
    } else if (number(end + 8, 2) != count)
      fail("inconsistent ZIP entry counts");
    if (count > 100000 || offset > end || size > end - offset)
      fail("invalid ZIP central directory");
    uint64_t cursor = offset, limit = offset + size;
    std::string prefix;
    for (uint64_t k = 0; k < count; ++k) {
      if (cursor > limit || limit - cursor < 46 ||
          number(cursor, 4) != 0x02014b50)
        fail("invalid ZIP entry");
      auto flags = number(cursor + 8, 2);
      if (flags & 1)
        fail("encrypted ZIP is unsupported");
      Entry entry{uint16_t(number(cursor + 10, 2)),
                  uint32_t(number(cursor + 16, 4)), number(cursor + 20, 4),
                  number(cursor + 24, 4), number(cursor + 42, 4)};
      uint64_t length = number(cursor + 28, 2), extra = number(cursor + 30, 2),
               comment = number(cursor + 32, 2);
      if (length + extra + comment > limit - cursor - 46)
        fail("truncated ZIP entry");
      std::string name(bytes.data() + cursor + 46, size_t(length));
      bool need_size = entry.size == 0xffffffff,
           need_compressed = entry.compressed == 0xffffffff,
           need_offset = entry.offset == 0xffffffff;
      for (uint64_t p = cursor + 46 + length, e = p + extra; p < e;) {
        if (e - p < 4)
          fail("invalid ZIP extra field");
        auto tag = number(p, 2), n = number(p + 2, 2);
        p += 4;
        if (n > e - p)
          fail("truncated ZIP extra field");
        if (tag == 1) {
          uint64_t q = p;
          auto next = [&]() {
            if (p + n - q < 8)
              fail("truncated ZIP64 extra field");
            auto v = number(q, 8);
            q += 8;
            return v;
          };
          if (need_size) {
            entry.size = next();
            need_size = false;
          }
          if (need_compressed) {
            entry.compressed = next();
            need_compressed = false;
          }
          if (need_offset) {
            entry.offset = next();
            need_offset = false;
          }
        }
        p += n;
      }
      if (need_size || need_compressed || need_offset || number(cursor + 34, 2))
        fail("unsupported ZIP64 entry");
      if (entry.method != 0 && entry.method != 8)
        fail("unsupported ZIP compression");
      if (entry.size > (uint64_t(1) << 30) || entry.offset > offset ||
          offset - entry.offset < 30)
        fail("invalid ZIP record bounds");
      auto slash = name.find('/');
      if (slash == std::string::npos || slash == 0)
        fail("expected PyTorch ZIP root directory");
      if (prefix.empty())
        prefix = name.substr(0, slash + 1);
      if (name.compare(0, prefix.size(), prefix) != 0)
        fail("inconsistent ZIP root");
      name = name.substr(prefix.size());
      if (!entries.emplace(name, entry).second)
        fail("duplicate ZIP record: " + name);
      cursor += 46 + length + extra + comment;
    }
    if (cursor != limit)
      fail("trailing ZIP directory data");
  }
  std::vector<char> record(const std::string &name) const {
    auto it = entries.find(name);
    if (it == entries.end())
      fail("missing ZIP record: " + name);
    const auto &e = it->second;
    if (number(e.offset, 4) != 0x04034b50 || (number(e.offset + 6, 2) & 1) ||
        number(e.offset + 8, 2) != e.method)
      fail("invalid local ZIP header");
    uint64_t begin =
        e.offset + 30 + number(e.offset + 26, 2) + number(e.offset + 28, 2);
    if (begin > bytes.size() || e.compressed > bytes.size() - begin)
      fail("truncated ZIP record");
    std::vector<char> out(size_t(e.size));
    if (e.method == 0) {
      if (e.compressed != e.size)
        fail("stored ZIP size mismatch");
      if (e.size)
        std::memcpy(out.data(), bytes.data() + begin, size_t(e.size));
    } else {
      z_stream stream{};
      char empty;
      stream.next_in =
          reinterpret_cast<Bytef *>(const_cast<char *>(bytes.data() + begin));
      stream.avail_in = uInt(e.compressed);
      stream.next_out =
          reinterpret_cast<Bytef *>(out.empty() ? &empty : out.data());
      stream.avail_out = uInt(out.empty() ? 1 : out.size());
      if (inflateInit2(&stream, -MAX_WBITS) != Z_OK)
        fail("cannot initialize ZIP decompression");
      int status = inflate(&stream, Z_FINISH);
      inflateEnd(&stream);
      if (status != Z_STREAM_END || stream.total_out != e.size ||
          stream.total_in != e.compressed)
        fail("invalid compressed ZIP data");
    }
    auto crc =
        crc32(0, reinterpret_cast<const Bytef *>(out.data()), uInt(out.size()));
    if (crc != e.crc)
      fail("ZIP checksum mismatch: " + name);
    return out;
  }
};
Archive::Archive(const std::string &path)
    : impl_(std::make_shared<Impl>(path)) {}
Archive::Archive(const std::vector<char> &b)
    : impl_(std::make_shared<Impl>(b)) {}
std::vector<char> Archive::record(const std::string &name) const {
  return impl_->record(name);
}
std::vector<std::string> Archive::records() const {
  std::vector<std::string> result;
  for (const auto &entry : impl_->entries)
    result.push_back(entry.first);
  return result;
}
Ptr Archive::root() const {
  if (impl_->entries.count("byteorder")) {
    auto b = record("byteorder");
    if (std::string(b.begin(), b.end()) != "little")
      fail("only little-endian storage is supported");
  }
  return parse(record("data.pkl"), *this);
}
bool is_zip(const std::string &path) {
  std::ifstream f(path, std::ios::binary);
  char b[4]{};
  if (!f)
    fail("cannot open " + path);
  f.read(b, 4);
  return std::memcmp(b, "PK\003\004", 4) == 0;
}

std::vector<double> Tensor::doubles() const {
  if (!storage || shape.size() != stride.size() || shape.size() > 8 ||
      offset < 0)
    fail("invalid tensor metadata");
  size_t count = 1, width = 0;
  if (dtype == "DoubleStorage" || dtype == "LongStorage")
    width = 8;
  else if (dtype == "FloatStorage" || dtype == "IntStorage")
    width = 4;
  else if (dtype == "BoolStorage" || dtype == "ByteStorage")
    width = 1;
  else
    fail("unsupported tensor dtype " + dtype);
  for (size_t k = 0; k < shape.size(); ++k) {
    if (shape[k] < 0 || stride[k] < 0 || uint64_t(shape[k]) > (1u << 27) ||
        (shape[k] && count > (1u << 27) / uint64_t(shape[k])))
      fail("invalid tensor shape or stride");
    count *= size_t(shape[k]);
  }
  std::vector<double> out(count);
  for (size_t i = 0; i < count; ++i) {
    uint64_t off = uint64_t(offset), index = i;
    for (size_t k = shape.size(); k-- > 0;) {
      uint64_t coord = index % uint64_t(shape[k]);
      index /= uint64_t(shape[k]);
      if (coord && uint64_t(stride[k]) >
                       (std::numeric_limits<uint64_t>::max() - off) / coord)
        fail("tensor offset overflow");
      off += coord * uint64_t(stride[k]);
    }
    if (off >= storage->size() / width)
      fail("tensor exceeds storage");
    const char *p = storage->data() + off * width;
    if (dtype == "DoubleStorage") {
      double x;
      std::memcpy(&x, p, 8);
      out[i] = x;
    } else if (dtype == "FloatStorage") {
      float x;
      std::memcpy(&x, p, 4);
      out[i] = x;
    } else if (dtype == "LongStorage") {
      int64_t x;
      std::memcpy(&x, p, 8);
      out[i] = double(x);
    } else if (dtype == "IntStorage") {
      int32_t x;
      std::memcpy(&x, p, 4);
      out[i] = x;
    } else
      out[i] = static_cast<unsigned char>(*p);
    if (!std::isfinite(out[i]))
      fail("nonfinite tensor value");
  }
  return out;
}

Ptr parse(const std::vector<char> &data, const Archive &archive) {
  if (data.size() > 32 * 1024 * 1024)
    fail("pickle record too large");
  size_t pos = 0, operations = 0;
  std::vector<Ptr> stack, memo;
  std::vector<size_t> marks;
  std::map<std::string, std::shared_ptr<std::vector<char>>> storages;
  size_t storage_bytes = 0;
  auto byte = [&]() -> unsigned {
    if (pos >= data.size())
      fail("truncated pickle");
    return static_cast<unsigned char>(data[pos++]);
  };
  auto uint = [&](int n) {
    uint64_t x = 0;
    for (int k = 0; k < n; ++k)
      x |= uint64_t(byte()) << (8 * k);
    return x;
  };
  auto string = [&](size_t n) {
    if (n > data.size() - pos)
      fail("truncated string");
    std::string s(data.data() + pos, n);
    pos += n;
    return s;
  };
  auto line = [&]() {
    auto begin = pos;
    while (pos < data.size() && data[pos] != '\n')
      ++pos;
    if (pos == data.size())
      fail("truncated global");
    auto s = std::string(data.data() + begin, pos - begin);
    ++pos;
    return s;
  };
  auto pop = [&]() {
    if (stack.empty())
      fail("pickle stack underflow");
    auto x = stack.back();
    stack.pop_back();
    return x;
  };
  auto mark = [&]() {
    if (marks.empty())
      fail("missing pickle mark");
    auto n = marks.back();
    marks.pop_back();
    if (n > stack.size())
      fail("invalid mark");
    return n;
  };
  auto seq = [&](size_t n) {
    if (n > stack.size())
      fail("tuple underflow");
    auto x = value(Value::Sequence);
    x->items.assign(stack.end() - n, stack.end());
    stack.resize(stack.size() - n);
    stack.push_back(x);
  };
  auto assign = [&](Ptr dict, size_t start) {
    if ((stack.size() - start) % 2)
      fail("odd dictionary payload");
    for (size_t k = start; k < stack.size(); k += 2)
      dict->fields[key(stack[k])] = stack[k + 1];
    stack.resize(start);
  };
  auto memo_put = [&](size_t i) {
    if (i > 1000000 || stack.empty())
      fail("invalid memo");
    if (memo.size() <= i)
      memo.resize(i + 1);
    memo[i] = stack.back();
  };
  auto memo_get = [&](size_t i) {
    if (i >= memo.size() || !memo[i])
      fail("invalid memo reference");
    stack.push_back(memo[i]);
  };
  while (pos < data.size()) {
    if (++operations > 10000000 || stack.size() > 1000000 ||
        marks.size() > 10000)
      fail("pickle resource limit");
    unsigned op = byte();
    switch (op) {
    case 0x80: {
      auto p = byte();
      if (p < 2 || p > 5)
        fail("unsupported pickle protocol");
      break;
    }
    case 0x95:
      uint(8);
      break; // FRAME; all reads still bounded by the record
    case '.':
      if (stack.size() != 1 || !marks.empty() || pos != data.size())
        fail("invalid pickle end");
      return stack[0];
    case '(':
      marks.push_back(stack.size());
      break;
    case 'N':
      stack.push_back(value(Value::Null));
      break;
    case 0x88:
      stack.push_back(numeric(1));
      break;
    case 0x89:
      stack.push_back(numeric(0));
      break;
    case 'K':
      stack.push_back(numeric(byte()));
      break;
    case 'M':
      stack.push_back(numeric(uint(2)));
      break;
    case 'J':
      stack.push_back(numeric(int32_t(uint(4))));
      break;
    case 0x8a: {
      unsigned n = byte();
      if (n > 7)
        fail("integer too large");
      uint64_t v = uint(n);
      if (n && (v & (uint64_t(1) << (8 * n - 1))))
        v |= (~uint64_t(0)) << (8 * n);
      stack.push_back(numeric(double(int64_t(v))));
      break;
    }
    case 'G': {
      uint64_t bits = 0;
      for (int i = 0; i < 8; ++i)
        bits = (bits << 8) | byte();
      double x;
      std::memcpy(&x, &bits, 8);
      stack.push_back(numeric(x));
      break;
    }
    case 'X':
    case 0x8c:
    case 'B':
    case 'C': {
      size_t n = (op == 0x8c || op == 'C') ? byte() : uint(4);
      auto x = value(Value::String);
      x->text = string(n);
      stack.push_back(x);
      break;
    }
    case 'c': {
      auto m = line();
      auto n = line();
      auto x = value(Value::Global);
      x->text = m + " " + n;
      stack.push_back(x);
      break;
    }
    case 0x93: {
      auto n = pop(), m = pop();
      auto x = value(Value::Global);
      x->text = m->text + " " + n->text;
      stack.push_back(x);
      break;
    }
    case 'q':
      memo_put(byte());
      break;
    case 'r':
      memo_put(uint(4));
      break;
    case 0x94:
      memo_put(memo.size());
      break;
    case 'h':
      memo_get(byte());
      break;
    case 'j':
      memo_get(uint(4));
      break;
    case ')':
    case ']':
    case 0x8f:
      stack.push_back(value(Value::Sequence));
      break;
    case '}':
      stack.push_back(value(Value::Mapping));
      break;
    case 0x85:
      seq(1);
      break;
    case 0x86:
      seq(2);
      break;
    case 0x87:
      seq(3);
      break;
    case 't': {
      auto n = mark();
      seq(stack.size() - n);
      break;
    }
    case 'l': {
      auto n = mark();
      seq(stack.size() - n);
      break;
    }
    case 'd': {
      auto n = mark();
      auto x = value(Value::Mapping);
      assign(x, n);
      stack.push_back(x);
      break;
    }
    case 'a': {
      auto x = pop();
      if (stack.empty())
        fail("append underflow");
      stack.back()->items.push_back(x);
      break;
    }
    case 'e':
    case 0x90: {
      auto n = mark();
      if (!n)
        fail("append underflow");
      auto x = stack[n - 1];
      x->items.insert(x->items.end(), stack.begin() + n, stack.end());
      stack.resize(n);
      break;
    }
    case 's': {
      auto v = pop(), k = pop();
      if (stack.empty())
        fail("setitem underflow");
      stack.back()->fields[key(k)] = v;
      break;
    }
    case 'u': {
      auto n = mark();
      if (!n)
        fail("setitems underflow");
      assign(stack[n - 1], n);
      break;
    }
    case 'Q': {
      auto pid = pop();
      if (pid->items.size() < 5 || pid->at(0)->text != "storage")
        fail("unsupported persistent reference");
      auto name = pid->at(2)->text;
      auto dtype = pid->at(1)->text;
      auto space = dtype.find_last_of(' ');
      dtype = dtype.substr(space + 1);
      if (name.find_first_not_of("0123456789") != std::string::npos)
        fail("invalid storage key");
      if (!storages.count(name)) {
        auto bytes = archive.record("data/" + name);
        storage_bytes += bytes.size();
        if (storage_bytes > (size_t(1) << 31))
          fail("checkpoint storage limit");
        storages[name] = std::make_shared<std::vector<char>>(std::move(bytes));
      }
      auto x = value(Value::TensorValue);
      x->tensor.storage = storages[name];
      x->tensor.dtype = dtype;
      x->tensor.shape = {pid->at(4)->integer()};
      x->tensor.stride = {1};
      stack.push_back(x);
      break;
    }
    case 'R':
    case 0x81: {
      auto args = pop(), cls = pop();
      const auto &name = cls->text;
      if (cls->kind != Value::Global)
        fail("invalid constructor");
      Ptr x;
      if (name == "collections OrderedDict") {
        x = value(Value::Mapping);
        if (!args->items.empty())
          for (auto &pair : args->at(0)->items)
            x->fields[key(pair->at(0))] = pair->at(1);
      } else if (name == "torch._utils _rebuild_tensor_v2" ||
                 name == "torch._utils _rebuild_tensor") {
        x = value(Value::TensorValue);
        x->tensor = args->at(0)->tensor;
        x->tensor.offset = args->at(1)->integer();
        x->tensor.shape = integers(args->at(2));
        x->tensor.stride = integers(args->at(3));
      } else if (name == "torch._utils _rebuild_parameter" ||
                 name == "torch._utils _rebuild_parameter_with_state" ||
                 name == "torch.jit._pickle restore_type_tag") {
        x = args->at(0);
      } else if (name == "torch.jit._pickle build_intlist" ||
                 name == "torch.jit._pickle build_tensorlist" ||
                 name == "__builtin__ set" || name == "builtins set") {
        x = args->items.empty() ? value(Value::Sequence) : args->at(0);
      } else if (name == "_codecs encode") {
        if (args->at(1)->text != "latin1")
          fail("unsupported byte encoding");
        x = value(Value::String);
        const auto &s = args->at(0)->text;
        for (size_t i = 0; i < s.size(); ++i) {
          unsigned c = static_cast<unsigned char>(s[i]);
          if (c >= 0xc2 && c <= 0xc3) {
            if (++i >= s.size())
              fail("invalid latin1");
            c = ((c & 3) << 6) | (static_cast<unsigned char>(s[i]) & 63);
          } else if (c >= 128)
            fail("invalid latin1");
          x->text.push_back(char(c));
        }
      } else {
        // Preserve inert records; never import a class or invoke a reducer.
        x = value(Value::Object);
        x->text = name;
        x->items = args->items;
        if (name == "e3nn.o3._irreps Irreps" ||
            name == "e3nn.o3._irreps Irrep") {
          if (x->items.size() == 1 && x->items[0]->kind == Value::Sequence)
            x->items = x->items[0]->items;
        }
      }
      stack.push_back(x);
      break;
    }
    case 'b': {
      auto state = pop();
      if (stack.empty())
        fail("build underflow");
      auto obj = stack.back();
      if (state->kind == Value::Mapping || state->kind == Value::Object)
        obj->fields.insert(state->fields.begin(), state->fields.end());
      else if (state->kind == Value::Sequence)
        for (auto &d : state->items)
          obj->fields.insert(d->fields.begin(), d->fields.end());
      else
        fail("unsupported object state");
      break;
    }
    default:
      fail("unsupported pickle opcode " + std::to_string(op) + " at " +
           std::to_string(pos - 1));
    }
  }
  fail("pickle has no STOP");
}
Ptr attribute(const Ptr &p, const std::string &name) {
  auto it = p->fields.find(name);
  if (it != p->fields.end())
    return it->second;
  for (const auto &k : {"_parameters", "_buffers", "_modules"}) {
    auto d = p->fields.find(k);
    if (d != p->fields.end()) {
      auto x = d->second->fields.find(name);
      if (x != d->second->fields.end())
        return x->second;
    }
  }
  fail("missing module attribute " + name + " on " + p->text);
}
Ptr module(const Ptr &p, const std::string &path) {
  auto out = p;
  std::istringstream in(path);
  std::string s;
  while (std::getline(in, s, '.'))
    out = attribute(out, s);
  return out;
}
std::map<std::string, Tensor> state_dict(const Ptr &p) {
  std::map<std::string, Tensor> out;
  std::set<const Value *> active;
  std::function<void(Ptr, std::string, int)> visit =
      [&](Ptr obj, std::string prefix, int depth) {
        if (depth > 100 || !active.insert(obj.get()).second)
          fail("cyclic module graph");
        for (auto &[k, v] : obj->fields) {
          if (k == "_parameters" || k == "_buffers")
            for (auto &[n, t] : v->fields)
              if (t->kind == Value::TensorValue)
                out[prefix + n] = t->tensor;
          if (k == "_modules")
            for (auto &[n, m] : v->fields)
              visit(m, prefix + n + ".", depth + 1);
          if (v->kind == Value::TensorValue)
            out[prefix + k] = v->tensor;
          if (v->kind == Value::Object && v->text.rfind("__torch__.", 0) == 0)
            visit(v, prefix + k + ".", depth + 1);
        }
        active.erase(obj.get());
      };
  visit(p, "", 0);
  return out;
}
} // namespace checkpoint
