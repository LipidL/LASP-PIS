#include "weights.hpp"
#include <algorithm>
#include <checkpoint/checkpoint.hpp>
#include <cmath>
#include <cstring>
#include <fstream>
#include <map>
#include <numeric>
#include <stdexcept>

namespace mace_detail {
namespace {
using checkpoint::Ptr;
void require(bool ok, const std::string &s) {
  if (!ok)
    throw std::runtime_error("unsupported MH1 checkpoint: " + s);
}
Ptr at(const Ptr &p, const std::string &s) {
  return checkpoint::module(p, s);
}
checkpoint::Tensor tensor(const Ptr &p, const std::string &s) {
  auto v = at(p, s);
  require(v->kind == checkpoint::Value::TensorValue, "expected tensor " + s);
  return v->tensor;
}
std::vector<double> data(const Ptr &p, const std::string &s) {
  return tensor(p, s).doubles();
}
double scalar(const Ptr &p, const std::string &s) {
  auto v = at(p, s);
  if (v->kind == checkpoint::Value::Number)
    return v->number;
  auto d = v->tensor.doubles();
  require(d.size() == 1, "expected scalar " + s);
  return d[0];
}
struct Irrep {
  int mul, l, parity, offset, component;
};
std::vector<Irrep> irreps(Ptr p) {
  std::vector<Irrep> out;
  int offset = 0, component = 0;
  require(p->items.size() <= 64, "too many irreps");
  for (auto x : p->items) {
    if (x->items.size() == 1)
      x = x->at(0);
    int mul = int(x->at(0)->integer()), l = int(x->at(1)->at(0)->integer()),
        parity = int(x->at(1)->at(1)->integer());
    require(mul > 0 && mul <= 4096 && l >= 0 && l <= 3 && std::abs(parity) == 1, "irreps");
    out.push_back({mul, l, parity, offset, component});
    offset += mul * (2 * l + 1);
    component += 2 * l + 1;
  }
  require(!out.empty(), "empty irreps");
  return out;
}
int dimension(const std::vector<Irrep> &r) {
  const auto &x = r.back();
  return x.offset + x.mul * (2 * x.l + 1);
}
template <class T> void put(std::string &bytes, T x) {
  bytes.append(reinterpret_cast<const char *>(&x), sizeof(x));
}
struct Weights {
  std::map<std::string, std::string> records;
  void floats(const std::string &name, const std::vector<double> &d) {
    std::string bytes;
    bytes.reserve(d.size() * 4);
    for (double x : d) {
      require(std::isfinite(x) && std::isfinite(float(x)), "nonfinite " + name);
      put(bytes, float(x));
    }
    require(records.emplace(name, std::move(bytes)).second, "duplicate " + name);
  }
  void ints(const std::string &name, const std::vector<int> &d) {
    std::string bytes;
    for (int x : d)
      put(bytes, int32_t(x));
    require(records.emplace(name, std::move(bytes)).second, "duplicate " + name);
  }
  std::string encode() const {
    std::string bytes = "PWMH1001";
    put(bytes, uint32_t(records.size()));
    for (auto &[name, d] : records) {
      put(bytes, uint32_t(name.size()));
      put(bytes, uint64_t(d.size()));
      bytes += name;
      bytes += d;
    }
    return bytes;
  }
};
void linear(Weights &ar, const std::string &name, const Ptr &m, std::vector<int> select_in = {},
            std::vector<int> select_out = {}) {
  auto ins = irreps(at(m, "irreps_in")), outs = irreps(at(m, "irreps_out"));
  auto weights = data(m, "weight");
  std::vector<int> paths;
  std::vector<double> packed;
  size_t offset = 0;
  auto instructions = at(m, "instructions");
  require(instructions->items.size() <= 128, "linear path count");
  for (auto inst : instructions->items) {
    int i = int(inst->at(0)->integer()), o = int(inst->at(1)->integer());
    require(i >= 0 && i < int(ins.size()) && o >= 0 && o < int(outs.size()),
            "linear instruction indices");
    int ni = ins[i].mul, no = outs[o].mul;
    require(ins[i].l == outs[o].l && ins[i].parity == outs[o].parity, "linear irreps mismatch");
    auto shape = inst->at(2);
    require(shape->items.size() == 2 && shape->at(0)->integer() == ni &&
                shape->at(1)->integer() == no,
            "linear path shape");
    require(offset + size_t(ni) * no <= weights.size(), "linear weight count");
    auto si = select_in, so = select_out;
    if (si.empty()) {
      si.resize(ni);
      std::iota(si.begin(), si.end(), 0);
    } else
      require(ins.size() == 1, "input head selection");
    if (so.empty()) {
      so.resize(no);
      std::iota(so.begin(), so.end(), 0);
    } else
      require(outs.size() == 1, "output head selection");
    paths.insert(paths.end(),
                 {select_in.empty() ? ins[i].offset : 0, select_out.empty() ? outs[o].offset : 0,
                  int(si.size()), int(so.size()), 2 * ins[i].l + 1, int(packed.size())});
    double scale = inst->at(3)->number;
    for (int a : si)
      for (int b : so) {
        require(a >= 0 && a < ni && b >= 0 && b < no, "head index");
        packed.push_back(weights[offset + size_t(a) * no + b] * scale);
      }
    offset += size_t(ni) * no;
  }
  require(offset == weights.size(), "unconsumed linear weights");
  ar.ints(name + ".paths", paths);
  ar.ints(name + ".dims", {select_in.empty() ? dimension(ins) : int(select_in.size()),
                           select_out.empty() ? dimension(outs) : int(select_out.size())});
  ar.floats(name + ".weight", packed);
}
std::vector<double> embedding(const Ptr &m, int species) {
  auto ins = irreps(at(m, "irreps_in")), outs = irreps(at(m, "irreps_out"));
  auto inst = at(m, "instructions");
  require(ins.size() == 1 && outs.size() == 1 && ins[0].l == 0 && outs[0].l == 0 &&
              ins[0].mul == species && outs[0].mul == 512 && inst->items.size() == 1,
          "embedding architecture");
  auto w = data(m, "weight");
  require(w.size() == size_t(species) * 512, "embedding weights");
  auto instruction = inst->at(0);
  require(instruction->at(0)->integer() == 0 && instruction->at(1)->integer() == 0,
          "embedding path");
  for (auto &x : w)
    x *= instruction->at(3)->number;
  return w;
}
void radial(Weights &ar, const std::string &name, const Ptr &m, const std::vector<double> &source,
            const std::vector<double> &target, int species, int depth, int output) {
  auto net = at(m, "net");
  auto w = data(net, "0.weight");
  require(tensor(net, "0.weight").shape == std::vector<int64_t>({64, 1034}),
          "radial input dimensions");
  std::vector<double> s(species * 64), t(species * 64), r(64 * 10);
  for (int z = 0; z < species; ++z)
    for (int o = 0; o < 64; ++o)
      for (int i = 0; i < 512; ++i) {
        s[z * 64 + o] += source[z * 512 + i] * w[o * 1034 + 10 + i];
        t[z * 64 + o] += target[z * 512 + i] * w[o * 1034 + 522 + i];
      }
  for (int o = 0; o < 64; ++o)
    for (int i = 0; i < 10; ++i)
      r[o * 10 + i] = w[o * 1034 + i];
  ar.floats(name + ".source", s);
  ar.floats(name + ".target", t);
  ar.floats(name + ".radial", r);
  auto children = net->get("_modules");
  require(children->fields.size() == size_t(depth * 3 + 1), "radial network depth");
  for (int k = 0; k <= depth * 3; ++k) {
    auto block = at(net, std::to_string(k));
    if (k % 3 == 2) {
      require(block->text == "torch.nn.modules.activation SiLU", "radial activation");
      continue;
    }
    if (k % 3 == 1)
      require(scalar(block, "eps") == 1e-5, "radial LayerNorm epsilon");
    else
      require(tensor(block, "weight").shape ==
                  std::vector<int64_t>({k == depth * 3 ? output : 64, k == 0 ? 1034 : 64}),
              "radial matrix shape");
    ar.floats(name + "." + std::to_string(k) + ".weight", data(block, "weight"));
    ar.floats(name + "." + std::to_string(k) + ".bias", data(block, "bias"));
  }
}
struct CG {
  int l1, l2, lo, a, b, c;
  double value;
};
const CG couplings[] = {
#include "cg.inc"
};
void tensor_product(Weights &ar, const std::string &name, const Ptr &m, int layer) {
  auto ins = irreps(at(m, "irreps_in1")), sh = irreps(at(m, "irreps_in2")),
       outs = irreps(at(m, "irreps_out"));
  auto instructions = at(m, "instructions");
  require(instructions->items.size() == size_t(layer ? 10 : 4), "tensor product path count");
  require(sh.size() == 4, "spherical harmonic irreps");
  for (int l = 0; l < 4; ++l)
    require(sh[l].l == l && sh[l].mul == 1 && sh[l].parity == (l % 2 ? -1 : 1),
            "spherical harmonic convention");
  std::vector<int> terms;
  std::vector<double> values;
  int path = 0;
  for (auto inst : instructions->items) {
    int i = int(inst->at(0)->integer()), j = int(inst->at(1)->integer()),
        o = int(inst->at(2)->integer());
    require(i >= 0 && i < int(ins.size()) && j >= 0 && j < int(sh.size()) && o >= 0 &&
                o < int(outs.size()),
            "tensor product instruction");
    require(inst->at(3)->text == "uvu" && inst->at(4)->integer() == 1 &&
                inst->at(6)->items.size() == 2 && inst->at(6)->at(0)->integer() == 128 &&
                inst->at(6)->at(1)->integer() == 1,
            "tensor product connectivity");
    require(ins[i].l <= 1 && ins[i].mul == 128 && outs[o].mul == 128,
            "tensor product multiplicity");
    bool found = false;
    for (const auto &cg : couplings)
      if (cg.l1 == ins[i].l && cg.l2 == sh[j].l && cg.lo == outs[o].l) {
        found = true;
        terms.insert(terms.end(), {path, ins[i].component + cg.a, sh[j].l * sh[j].l + cg.b,
                                   outs[o].component + cg.c});
        // Match NumPy float32 multiplication in the Python exporter.
        values.push_back(float(cg.value) * float(inst->at(5)->number));
      }
    require(found, "unavailable coupling coefficients");
    ++path;
  }
  ar.ints(name + ".terms", terms);
  ar.floats(name + ".values", values);
  ar.ints(name + ".dims", {path, ins.back().component + 2 * ins.back().l + 1,
                           outs.back().component + 2 * outs.back().l + 1});
}
void polynomial(Weights &ar, const std::string &name, const Ptr &product, int layer) {
  auto contractions = at(product, "symmetric_contractions.contractions")->get("_modules");
  require(contractions->fields.size() == size_t(layer ? 1 : 2), "product contraction count");
  std::vector<int> terms;
  std::vector<double> coefficients;
  int out_offset = 0;
  for (int c = 0; c < (layer ? 1 : 2); ++c) {
    auto contraction = at(product, "symmetric_contractions.contractions." + std::to_string(c));
    require(scalar(contraction, "correlation") == 3, "product correlation");
    int outputs = c == 0 ? 1 : 3;
    for (int degree = 1; degree <= 3; ++degree) {
      auto ut = tensor(contraction, "U_matrix_" + std::to_string(degree));
      auto u = ut.doubles();
      auto wt = tensor(contraction,
                       degree == 3 ? "weights_max" : "weights." + std::to_string(2 - degree));
      auto w = wt.doubles();
      require(wt.shape.size() == 3 && wt.shape[0] == 1 && wt.shape[2] == 512,
              "contraction weight shape");
      int paths = int(wt.shape[1]);
      std::vector<int64_t> expected;
      if (outputs > 1)
        expected.push_back(outputs);
      for (int d = 0; d < degree; ++d)
        expected.push_back(16);
      expected.push_back(paths);
      require(ut.shape == expected, "coupling tensor shape");
      for (int out = 0; out < outputs; ++out)
        for (int a = 0; a < 16; ++a)
          for (int b = (degree >= 2 ? a : 0); b < (degree >= 2 ? 16 : 1); ++b)
            for (int cc = (degree == 3 ? b : 0); cc < (degree == 3 ? 16 : 1); ++cc) {
              std::vector<int> indices{a};
              if (degree >= 2)
                indices.push_back(b);
              if (degree == 3)
                indices.push_back(cc);
              auto permutation = indices;
              std::vector<double> coupling(paths, 0), coeff(512, 0);
              do {
                size_t index = out;
                for (int x : permutation)
                  index = index * 16 + x;
                for (int p = 0; p < paths; ++p)
                  coupling[p] += u[index * paths + p];
              } while (std::next_permutation(permutation.begin(), permutation.end()));
              for (int p = 0; p < paths; ++p)
                if (coupling[p] != 0)
                  for (int channel = 0; channel < 512; ++channel)
                    coeff[channel] += coupling[p] * w[p * 512 + channel];
              double maximum = 0;
              for (auto x : coeff)
                maximum = std::max(maximum, std::abs(x));
              if (maximum < 1e-12)
                continue;
              terms.insert(terms.end(), {out_offset + out, degree, a, degree >= 2 ? b : -1,
                                         degree == 3 ? cc : -1});
              coefficients.insert(coefficients.end(), coeff.begin(), coeff.end());
            }
    }
    out_offset += outputs;
  }
  ar.ints(name + ".terms", terms);
  ar.floats(name + ".coefficients", coefficients);
}
} // namespace
std::string weight_data(const std::string &path, const std::string &requested_head) {
  if (!checkpoint::is_zip(path)) {
    require(requested_head.empty(),
            "head selection requires an original checkpoint; an archive already contains one head");
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    auto n = in.tellg();
    require(n >= 0 && uint64_t(n) < (uint64_t(1) << 31), "archive size");
    std::string bytes(size_t(n), '\0');
    in.seekg(0);
    if (!in.read(bytes.data(), bytes.size()))
      throw std::runtime_error("cannot read MH1 archive: " + path);
    return bytes;
  }
  checkpoint::Archive archive(path);
  auto root = archive.root();
  require(root->text == "mace.modules.models ScaleShiftMACE", "expected ScaleShiftMACE");
  auto heads = at(root, "heads");
  auto head_name = requested_head.empty() ? "matpes_r2scan" : requested_head;
  int head = -1;
  for (size_t i = 0; i < heads->items.size(); ++i)
    if (heads->at(i)->text == head_name)
      head = int(i);
  require(head >= 0, "unknown head: " + head_name);
  require(heads->items.size() == 6, "expected six-head MH1 architecture");
  require(at(root, "interactions")->get("_modules")->fields.size() == 2 &&
              at(root, "products")->get("_modules")->fields.size() == 2 &&
              at(root, "readouts")->get("_modules")->fields.size() == 2,
          "two-layer architecture");
  require(scalar(root, "r_max") == 6 && scalar(root, "radial_embedding.apply_cutoff") == 0,
          "radial cutoff");
  auto atomic_numbers = data(root, "atomic_numbers");
  int species = int(atomic_numbers.size());
  require(species > 0 && species <= 118, "element count");
  Weights ar;
  std::vector<int> zs;
  for (double z : atomic_numbers) {
    require(z == std::trunc(z) && z >= 1 && z <= 118, "atomic number");
    zs.push_back(int(z));
  }
  ar.ints("atomic_numbers", zs);
  ar.floats("embedding", embedding(at(root, "node_embedding.linear"), species));
  auto e0 = data(root, "atomic_energies_fn.atomic_energies");
  require(e0.size() == heads->items.size() * species, "atomic baseline shape");
  ar.floats("e0",
            std::vector<double>(e0.begin() + head * species, e0.begin() + (head + 1) * species));
  auto scales = data(root, "scale_shift.scale"), shifts = data(root, "scale_shift.shift");
  require(scales.size() == 6 && shifts.size() == 6, "scale-shift shape");
  auto re = at(root, "radial_embedding"), zbl = at(root, "pair_repulsion_fn");
  ar.floats("constants", {6, scalar(re, "bessel_fn.prefactor"), scalar(re, "cutoff_fn.p"),
                          scalar(re, "distance_transform.a"), scalar(re, "distance_transform.q"),
                          scalar(re, "distance_transform.p"), scales[head], shifts[head],
                          scalar(zbl, "a_exp"), scalar(zbl, "a_prefactor"), scalar(zbl, "p"),
                          scalar(root, "interactions.0.equivariant_nonlin.act_scalars.acts.0.cst"),
                          scalar(root, "interactions.0.equivariant_nonlin.act_gates.acts.0.cst"),
                          scalar(root, "readouts.1.non_linearity.acts.0.cst")});
  for (auto name : {"act_scalars", "act_gates"})
    require(
        scalar(root, std::string("interactions.0.equivariant_nonlin.") + name + ".acts.0.cst") ==
            scalar(root, std::string("interactions.1.equivariant_nonlin.") + name + ".acts.0.cst"),
        "different gate normalization");
  ar.floats("bessel", data(re, "bessel_fn.bessel_weights"));
  ar.floats("radii", data(re, "distance_transform.covalent_radii"));
  ar.floats("zbl.radii", data(zbl, "covalent_radii"));
  ar.floats("zbl.c", data(zbl, "c"));
  for (int layer = 0; layer < 2; ++layer) {
    auto block = at(root, "interactions." + std::to_string(layer));
    auto prefix = "layer" + std::to_string(layer);
    require(block->text == "mace.modules.blocks RealAgnosticResidualNonLinearInteractionBlock",
            "interaction class");
    for (auto name : {"linear_up", "linear_res", "linear_1", "linear_2", "skip_tp"})
      linear(ar, prefix + "." + name, at(block, name));
    ar.floats(prefix + ".norm", {scalar(block, "alpha"), scalar(block, "beta")});
    auto source = embedding(at(block, "source_embedding"), species),
         target = embedding(at(block, "target_embedding"), species);
    radial(ar, prefix + ".radial", at(block, "conv_tp_weights"), source, target, species, 3,
           (layer ? 10 : 4) * 128);
    radial(ar, prefix + ".density", at(block, "density_fn"), source, target, species, 1, 1);
    tensor_product(ar, prefix + ".tp", at(block, "conv_tp"), layer);
    auto product = at(root, "products." + std::to_string(layer));
    polynomial(ar, prefix + ".product", product, layer);
    linear(ar, prefix + ".product.linear", at(product, "linear"));
  }
  linear(ar, "readout0", at(root, "readouts.0.linear"), {}, {head});
  std::vector<int> selection(16);
  std::iota(selection.begin(), selection.end(), head * 16);
  linear(ar, "readout1.up", at(root, "readouts.1.linear_1"), {}, selection);
  linear(ar, "readout1.down", at(root, "readouts.1.linear_2"), selection, {head});
  return ar.encode();
}
} // namespace mace_detail
