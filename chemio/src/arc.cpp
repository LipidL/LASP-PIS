#include <chemio/structure.hpp>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace chemio {
namespace {
const std::vector<std::string> symbols = [] {
  std::istringstream in("X H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr Rf Db Sg Bh Hs Mt Ds Rg Cn Nh Fl Mc Lv Ts Og");
  std::vector<std::string> out; std::string s;
  while (in >> s) out.push_back(s);
  return out;
}();
double number(const std::string& s) {
  std::size_t n; double x = std::stod(s, &n);
  if (n != s.size() || !std::isfinite(x)) throw std::runtime_error("invalid number: " + s);
  return x;
}
}

std::array<double, 9> cell_from_parameters(double a, double b, double c,
                                         double alpha, double beta, double gamma) {
  for (double x : {a,b,c,alpha,beta,gamma})
    if (!std::isfinite(x) || x <= 0) throw std::runtime_error("invalid cell parameters");
  if (alpha >= 180 || beta >= 180 || gamma >= 180) throw std::runtime_error("invalid cell angles");
  constexpr double rad = 0.017453292519943295769;
  double ca=std::cos(alpha*rad), cb=std::cos(beta*rad), cg=std::cos(gamma*rad), sg=std::sin(gamma*rad);
  double cx=c*cb, cy=c*(ca-cb*cg)/sg, cz2=c*c-cx*cx-cy*cy;
  if (sg < 1e-12 || cz2 <= c*c*1e-14) throw std::runtime_error("degenerate or impossible cell");
  return {a,0,0,b*cg,b*sg,0,cx,cy,std::sqrt(cz2)};
}

std::vector<Structure> read(const std::string& path) {
  auto dot=path.find_last_of('.'); std::string ext=dot==std::string::npos ? "" : path.substr(dot);
  std::transform(ext.begin(),ext.end(),ext.begin(),[](unsigned char c){return char(std::tolower(c));});
  if (ext != ".arc") throw std::runtime_error("unsupported structure format: " + ext + " (supported: .arc)");
  std::ifstream in(path);
  if (!in) throw std::runtime_error("cannot open structure: " + path);
  std::vector<Structure> frames; Structure current;
  bool periodic=false, have_cell=false, first_end=false, pending=false;
  std::string line; std::size_t lineno=0;
  while (std::getline(in,line)) {
    ++lineno;
    try {
      std::istringstream words(line); std::vector<std::string> f; std::string s;
      while (words >> s) f.push_back(s);
      if (f.empty()) continue;
      if (f[0][0]=='!') continue;
      if (f[0]=="PBC=ON" || f[0]=="PBC=OFF") {
        if (pending || have_cell) throw std::runtime_error("periodicity header inside frame");
        periodic=f[0]=="PBC=ON"; continue;
      }
      if (f[0]=="end" || f[0]=="END") {
        if (current.elements.empty()) throw std::runtime_error("empty frame");
        if (!first_end) { first_end=true; continue; }
        current.pbc={periodic,periodic,periodic};
        if (periodic && !have_cell) throw std::runtime_error("periodic frame has no cell");
        frames.push_back(std::move(current)); current=Structure{};
        have_cell=false; first_end=false; pending=false; continue;
      }
      if (first_end) throw std::runtime_error("expected second end marker");
      if (f[0]=="PBC") {
        if (have_cell || !current.elements.empty() || f.size()<7) throw std::runtime_error("invalid PBC record");
        current.cell=cell_from_parameters(number(f[1]),number(f[2]),number(f[3]),number(f[4]),number(f[5]),number(f[6]));
        have_cell=true; pending=true; continue;
      }
      if (f.size()>=5 && f[4]=="CORE") {
        if (periodic && !have_cell) throw std::runtime_error("atom before cell");
        auto it=std::find(symbols.begin()+1,symbols.end(),f[0]);
        if (it==symbols.end()) throw std::runtime_error("unknown element: " + f[0]);
        current.elements.push_back(int(it-symbols.begin()));
        for (int k=1;k<=3;++k) current.positions.push_back(number(f[k]));
        pending=true; continue;
      }
      if (have_cell || !current.elements.empty()) throw std::runtime_error("unrecognized atom record");
      if (pending) throw std::runtime_error("unexpected frame title");
      current.title=line; pending=true;
    } catch (const std::exception& e) {
      throw std::runtime_error(path+":"+std::to_string(lineno)+": "+e.what());
    }
  }
  if (in.bad()) throw std::runtime_error("error reading structure: " + path);
  if (pending || have_cell || first_end) throw std::runtime_error(path+": incomplete ARC frame");
  if (frames.empty()) throw std::runtime_error(path+": no structures");
  return frames;
}
Structure read_frame(const std::string& path, std::size_t frame) {
  auto frames=read(path);
  if (frame>=frames.size()) throw std::runtime_error("frame index out of range: " + std::to_string(frame));
  return std::move(frames[frame]);
}
}
