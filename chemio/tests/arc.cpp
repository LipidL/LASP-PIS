#include <chemio/structure.hpp>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>

void require(bool ok) { if (!ok) throw std::runtime_error("ARC test failed"); }
int main() {
  const auto path=std::filesystem::temp_directory_path()/"chemio-test.arc";
  const std::string frame="Energy 0 0 -1\n!DATE\nPBC 21.9552 15.2646 6.877 91.8679 68.0603 72.9361\nTi 1 2 3 CORE 1 X X 0 1\nO -1 0 2 CORE 1 X X 0 2\nend\nend\n";
  auto write=[&](const std::string& s){ std::ofstream f(path); f<<s; };
  try {
    write("!BIOSYM archive 2\nPBC=ON\n"+frame+frame);
    auto s=chemio::read(path.string()); require(s.size()==2 && s[0].elements==std::vector<int>({22,8}));
    require(chemio::read_frame(path.string(),1).positions[3]==-1);
    auto c=s[0].cell;
    auto dot=[&](int i,int j){ double v=0; for(int k=0;k<3;++k)v+=c[i*3+k]*c[j*3+k]; return v; };
    require(std::abs(std::sqrt(dot(2,2))-6.877)<1e-12);
    require(std::abs(std::acos(dot(1,2)/std::sqrt(dot(1,1)*dot(2,2)))*180/std::acos(-1.)-91.8679)<1e-10);
    for (const auto& bad : {std::string("PBC=ON\nPBC 1 1 1 90 90 90\nO x 0 0 CORE 1 X X 0 1\nend\nend\n"),
                            std::string("PBC=ON\nPBC 1 1 1 90 90 90\nXx 0 0 0 CORE 1 X X 0 1\nend\nend\n"),
                            std::string("PBC=ON\nPBC 1 1 1 90 90 90\nO 0 0 0 CORE 1 X X 0 1\nend\n"),
                            std::string("PBC=ON\nPBC 1 1 1 1 1 179\n")}) {
      write(bad); bool failed=false; try { chemio::read(path.string()); } catch(const std::exception&) { failed=true; } require(failed);
    }
    std::filesystem::remove(path);
    std::cout<<"ARC geometry, multiple frames, and invalid input passed\n";
  } catch (...) { std::filesystem::remove(path); throw; }
}
