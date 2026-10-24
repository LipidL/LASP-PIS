#!/usr/bin/env python3
"""Build isolated consumers and C/C++/CUDA clients against relocated installations."""
import argparse
from pathlib import Path
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def run(*args):
    subprocess.run([str(a) for a in args], check=True)


def check(folder, arch, jobs):
    prefix = folder / 'original-prefix'
    relocated = folder / 'relocated-prefix'
    for project in ('lapis', 'chemio', 'checkpoint', 'mace'):
        source = folder / project
        source.mkdir()
        for name in ('CMakeLists.txt', 'cmake', 'include', 'src'):
            origin = ROOT / project / name
            if origin.is_dir():
                shutil.copytree(origin, source / name)
            else:
                shutil.copy2(origin, source / name)
        build = folder / ('build-' + project)
        options = ['-DBUILD_TESTING=OFF', '-DCMAKE_BUILD_TYPE=Release',
                   '-DCMAKE_CUDA_ARCHITECTURES=' + arch,
                   '-DCMAKE_PREFIX_PATH=' + str(relocated)]
        if project == 'lapis':
            options += ['-DLAPIS_BUILD_EXAMPLES=OFF', '-DLAPIS_BUILD_EXPLICIT_REFERENCE=ON']
        if project == 'mace':
            options += ['-DMACE_BUILD_EXPLICIT_REFERENCE=ON']
        run('cmake', '-S', source, '-B', build, *options)
        run('cmake', '--build', build, '-j', jobs)
        run('cmake', '--install', build, '--prefix', prefix if project == 'lapis' else relocated)
        if project == 'lapis':
            prefix.rename(relocated)
            # Consumers see only the installation, not the source/runtime build.
            source.rename(folder / 'framework-source-not-in-search-path')
        if project == 'mace':
            run(relocated / 'bin' / (project + '-main'), '--help')
    client = folder / 'client'
    client.mkdir()
    (client / 'CMakeLists.txt').write_text("""cmake_minimum_required(VERSION 3.24)
project(package_client LANGUAGES C CXX CUDA)
find_package(mace CONFIG REQUIRED)
add_executable(c_client main.c)
set_target_properties(c_client PROPERTIES LINKER_LANGUAGE CXX)
target_link_libraries(c_client PRIVATE mace::mace)
add_executable(cpp_client main.cpp)
target_link_libraries(cpp_client PRIVATE mace::mace)
add_library(cuda_client STATIC extension.cu)
target_link_libraries(cuda_client PRIVATE lapis::lapis)
""")
    (client / 'main.c').write_text("""#include <mace/c_api.h>
int main(void) { mace_destroy(NULL); return 0; }
""")
    (client / 'main.cpp').write_text("""#include <mace/mh1.hpp>
int main(int argc, char** argv) {
  if (argc == 2) {
    mace::MH1 m(argv[1]);
    lapis::Structure s; s.cell = {10,0,0,0,10,0,0,0,10};
    m.calculate(s);
  }
}
""")
    (client / 'extension.cu').write_text("""#include <lapis/pair.cuh>
struct Op {
  static constexpr int threads=32, tile_edges=4, scratch_alignment=16;
  static const char* name() { return "external"; }
  struct State {};
  lapis::WorkspaceSpec plan(int) const { return {}; }
  void setup(lapis::Context&, void*) {}
  void postprocess(lapis::Context&, void*) {}
  __device__ void prepare(int, State&, void*) {}
  template<class Tile> __device__ void consume(int, const Tile&, State&, void*) {}
  __device__ void finalize(int, State&, void*) {}
};
void launch(lapis::Context& ctx) { lapis::execute(ctx, Op{}); }
""")
    run('cmake', '-S', client, '-B', folder / 'build-client',
        '-DCMAKE_PREFIX_PATH=' + str(relocated), '-DCMAKE_CUDA_ARCHITECTURES=' + arch)
    run('cmake', '--build', folder / 'build-client', '-j', jobs)
    run(folder / 'build-client/c_client')
    run(folder / 'build-client/cpp_client')
    print('Installed Lapis, Chemio, checkpoint, isolated MACE library/executable, and C/C++/CUDA clients passed.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cuda-architectures', default='86')
    parser.add_argument('--jobs', default='4')
    parser.add_argument('--work-dir', type=Path, help='New directory in which to retain builds')
    args = parser.parse_args()
    if args.work_dir:
        args.work_dir.mkdir(parents=True, exist_ok=False)
        check(args.work_dir.resolve(), args.cuda_architectures, args.jobs)
    else:
        with tempfile.TemporaryDirectory(prefix='lapis-packages-') as directory:
            check(Path(directory), args.cuda_architectures, args.jobs)
