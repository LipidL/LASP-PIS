# Chemio

Independent C++17 structure I/O. No CUDA, Lapis, model, or Python dependencies.
`chemio::read(path)` reads all ARC frames; `read_frame(path, index)` selects a
zero-based frame. Structures own double precision Cartesian Angstrom coordinates,
atomic numbers, row-vector cells, periodicity, and the original frame title.

The initial reader supports BIOSYM/LASP `.arc` files with `CORE` atom records,
optional titles and `!DATE` records, six cell parameters (angles in degrees), and
two `end` markers per frame. Unknown elements, malformed records, impossible
cells, and incomplete frames are errors with file and line context. Reference
energies in titles are metadata, never inference results. Additional formats can
implement the same owned structure interface.

Build with `cmake -S . -B build`, then `cmake --build build` and
`ctest --test-dir build`. Install and consume with `find_package(chemio CONFIG
REQUIRED)` and `target_link_libraries(client PRIVATE chemio::chemio)`.

The ARC layout was informed by the Rust chemkit parser. Cell matrices follow the
row-vector convention explicitly; malformed atom lines are not skipped.
