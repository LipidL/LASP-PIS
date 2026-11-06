Add a fully periodic `.arc` file here and rebuild. Every frame automatically
runs every HPNN and MACE physics test, including all atomic force components
and all nine strain derivatives. Use unique alphanumeric/underscore filenames;
the filename and frame index become the GoogleTest parameter name.

Keep fixtures small: force differences cost four evaluations per coordinate,
and supercell tests also evaluate eight copies. Structures must contain elements
supported by both configured models and represent ordinary bound configurations
with negative total energy for those models. Highly compressed or arbitrary
random configurations need not have negative energies.

Built-in fixtures in `../physics.cpp` additionally exercise empty cells, isolated
atoms, molecules in vacuum, crystals, a slab, and skewed periodic cells.
