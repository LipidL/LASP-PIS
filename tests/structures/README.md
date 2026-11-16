Add a fully periodic `.arc` file here and rebuild. Every frame automatically
runs every MACE physics test, including all atomic force components
and all nine strain derivatives. Use unique alphanumeric/underscore filenames;
the filename and frame index become the GoogleTest parameter name.

Keep fixtures small: force differences cost four evaluations per coordinate,
and supercell tests also evaluate eight copies. Structures must contain elements
supported by the configured MACE model and represent ordinary bound configurations
with negative total energy for that model. Highly compressed or arbitrary
random configurations need not have negative energies.

Built-in fixtures in `../physics.cpp` additionally exercise empty cells, isolated
atoms, molecules in vacuum, crystals, a slab, and skewed periodic cells.
