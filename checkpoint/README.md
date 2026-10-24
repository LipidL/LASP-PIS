# Native checkpoint data reader

Independent C++17 package used privately by MACE. It uses a ZIP64 reader
with zlib decompression and a bounded data-only pickle decoder. Python class and reducer
names become inert records: Python code and TorchScript graphs are never run.
It has no LibTorch or Python runtime dependency. Lapis and Chemio have no
dependency on this package.

```sh
cmake -S . -B build
cmake --build build
ctest --test-dir build
cmake --install build --prefix /path/to/install
```

Consumers use `find_package(checkpoint 0.1.0 EXACT CONFIG REQUIRED)` and
`checkpoint::checkpoint`. zlib is the only external dependency. This avoids
loading a second, potentially incompatible LibTorch into Python frontend
processes. Stored and deflated records, ZIP64 metadata, and CRC checks are
supported. Encrypted and split ZIP archives are rejected.

The decoder supports the pickle operations and dense float32/float64/integer
tensor storage used by the supplied eager MH1 checkpoint, including storage
offsets, strides, memo references, and embedded byte strings. This is not a general
Python unpickler or a universal PyTorch model
loader. Model projects must validate architecture and interpret metadata.
Unsupported opcodes, storage types, byte orders, invalid references, nonfinite
tensor data, and out-of-bounds tensor views are errors. Resource limits apply to
pickle size, operations, records, and tensor storage.

`checkpoint_inspect` is a development-only tensor/metadata inspection utility.
