#!/usr/bin/env python3
"""Local hook commands. Requires Python 3.10+; never rewrites source files."""

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
FORMAT_VERSION = "23.1.4"
CPP_SUFFIXES = {".c", ".cc", ".cpp", ".cxx", ".h", ".hh", ".hpp", ".hxx", ".cu", ".cuh"}
CONFLICT = re.compile(r"^(<{7}|>{7}|\|{7})( |$)|^={7}$")


def run(command):
    print("+ " + " ".join(map(str, command)), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def source(paths):
    errors = []
    for path in paths:
        data = path.read_bytes()
        # --all also accepts tracked binary assets; pre-commit filters these out.
        if b"\0" in data:
            continue
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            errors.append(f"{path}: expected UTF-8 text")
            continue
        if data and not data.endswith(b"\n"):
            errors.append(f"{path}: missing final newline")
        for number, line in enumerate(text.splitlines(), 1):
            # Two spaces intentionally create a hard line break in Markdown.
            markdown_break = (
                path.suffix == ".md" and line.endswith("  ")
                and bool(line[:-2]) and not line[:-2].endswith((" ", "\t"))
            )
            if line.endswith((" ", "\t")) and not markdown_break:
                errors.append(f"{path}:{number}: trailing whitespace")
            if CONFLICT.match(line):
                errors.append(f"{path}:{number}: merge conflict marker")
        try:
            if path.suffix == ".py":
                compile(data, str(path), "exec")
            elif path.suffix == ".json":
                json.loads(text)
        except (SyntaxError, ValueError) as error:
            errors.append(f"{path}: {error}")
    if errors:
        print("\n".join(errors), file=sys.stderr)
    return bool(errors)


def tracked_files():
    names = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT)
    return [Path(os.fsdecode(name)) for name in names.split(b"\0") if name]


def formatting(paths):
    executable = os.environ.get("CLANG_FORMAT", "clang-format")
    version = subprocess.check_output([executable, "--version"], text=True)
    if not re.search(r"\bversion " + re.escape(FORMAT_VERSION) + r"\b", version):
        raise ValueError(f"Expected clang-format {FORMAT_VERSION}; found {version.strip()}")
    # A style change must also validate files not otherwise included in the hook.
    if Path(".clang-format") in paths:
        paths = tracked_files()
    paths = [path for path in paths if path.suffix in CPP_SUFFIXES and path.is_file()]
    # Bound command length even when checking the whole repository.
    for offset in range(0, len(paths), 50):
        run([executable, "--dry-run", "--Werror", "--style=file", *map(str, paths[offset:offset + 50])])
    return 0


def full(build, jobs):
    run([
        "cmake", "-S", ".", "-B", str(build),
        "-DCMAKE_BUILD_TYPE=Release", "-DBUILD_TESTING=ON",
        "-DLAPIS_BUILD_MACE=ON", "-DLAPIS_BUILD_HPNN=ON", "-DLAPIS_BUILD_D3=ON",
        "-DLAPIS_TEST_DISCOVER_CASES=OFF",
        f"-DPython3_EXECUTABLE={sys.executable}",
    ])
    run(["cmake", "--build", str(build), "--config", "Release", "--parallel", str(jobs)])
    run([
        "ctest", "--test-dir", str(build), "-C", "Release",
        "--parallel", str(jobs), "--output-on-failure", "--no-tests=error", "--timeout", "300",
    ])
    return 0


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("check", choices=("source", "format", "full"))
    parser.add_argument("--all", action="store_true", help="check all tracked files")
    parser.add_argument("--build-dir", type=Path, default=Path("build"),
                        help="full-check build directory (default: build)")
    parser.add_argument("--jobs", type=positive_int, default=4,
                        help="full-check build/CTest parallelism (default: 4)")
    parser.add_argument("files", nargs="*")
    args = parser.parse_args()
    os.chdir(ROOT)
    try:
        if args.check == "full":
            return full(args.build_dir, args.jobs)
        paths = tracked_files() if args.all else [Path(name) for name in args.files]
        paths = [path for path in paths if path.is_file() and not path.is_symlink()]
        return source(paths) if args.check == "source" else formatting(paths)
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        print(f"Check failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
