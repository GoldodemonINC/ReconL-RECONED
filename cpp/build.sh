#!/usr/bin/env bash
# Build the C++ wrapper and test host against the shipped reconl DLL.
#
# Prerequisite: the Rust FFI crate has been built (cargo build -p reconl-ffi).
# The wrapper is header-only (reconl_cpp.hpp) and the test host compiles against
# the C ABI header (include/reconl/reconl.h) and links against the DLL/SO.
#
# Usage:
#   cpp/build.sh              # build both
#   cpp/build.sh host         # just the test host
#   cpp/build.sh clean        # remove build output

set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
cpp_root="$root/cpp"
build_dir="$cpp_root/build"
include_dir="$root/include"

cc="${CXX:-g++}"
cxxstd="${CXXSTD:-c++17}"
warning_flags="-Wall -Wextra -Wpedantic -Wconversion -Wshadow"

# Detect platform
case "$(uname -s)" in
    MINGW*|MSYS*|CYGWIN*)
        lib_ext=.dll
        lib_name=reconl.dll
        exe_ext=.exe
        ;;
    Darwin)
        lib_ext=.dylib
        lib_name=libreconl.dylib
        exe_ext=
        ;;
    *)
        lib_ext=.so
        lib_name=libreconl.so
        exe_ext=
        ;;
esac

lib_dir="$root/target/release"
library="$lib_dir/$lib_name"

# If release not available, fall back to debug
if [ ! -f "$library" ] && [ -f "$root/target/debug/$lib_name" ]; then
    lib_dir="$root/target/debug"
    library="$lib_dir/$lib_name"
fi

link_flags="-L\"$lib_dir\" -lreconl -Wl,-rpath,\"$lib_dir\""

usage() {
    echo "Usage: $0 [host|clean]"
    echo "  host  — build the test host (default)"
    echo "  clean — remove build output"
    exit 0
}

do_clean() {
    rm -rf "$build_dir"
    echo "cleaned $build_dir"
}

do_build_host() {
    mkdir -p "$build_dir"

    if [ ! -f "$library" ]; then
        echo "ERROR: $library not found. Build the Rust FFI crate first:"
        echo "  cargo build -p reconl-ffi --release"
        exit 1
    fi

    echo "== building C++ test host =="
    "$cc" $warning_flags -std=$cxxstd -O2 \
        -I"$include_dir/reconl" \
        -I"$cpp_root/include" \
        -o "$build_dir/test_host$exe_ext" \
        "$cpp_root/src/main.cpp" \
        $link_flags

    echo "built: $build_dir/test_host$exe_ext"
    echo "  (links against $library)"

    # If the library is not on the RPATH, copy it beside the executable for testing
    if [ ! -f "$build_dir/$lib_name" ]; then
        cp "$library" "$build_dir/$lib_name"
        echo "  copied $lib_name beside the executable for testing"
    fi
}

# Main
case "${1:-host}" in
    host)  do_build_host ;;
    clean) do_clean ;;
    *)     usage ;;
esac
