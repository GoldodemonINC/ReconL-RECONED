#!/usr/bin/env bash
# Build and run the C# test host against the shipped reconl DLL.
#
# Prerequisite: the Rust FFI crate has been built (cargo build -p reconl-ffi).
# The C# wrapper P/Invokes the C ABI directly via DllImport.
#
# Usage:
#   cs/build.sh              # build and run the test host
#   cs/build.sh run          # just run (assumes built)
#   cs/build.sh clean        # remove build output

set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
cs_root="$root/cs"
build_dir="$cs_root/build"

# Detect platform
case "$(uname -s)" in
    MINGW*|MSYS*|CYGWIN*)
        lib_ext=.dll
        lib_name=reconl.dll
        ;;
    Darwin)
        lib_ext=.dylib
        lib_name=libreconl.dylib
        ;;
    *)
        lib_ext=.so
        lib_name=libreconl.so
        ;;
esac

lib_dir="$root/target/release"
library="$lib_dir/$lib_name"

# If release not available, fall back to debug
if [ ! -f "$library" ] && [ -f "$root/target/debug/$lib_name" ]; then
    lib_dir="$root/target/debug"
    library="$lib_dir/$lib_name"
fi

usage() {
    echo "Usage: $0 [run|clean]"
    echo "  run   — build and run the test host (default)"
    echo "  clean — remove build output"
    exit 0
}

do_clean() {
    rm -rf "$build_dir"
    echo "cleaned $build_dir"
}

do_build_run() {
    mkdir -p "$build_dir"

    if [ ! -f "$library" ]; then
        echo "ERROR: $library not found. Build the Rust FFI crate first:"
        echo "  cargo build -p reconl-ffi --release"
        exit 1
    fi

    # Copy the library into the build dir so DllImport finds it
    cp "$library" "$build_dir/$lib_name"
    echo "== building C# test host =="
    echo "  copied $lib_name to $build_dir"

    cd "$cs_root"
    dotnet new console -n TestHost -o "$build_dir" --force 2>/dev/null || true
    # Copy our Program.cs and ReconL.cs into the project
    cp "$cs_root/src/Program.cs" "$build_dir/Program.cs"
    cp "$cs_root/include/ReconL.cs" "$build_dir/ReconL.cs"

    # Set the library path for DllImport — on Windows the DLL must be beside the exe
    # or in the PATH. We run from the build dir so it resolves.
    cd "$build_dir"
    # Enable unsafe code for the P/Invoke wrapper
    if ! grep -q 'AllowUnsafeBlocks' "$build_dir/$lib_name""" 2>/dev/null; then
        sed -i 's/<PropertyGroup>/<PropertyGroup>\r\n    <AllowUnsafeBlocks>true<\/AllowUnsafeBlocks>/' "$build_dir/TestHost.csproj"
    fi
    dotnet build -c Release --nologo 2>&1 | tail -5
    echo "  built: $build_dir/bin/Release/net*"

    echo ""
    echo "== running C# test host =="
    dotnet run -c Release --no-build 2>&1
}

# Main
case "${1:-run}" in
    run)   do_build_run ;;
    clean) do_clean ;;
    *)     usage ;;
esac
