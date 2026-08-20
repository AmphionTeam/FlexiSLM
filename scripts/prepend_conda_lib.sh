#!/usr/bin/env bash
# Prepend the active conda/Python lib directory to LD_LIBRARY_PATH so
# torchcodec can load FFmpeg/OpenVINO against conda's libstdc++
# (CXXABI_1.3.15) instead of the older system libstdc++.
#
# Source this in the launching shell before any Python process starts.
# Setting LD_LIBRARY_PATH from inside Python is too late.

_flexislm_conda_lib=""
if [ -n "${CONDA_PREFIX:-}" ] && [ -d "${CONDA_PREFIX}/lib" ]; then
    _flexislm_conda_lib="${CONDA_PREFIX}/lib"
else
    _flexislm_python="$(command -v python 2>/dev/null || true)"
    if [ -n "${_flexislm_python}" ]; then
        _flexislm_conda_lib="$(cd "$(dirname "${_flexislm_python}")/../lib" && pwd 2>/dev/null || true)"
    fi
    unset _flexislm_python
fi

if [ -n "${_flexislm_conda_lib}" ] && [ -d "${_flexislm_conda_lib}" ]; then
    case ":${LD_LIBRARY_PATH:-}:" in
        *":${_flexislm_conda_lib}:"*) ;;
        *)
            export LD_LIBRARY_PATH="${_flexislm_conda_lib}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
            ;;
    esac
    echo "INFO: LD_LIBRARY_PATH prepends ${_flexislm_conda_lib} (conda libstdc++ for torchcodec/FFmpeg)"
else
    echo "Warning: could not resolve conda/Python lib dir for LD_LIBRARY_PATH" >&2
fi
unset _flexislm_conda_lib
