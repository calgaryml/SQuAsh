export ROCM_PATH="/opt/rocm/"
export CXX="$ROCM_PATH/bin/hipcc"
cmake -S . -B build
export CTEST_OUTPUT_ON_FAILURE=1
cmake --build build --clean-first
cmake --build build --target test