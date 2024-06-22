add_executable(testlib test_kernels.cpp)
target_compile_features(testlib PRIVATE cxx_std_17)
target_link_libraries(testlib PRIVATE squashlib)
add_test(NAME testlibtest COMMAND testlib)
