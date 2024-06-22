set(HEADER_LIST "${SQuAsh_SOURCE_DIR}/include/kernels.hpp")

add_library(squashlib kernels.cpp ${HEADER_LIST})

target_include_directories(squashlib PUBLIC ../include)

target_compile_features(squashlib PUBLIC cxx_std_11)
