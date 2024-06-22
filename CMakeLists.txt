cmake_minimum_required(VERSION 3.29.6)

project(SQuAsh
        LANGUAGES CXX HIP
        VERSION 0.0.1
)
if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_EXTENSIONS OFF)
  include(CTest)
endif()

set(CMAKE_MODULE_PATH "$ENV{ROCM_PATH}/lib/cmake/hip" ${CMAKE_MODULE_PATH})
find_package(HIP REQUIRED)
find_package(ROCM REQUIRED)
enable_language(HIP)

add_subdirectory(src)


if((CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
   AND BUILD_TESTING)
  add_subdirectory(test)
endif()

