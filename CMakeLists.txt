cmake_minimum_required(VERSION 3.16)
project(slmpt LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -funroll-loops -march=native -funsafe-math-optimizations -ffp-contract=fast")

find_package(Threads REQUIRED)

add_library(slmpt_core STATIC
  src/model.cpp
  src/optimizer.cpp
  src/tokenizer.cpp
  src/dataset.cpp
  src/sampling.cpp
  src/serialization.cpp
)
target_include_directories(slmpt_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(slmpt_core PUBLIC Threads::Threads)

foreach(app train chat gradcheck bpe)
  add_executable(slmpt_${app} apps/${app}.cpp)
  target_link_libraries(slmpt_${app} PRIVATE slmpt_core)
endforeach()
