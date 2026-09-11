# Builds llamacpp_logits against a llama.cpp checkout, linked statically.
#
#   cmake -S scripts/llamacpp_logits -B <build> -DLLAMA_CPP=<llama.cpp checkout> [-DGGML_CUDA=ON]
#   cmake --build <build> --config Release --target llamacpp_logits
cmake_minimum_required(VERSION 3.14)
project(llamacpp_logits C CXX)

set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(LLAMA_CURL OFF CACHE BOOL "" FORCE)

add_subdirectory(${LLAMA_CPP} llama.cpp)
add_executable(llamacpp_logits llamacpp_logits.cpp)
target_link_libraries(llamacpp_logits PRIVATE llama)
