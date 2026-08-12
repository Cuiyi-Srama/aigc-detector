# AIGC Detector — LLM Pro 引擎 (Android JNI)
# 基于官方 llama.cpp (b10369) 编译静态库, 支持 gemma4 等新架构
cmake_minimum_required(VERSION 3.14)
project(aigcllm C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_STANDARD 11)

# 子项目模式: 关闭一切非必要组件, 只编 ggml + src 核心库
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_APP OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_COMMON OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_MTMD OFF CACHE BOOL "" FORCE)
set(LLAMA_CURL OFF CACHE BOOL "" FORCE)
set(LLAMA_NATIVE OFF CACHE BOOL "" FORCE)

add_subdirectory(llama-cpp)

add_library(aigcllm SHARED LlmProEngine.cpp)
target_include_directories(aigcllm PRIVATE
    llama-cpp/include
    llama-cpp/ggml/include
)
target_link_libraries(aigcllm PRIVATE llama)
