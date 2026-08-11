cmake_minimum_required(VERSION 3.22)
project(aigcllm C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_STANDARD 11)

if(NOT DEFINED CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(LLAMA_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/llama)

set(LLAMA_CORE_SRCS
    ${LLAMA_SRC_DIR}/llama.cpp
    ${LLAMA_SRC_DIR}/llama-adapter.cpp
    ${LLAMA_SRC_DIR}/llama-arch.cpp
    ${LLAMA_SRC_DIR}/llama-batch.cpp
    ${LLAMA_SRC_DIR}/llama-chat.cpp
    ${LLAMA_SRC_DIR}/llama-context.cpp
    ${LLAMA_SRC_DIR}/llama-cparams.cpp
    ${LLAMA_SRC_DIR}/llama-grammar.cpp
    ${LLAMA_SRC_DIR}/llama-graph.cpp
    ${LLAMA_SRC_DIR}/llama-hparams.cpp
    ${LLAMA_SRC_DIR}/llama-impl.cpp
    ${LLAMA_SRC_DIR}/llama-io.cpp
    ${LLAMA_SRC_DIR}/llama-kv-cache.cpp
    ${LLAMA_SRC_DIR}/llama-kv-cache-iswa.cpp
    ${LLAMA_SRC_DIR}/llama-memory.cpp
    ${LLAMA_SRC_DIR}/llama-memory-hybrid.cpp
    ${LLAMA_SRC_DIR}/llama-memory-recurrent.cpp
    ${LLAMA_SRC_DIR}/llama-mmap.cpp
    ${LLAMA_SRC_DIR}/llama-model-loader.cpp
    ${LLAMA_SRC_DIR}/llama-model-saver.cpp
    ${LLAMA_SRC_DIR}/llama-model.cpp
    ${LLAMA_SRC_DIR}/llama-sampling.cpp
    ${LLAMA_SRC_DIR}/llama-vocab.cpp
    ${LLAMA_SRC_DIR}/unicode.cpp
    ${LLAMA_SRC_DIR}/unicode-data.cpp
)

set(LLAMA_GGML_SRCS
    ${LLAMA_SRC_DIR}/ggml.c
    ${LLAMA_SRC_DIR}/ggml-alloc.c
    ${LLAMA_SRC_DIR}/ggml-backend.cpp
    ${LLAMA_SRC_DIR}/ggml-backend-reg.cpp
    ${LLAMA_SRC_DIR}/ggml-opt.cpp
    ${LLAMA_SRC_DIR}/ggml-quants.c
    ${LLAMA_SRC_DIR}/ggml-threading.cpp
    ${LLAMA_SRC_DIR}/gguf.cpp
)

set(LLAMA_CPU_SRCS
    ${LLAMA_SRC_DIR}/ggml-cpu/ggml-cpu.c
    ${LLAMA_SRC_DIR}/ggml-cpu/ggml-cpu.cpp
    ${LLAMA_SRC_DIR}/ggml-cpu/ops.cpp
    ${LLAMA_SRC_DIR}/ggml-cpu/binary-ops.cpp
    ${LLAMA_SRC_DIR}/ggml-cpu/unary-ops.cpp
    ${LLAMA_SRC_DIR}/ggml-cpu/repack.cpp
    ${LLAMA_SRC_DIR}/ggml-cpu/traits.cpp
    ${LLAMA_SRC_DIR}/ggml-cpu/vec.cpp
    ${LLAMA_SRC_DIR}/ggml-cpu/quants.c
)

# 模型架构实现 (models/*.cpp)
file(GLOB LLAMA_MODELS_SRCS ${LLAMA_SRC_DIR}/models/*.cpp)

add_library(llama STATIC
    ${LLAMA_CORE_SRCS}
    ${LLAMA_GGML_SRCS}
    ${LLAMA_CPU_SRCS}
    ${LLAMA_MODELS_SRCS}
)

target_include_directories(llama PUBLIC
    ${LLAMA_SRC_DIR}
    ${LLAMA_SRC_DIR}/ggml-cpu
    ${LLAMA_SRC_DIR}/ggml-cpu/arch
    ${LLAMA_SRC_DIR}/ggml-cpu/amx
    ${LLAMA_SRC_DIR}/ggml-blas
    ${LLAMA_SRC_DIR}/ggml-metal
    ${LLAMA_SRC_DIR}/ggml-opencl
    ${LLAMA_SRC_DIR}/ggml-hexagon
    ${LLAMA_SRC_DIR}/common
    ${LLAMA_SRC_DIR}/nlohmann
    ${LLAMA_SRC_DIR}/models
)

target_compile_definitions(llama PRIVATE
    GGML_USE_CPU
    GGML_CPU_ARM_ARCH=8
    LLAMA_NO_METAL=1
)

add_library(aigcllm SHARED
    LlmProEngine.cpp
)

target_link_libraries(aigcllm llama android log)
target_include_directories(aigcllm PRIVATE ${LLAMA_SRC_DIR})