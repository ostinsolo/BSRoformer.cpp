cmake_minimum_required(VERSION 3.18)
project(cuda_batched_post LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)

add_executable(cuda_batched_post_bench
    main.cpp
    cuda_post_pipeline.cu
    ../../src/audio.cpp
)

target_include_directories(cuda_batched_post_bench PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../..
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../src
    ${CMAKE_CURRENT_SOURCE_DIR}/../../third_party
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../backends/roformer-mlx-cpp/include
)

target_link_libraries(cuda_batched_post_bench PRIVATE
    bs_roformer
    CUDA::cudart
    CUDA::cufft
)

set_target_properties(cuda_batched_post_bench PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
    CUDA_STANDARD 17
)

if(MSVC)
    target_compile_options(cuda_batched_post_bench PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/W3 /utf-8>)
endif()

bsr_copy_ggml_runtime_dlls(cuda_batched_post_bench)
