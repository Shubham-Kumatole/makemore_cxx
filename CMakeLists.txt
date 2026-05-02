cmake_minimum_required(VERSION 3.20)
project(makemore VERSION 1.0 LANGUAGES CXX)
set(CMAKE_PREFIX_PATH "/opt/libtorch")
find_package(Torch REQUIRED)
set(Python3_ROOT_DIR "/opt/homebrew/opt/python@3.14")
find_package(Python3 COMPONENTS Interpreter Development NumPy REQUIRED)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Source files - list explicitly to avoid GLOB pitfalls
add_executable(${PROJECT_NAME}
    src/main.cpp
    src/common.cpp
    src/mlp.cpp
    src/linear.cpp
    src/cpptorch.cpp
)
# Your own headers in include/
target_include_directories(${PROJECT_NAME} PRIVATE
${CMAKE_SOURCE_DIR}/include
${TORCH_INCLUDE_DIRS}
)
target_link_libraries(makemore PRIVATE 
    Python3::Python 
    Python3::NumPy 
    "${TORCH_LIBRARIES}"
)

target_compile_options(${PROJECT_NAME} PRIVATE
    -Wall
    -Wextra
    -Wpedantic
    -O2
)

# Copy a single file to the folder containing the executable
add_custom_command(TARGET makemore POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
            "${CMAKE_CURRENT_SOURCE_DIR}/names.txt"
            "$<TARGET_FILE_DIR:makemore>/names.txt"
    COMMENT "Copying names.txt to build directory"
)
