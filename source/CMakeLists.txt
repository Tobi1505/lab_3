
cmake_minimum_required(VERSION 3.20)

project(lab_3
		LANGUAGES C CXX
)

# ------------------------------------------------------------
# Options
# ------------------------------------------------------------
option(ENABLE_CUDA "Enable CUDA support" ON)

# ------------------------------------------------------------
# C++ standard
# ------------------------------------------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ------------------------------------------------------------
# CUDA (optional)
# ------------------------------------------------------------
if (ENABLE_CUDA)
	if (CMAKE_SYSTEM_NAME STREQUAL "Linux")
		enable_language(CUDA)
		message(STATUS "CUDA enabled")
		set(CMAKE_CUDA_STANDARD 17)
		set(CMAKE_CUDA_STANDARD_REQUIRED ON)
	else()
		message(WARNING "CUDA requested, but not supported on this system. Disabling CUDA.")
		set(ENABLE_CUDA OFF)
	endif()
endif()

# ------------------------------------------------------------
# Library
# ------------------------------------------------------------
add_library(lab_lib STATIC)

target_sources(lab_lib
		PRIVATE
		image/bitmap_image.cpp
		io/image_parser.cpp

		grayscale_image.cpp
		intermediate_image.cpp
		utilities.cpp
)

# CUDA sources (only when enabled)
if (ENABLE_CUDA)
	target_sources(lab_lib
			PRIVATE
			intermediate_image.cu
			gpu_memory_management.cu
			gpu_downsampling.cu
			gpu_NN_upscaling.cu
			gpu_LIN_upscaling.cu
			gpu_reductions.cu
			gpu_matrix_convolution.cu
			gpu_utilities.cu
	)
endif()

target_include_directories(lab_lib
		PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(lab_lib
		PUBLIC project_options project_libraries
)

# ------------------------------------------------------------
# Executable
# ------------------------------------------------------------
add_executable(lab main.cpp)
target_link_libraries(lab PRIVATE lab_lib)

# ------------------------------------------------------------
# IDE niceties (CLion / Visual Studio)
# ------------------------------------------------------------
get_target_property(lab_lib_sources lab_lib SOURCES)
source_group(TREE ${CMAKE_SOURCE_DIR} FILES ${lab_lib_sources})
