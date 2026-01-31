set(lab_benchmark_additional_files "" CACHE INTERNAL "")
set(lab_lib_additional_files "" CACHE INTERNAL "")
set(lab_test_additional_files "" CACHE INTERNAL "")

if(WIN32) # Benchmark
	list(APPEND lab_benchmark_additional_files "benchmark.h")
endif()
	
if(WIN32) # Lib	
	list(APPEND lab_lib_additional_files "authors.h")
	list(APPEND lab_lib_additional_files "utils.h")
	list(APPEND lab_lib_additional_files "import.h")
	list(APPEND lab_lib_additional_files "Node.h")
	list(APPEND lab_lib_additional_files "DiGraph.h")
	list(APPEND lab_lib_additional_files "Matrix.h")
endif()

if(WIN32) # Tests
	list(APPEND lab_test_additional_files "test.h")	
endif()
