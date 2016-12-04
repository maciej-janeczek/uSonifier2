cmake_minimum_required(VERSION 2.8 FATAL_ERROR)
### Set project parameters ###
set(ProjectName uSonifier)
project(${ProjectName})
set(Project_SRC src/main.cpp)

### Find required packages ###
find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)

### Set CXX and CUDA flags ###
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
elseif(COMPILER_SUPPORTS_CXX0X)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
else()
    message(STATUS "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif()
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -std=c++11 -use_fast_math -arch sm_32)
#set(CUDA_SEPARABLE_COMPILATION ON)

### Find all cpp files ###
file(GLOB_RECURSE Scene_SRC src/Scene/*.cpp)
file(GLOB_RECURSE CSoundManager_SRC src/CSoundManager/*.cpp)
file(GLOB_RECURSE Camera_SRC src/Camera/*.cpp)
file(GLOB_RECURSE Stereo_SRC src/Stereo/*.cpp)

### Set Csound and DUO camera libs labels ###
set(Csound_LIBS libcsound64.so libcsnd6.so)
set(DUO_LIB ${CMAKE_SOURCE_DIR}/lib/libDUO.so)

### Set elements to be compiled as static libraries ###
cuda_add_library( Scene ${Scene_SRC} )
cuda_add_library( CSoundManager ${CSoundManager_SRC} )
cuda_add_library( Stereo ${Stereo_SRC} )
cuda_add_library( Camera ${Camera_SRC} )
cuda_add_library( StereoCUDA "${CMAKE_SOURCE_DIR}/src/Stereo/cuda/cuStereo.cu" )
cuda_add_library( TypesCUDA "${CMAKE_SOURCE_DIR}/src/Stereo/dataTypes/cuda/cuTypes.cu" )

### Set Executable ###
cuda_add_executable(${ProjectName} ${Project_SRC}) 

### Link libraries to executable ###
target_link_libraries(${ProjectName} CSoundManager Scene Camera Stereo StereoCUDA TypesCUDA ${OpenCV_LIBS} ${DUO_LIB} ${Csound_LIBS} )
add_definitions("-DUSE_DOUBLE")