#add_library(nms SHARED
#${PROJECT_SOURCE_DIR}/include/src/nms.cpp
#)
#add_executable(yolo src.cpp)
# CMakeList.txt: torch_exe 的 CMake 项目，在此处包括源代码并定义
# 项目特定的逻辑。
#
cmake_minimum_required (VERSION 3.8)

project(demo)

# set(CMAKE_PREFIX_PATH "/home/ps/libtorch/share/cmake/Torch/") #//注意这里填自己解压libtorch时的路径

set(OpenCV_DIR "/usr/local/include/opencv4/")

# include_directories("./include/"   "/home/ps/libtorch/include"  "/home/ps/libtorch/include/torch/csrc/api/include/torch/nn/functional/")
include_directories("./dlpack/include/"  "/home/ps/tvm/include/" "/home/ps/tvm/3rdparty/dmlc-core/include/")
link_directories("/home/ps/tvm/build/")
# find_package(Torch REQUIRED)
find_package(OpenCV QUIET)

# if(NOT Torch_FOUND)
#     message(FATAL_ERROR "Pytorch Not Found!")
# endif(NOT Torch_FOUND)

if(NOT OpenCV_FOUND)
    find_package(OpenCV 2.4.3 QUIET)
    if(NOT OpenCV_FOUND)
        message(FATAL_ERROR "OpenCV > 2.4.3 not found.")
    endif()
endif()


# find_library(LIBTVM_PATH tvmlib /home/ps/tvm/build)

# aux_source_directory(./src/ SRC_LIST)
aux_source_directory(. SRC_LIST)


# 将源代码添加到此项目的可执行文件。
add_executable (demo ${SRC_LIST})

# target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS}) # -llibtvm_runtime.so -llibtvm.so
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} libtvm.so libtvm_runtime.so)

# # TODO: 如有需要，请添加测试并安装目标。
# # target_link_libraries(${PROJECT_NAME} ${TORCH_LIBRARIES} ${OpenCV_LIBS})
# target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} ${LIBTVM_PATH})
# target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
# set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 14)
# # set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 11)