cmake_minimum_required(VERSION 3.4.1)
set(TARGET native_lib)
project(${TARGET} CXX)
set(CMAKE_CXX_STANDARD 14)

set(build_DIR ${CMAKE_SOURCE_DIR}/../../../build)
set(OPENCV_ANDROID_SDK ${CMAKE_SOURCE_DIR}/../../../../opencv)

# message(SEND_ERROR "OPENCV_ANDROID_SDK set to ${OPENCV_ANDROID_SDK}")

file(GLOB JNI_SOURCES ${CMAKE_CURRENT_LIST_DIR}/kr_dailylime_torchcvhelloworld_MainActivity.cpp)

add_library(${TARGET} SHARED ${JNI_SOURCES})

file(GLOB PYTORCH_INCLUDE_DIRS "${build_DIR}/pytorch_android*.aar/headers")
file(GLOB PYTORCH_LINK_DIRS "${build_DIR}/pytorch_android*.aar/jni/${ANDROID_ABI}")

target_compile_options(${TARGET} PRIVATE
        -fexceptions
        )

set(BUILD_SUBDIR ${ANDROID_ABI})

find_library(PYTORCH_LIBRARY pytorch_jni
        PATHS ${PYTORCH_LINK_DIRS}
        NO_CMAKE_FIND_ROOT_PATH)
find_library(FBJNI_LIBRARY fbjni
        PATHS ${PYTORCH_LINK_DIRS}
        NO_CMAKE_FIND_ROOT_PATH)

# OpenCV
set(OPENCV_INCLUDE_DIR "${OPENCV_ANDROID_SDK}/native/jni/include")

target_include_directories(${TARGET} PRIVATE
        "${OPENCV_INCLUDE_DIR}"
        ${PYTORCH_INCLUDE_DIRS})

set(OPENCV_LIB_DIR "${OPENCV_ANDROID_SDK}/native/libs/${ANDROID_ABI}")

find_library(OPENCV_LIBRARY opencv_java4
        PATHS ${OPENCV_LIB_DIR}
        NO_CMAKE_FIND_ROOT_PATH)

target_link_libraries(${TARGET}
        ${PYTORCH_LIBRARY}
        ${FBJNI_LIBRARY}
        ${OPENCV_LIBRARY}
        log)