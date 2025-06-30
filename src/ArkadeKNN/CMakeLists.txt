# Macros a utilizar
add_compile_definitions(KN=10)
add_compile_definitions(FLOAT_MAX=3.402823466e+38)
add_compile_definitions(FLOAT_MIN=1.175494351e-38)

add_definitions(-DENHANCE)

embed_ptx(
        OUTPUT_TARGET ArkadeKNN-ptx
        PTX_LINK_LIBRARIES owl::owl
        DEFINES KN=20
        SOURCES DeviceCode.cu
)

add_library(ArkadeKNN_lib
        ArkadeModel.cpp
        BaseModel.cpp
        FastRNN.cpp
        CreateBVH.cpp
        VarGlobal.cpp
        TransMonotoma.cpp
        methods.h
)

set_target_properties(ArkadeKNN_lib PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_executable(ArkadeKNN
        MainCode.cpp
)

message(STATUS "ArkadeKNN-ptx target: ${ArkadeKNN-ptx}")


target_link_libraries(ArkadeKNN
        PRIVATE
        ArkadeKNN_lib
        ArkadeKNN-ptx
        owl::owl
)

target_link_libraries(ArkadeKNN_lib
        PRIVATE
        ArkadeKNN-ptx
        owl::owl
)

option(ARKADE_BUILD_PYARKADE "Build PyArkade Python Wrappers?" ON)
if (ARKADE_BUILD_PYARKADE)
  add_subdirectory(pyArkadeKNN)
endif()
