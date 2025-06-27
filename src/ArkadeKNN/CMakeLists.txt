
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

add_executable(ArkadeKNN
        MainCode.cpp
        ArkadeModel.cpp
        CreateBVH.cpp
        VarGlobal.cpp
        TransMonotoma.cpp
        methods.h
)

message(STATUS "ArkadeKNN-ptx target: ${ArkadeKNN-ptx}")


target_link_libraries(ArkadeKNN
        PRIVATE
        ArkadeKNN-ptx
        owl::owl
)
