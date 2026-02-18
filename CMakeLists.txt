cmake_minimum_required(VERSION 3.10)

project(robotic_arm_ik)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# -----------------------------
# Source files
# -----------------------------

set(KINEMATICS
    src/kinematics/AnalyticalIK.cpp
    src/kinematics/ForwardKinematics.cpp
)

set(PLANNING
    src/planning/TrajectoryPlanner.cpp
)

set(COLLISION
    src/collision/ObstacleAvoidance.cpp
)

set(CONTROL
    src/control/Controller.cpp
)

# -----------------------------
# Main robotic arm executable
# -----------------------------

add_executable(robotic_arm
    src/main.cpp
    ${KINEMATICS}
    ${PLANNING}
    ${COLLISION}
    ${CONTROL}
)

# -----------------------------
# ML test executable
# -----------------------------

add_executable(ml_test
    tests/ml_test.cpp
    src/kinematics/ML_IK.cpp
)

# -----------------------------
# ONNX Runtime paths
# (EDIT if your path differs)
# -----------------------------

set(ONNXRUNTIME_DIR "C:/onnxruntime-win-x64-1.24.1")

target_include_directories(ml_test PRIVATE
    ${ONNXRUNTIME_DIR}/include
)

target_link_directories(ml_test PRIVATE
    ${ONNXRUNTIME_DIR}/lib
)

target_link_libraries(ml_test
    onnxruntime
)

# -----------------------------
# Windows: copy DLL after build
# -----------------------------

add_custom_command(TARGET ml_test POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${ONNXRUNTIME_DIR}/lib/onnxruntime.dll"
    $<TARGET_FILE_DIR:ml_test>
)
