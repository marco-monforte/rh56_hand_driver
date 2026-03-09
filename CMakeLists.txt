cmake_minimum_required(VERSION 3.8)
project(rh56_hand_driver)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# --------------------------------------------------
# Dependencies
# --------------------------------------------------

set(CMAKE_PREFIX_PATH
  ${CMAKE_PREFIX_PATH}
  /opt/unitree_robotics
)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(hand_msgs REQUIRED)
find_package(unitree_sdk2 REQUIRED)
find_package(CycloneDDS REQUIRED)

# --------------------------------------------------
# C++ Node
# --------------------------------------------------

set(HAND_IDL_DIR
  ${CMAKE_CURRENT_SOURCE_DIR}/inspire_hand_ws/inspire_hand_sdk/hand_idl
)

add_executable(rh56dftp_driver_node
  src/rh56dftp_driver_node.cpp

  inspire_hand_ws/inspire_hand_sdk/hand_idl/inspire_hand_ctrl.cpp
  inspire_hand_ws/inspire_hand_sdk/hand_idl/inspire_hand_state.cpp
  inspire_hand_ws/inspire_hand_sdk/hand_idl/inspire_hand_touch.cpp

)

ament_target_dependencies(rh56dftp_driver_node
  rclcpp
  hand_msgs
)

target_include_directories(rh56dftp_driver_node PUBLIC
  ${HAND_IDL_DIR}
)

target_link_libraries(rh56dftp_driver_node
  unitree_sdk2
  CycloneDDS::ddsc
)

install(TARGETS
  rh56dftp_driver_node
  DESTINATION lib/${PROJECT_NAME}
)

# --------------------------------------------------
# Python Node
# --------------------------------------------------

install(PROGRAMS
  src/hand_mapper_node.py
  DESTINATION lib/${PROJECT_NAME}
)

# --------------------------------------------------
# Package
# --------------------------------------------------

ament_package()