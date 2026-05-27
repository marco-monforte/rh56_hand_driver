cmake_minimum_required(VERSION 3.8)
project(rh56_hand_driver)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# --------------------------------------------------
# Dependencies
# --------------------------------------------------

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(hand_msgs REQUIRED)

add_executable(rh56dftp_driver_node
  src/rh56dftp_driver_node.cpp
)

target_compile_features(rh56dftp_driver_node PUBLIC cxx_std_17)

ament_target_dependencies(rh56dftp_driver_node
  rclcpp
  hand_msgs
)

add_executable(hand_mapper_node
  src/hand_mapper_node.cpp
)

target_compile_features(hand_mapper_node PUBLIC cxx_std_17)

ament_target_dependencies(hand_mapper_node
  rclcpp
  hand_msgs
)

install(TARGETS
  rh56dftp_driver_node
  hand_mapper_node
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY
  launch
  DESTINATION share/${PROJECT_NAME}
)

# --------------------------------------------------
# Package
# --------------------------------------------------

ament_package()
