
include_directories(${PROJECT_SOURCE_DIR}/)

include_directories(${external_includes})

add_subdirectory(resampling)

add_subdirectory(flip_gt)

add_subdirectory(reg_dce)