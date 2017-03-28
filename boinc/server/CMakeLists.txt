cmake_minimum_required(VERSION 2.8)
project (NeuralNetwork_BOINC_server)

find_package( OpenSSL REQUIRED )


add_executable( ConvNetSeam_work_generator ConvNetSeam_work_generator.cxx)
target_link_libraries( ConvNetSeam_work_generator ${BOINC_SERVER_LIBRARIES} ${MYSQL_LIBRARIES} ConvNetEvent ${BOINC_LIBRARIES})

add_executable( ConvNetSeam_validator
	${BOINC_INCLUDE_DIR}/sched/assimilator
	${BOINC_INCLUDE_DIR}/sched/validate_util
	ConvNetSeam_assimilator.cxx
	#/home/tdesell/csg_boinc/lib/parse.cpp
)

