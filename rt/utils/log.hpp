#include <cstdio>
#include <iostream>

#define RT_WARN(description)                                                                                 \
    std::cerr << __FILE__ << ':' << __LINE__ << ": <" << __func__ << "> " << (description) << std::endl;

#define RT_VEC3(vec) (float)(vec)[0], (float)(vec)[1], (float)(vec)[2]
#define RT_VEC4(vec) (float)(vec)[0], (float)(vec)[1], (float)(vec)[2], (float)(vec)[3]

#define RT_VEC3_F "(%f,%f,%f)"
#define RT_VEC4_F "(%f,%f,%f,%f)"
