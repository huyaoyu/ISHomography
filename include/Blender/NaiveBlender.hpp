#ifndef __BLENDER_NAIVEBLENDER_HPP__
#define __BLENDER_NAIVEBLENDER_HPP__

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

#include <opencv2/calib3d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/stitching/detail/blenders.hpp>

#include "common/common.hpp"

using std::cout;
using std::endl;

using namespace cv;

class NaiveBlender
{
public:
    NaiveBlender();
    ~NaiveBlender();

    void warp_and_blend(InputArray _base, InputArray _new, InputArray _HNew, OutputArray _warpped, InputOutputArray _HOrigin);
};

#endif //__BLENDER_NAIVEBLENDER_HPP__