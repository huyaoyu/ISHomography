#ifndef __MATCHER_PAIRWISEHOMOGRAPHY_HPP__
#define __MATCHER_PAIRWISEHOMOGRAPHY_HPP__

#include <exception>
#include <boost/exception/all.hpp>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

#include <opencv2/calib3d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/stitching/detail/blenders.hpp>

#include <iostream>
#include <string>

#include "common/common.hpp"

using std::cout;
using std::endl;

using namespace cv;

class PairwiseHomography
{
public:
    PairwiseHomography();
    ~PairwiseHomography();

    void compute(InputArray _src, std::vector<Mat>& HsVec);

public:
    bool mFlagSmooth;
    int mSmoothKernelSize;
    int mMinHessian;
    real mMatchFilterRatio;
};

#endif // __MATCHER_PAIRWISEHOMOGRAPHY_HPP__