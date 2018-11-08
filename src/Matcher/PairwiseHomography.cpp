
#include "Matcher/PairwiseHomography.hpp"

using std::cout;
using std::endl;

using namespace cv;
using namespace cv::xfeatures2d;

PairwiseHomography::PairwiseHomography()
: mFlagSmooth(true), mSmoothKernelSize(7),
  mMinHessian(5000), mMatchFilterRatio(0.6)
{

}

PairwiseHomography::~PairwiseHomography()
{

}

void PairwiseHomography::compute(InputArray _src, std::vector<Mat>& HsVec)
{
    if ( false == _src.isMatVector() )
    {
        // It is an error for now.
        BOOST_THROW_EXCEPTION( exception_base() << ExceptionInfoString( "_src is expected to be an instance of std::vector<Mat>." ) );
    }

    std::vector<Mat> srcVec;
    _src.getMatVector( srcVec );

    Ptr<SURF> detector = SURF::create( mMinHessian );
    std::vector<KeyPoint> keypoints1, keypoints2;
    Mat descriptors1, descriptors2;

    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);
    std::vector< std::vector<DMatch> > knn_matches;

    std::vector<DMatch> good_matches;
    std::vector<Point2f> goodPnt1, goodPnt2;

    Mat img1, img2, temp;
    Mat H;

    // Process img1.
    img1 = srcVec.at(0).clone();

    // Smooth.
    if ( true == mFlagSmooth )
    {
        GaussianBlur( img1, temp, Size( mSmoothKernelSize, mSmoothKernelSize ), 0, 0, BORDER_CONSTANT); 
        img1 = temp.clone();
    }

    // Detect feature points.
    detector->detectAndCompute( img1, noArray(), keypoints1, descriptors1 );

    std::vector<Mat>::iterator iter;
    iter = srcVec.begin(); iter++;
    int count = 1;

    for ( ; iter != srcVec.end(); iter++ )
    {
        cout << "Process " << count << " pair." << endl;

        img2 = (*iter).clone();

        // Smooth.
        if ( true == mFlagSmooth )
        {
            GaussianBlur( img2, temp, Size( mSmoothKernelSize, mSmoothKernelSize ), 0, 0, BORDER_CONSTANT); 
            img2 = temp.clone();
        }

        // Detect feature points.
        detector->detectAndCompute( img2, noArray(), keypoints2, descriptors2 );

        // Match.
        matcher->knnMatch( descriptors1, descriptors2, knn_matches, 2 );

        std::cout << "Matches: " << knn_matches.size() << std::endl;

        for (size_t i = 0; i < knn_matches.size(); i++)
        {
            if (knn_matches[i][0].distance < mMatchFilterRatio * knn_matches[i][1].distance)
            {
                good_matches.push_back(knn_matches[i][0]);
                goodPnt1.push_back( keypoints1.at( knn_matches[i][0].queryIdx ).pt );
                goodPnt2.push_back( keypoints2.at( knn_matches[i][0].trainIdx ).pt );
            }
        }

        std::cout << "Good matches: " << good_matches.size() << std::endl;

        H = findHomography( goodPnt2, goodPnt1, RANSAC );

        HsVec.push_back( H.clone() );

        std::cout << "H = " << std::endl;
        std::cout << H << std::endl;

        // Interchange values.
        img1 = img2.clone();
        keypoints1 = keypoints2;
        descriptors1 = descriptors2;

        keypoints2.clear();
        descriptors2.release();

        knn_matches.clear();
        good_matches.clear();
        goodPnt1.clear();
        goodPnt2.clear();

        count++;
    }
}