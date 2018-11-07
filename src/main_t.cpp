
#include <iostream>
#include <string>

#include <exception>
#include <boost/exception/all.hpp>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

#ifdef HAVE_OPENCV_XFEATURES2D

#include <opencv2/calib3d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>

using namespace cv;
using namespace cv::xfeatures2d;

using std::cout;
using std::endl;

struct exception_base : virtual std::exception, virtual boost::exception { };

typedef boost::error_info<struct tag_info_string, std::string> ExceptionInfoString;

const char* keys =
    "{ help h |                               | Print help message. }"
    "{ input1 | ../Data/1458400805.000000.jpg | Path to input image 1. }"
    "{ input2 | ../Data/1458400806.000000.jpg | Path to input image 2. }";

typedef float real;

class PairwiseHomography
{
public:
    PairwiseHomography()
    : mFlagSmooth(true), mSmoothKernelSize(7),
      mMinHessian(5000), mMatchFilterRatio(0.6)
    { }

    ~PairwiseHomography() { }

    void compute(InputArray _src, OutputArray _Hs)
    {
        if ( false == _src.isMatVector() )
        {
            // It is an error for now.
            BOOST_THROW_EXCEPTION( exception_base() << ExceptionInfoString( "_src is expected to be an instance of std::vector<Mat>." ) );
        }

        if ( false == _Hs.isMatVector() )
        {
            // It is an error for now.
            BOOST_THROW_EXCEPTION( exception_base() << ExceptionInfoString( "_Hs is expected to be an instance of std::vector<Mat>." ) );
        }

        std::vector<Mat> srcVec;
        _src.getMatVector( srcVec );

        std::vector<Mat> HsVec;
        _Hs.getMatVector( HsVec );

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
        }
    }

public:
    bool mFlagSmooth;
    int mSmoothKernelSize;
    int mMinHessian;
    real mMatchFilterRatio;
};

int main_backup( int argc, char* argv[] );

int main( int argc, char* argv[] )
{
    CommandLineParser parser( argc, argv, keys );

    std::string imgFn1 = parser.get<String>("input1");
    std::string imgFn2 = parser.get<String>("input2");

    Mat img1 = imread( imgFn1, IMREAD_GRAYSCALE );
    Mat img2 = imread( imgFn2, IMREAD_GRAYSCALE );

    if ( img1.empty() || img2.empty() )
    {
        cout << "Could not open or find the image!\n" << endl;
        parser.printMessage();
        return -1;
    }

    std::vector<Mat> inputMats;
    std::vector<Mat> Hs;

    inputMats.push_back( img1 );
    inputMats.push_back( img2 );

    PairwiseHomography ph;
    ph.compute( inputMats, Hs );

    return 0;
}
#else
int main()
{
    std::cout << "This tutorial code needs the xfeatures2d contrib module to be run." << std::endl;
    return 0;
}
#endif

int main_backup( int argc, char* argv[] )
{
    CommandLineParser parser( argc, argv, keys );

    std::string imgFn1 = parser.get<String>("input1");
    std::string imgFn2 = parser.get<String>("input2");

    Mat img1 = imread( imgFn1, IMREAD_GRAYSCALE );
    Mat img2 = imread( imgFn2, IMREAD_GRAYSCALE );

    if ( img1.empty() || img2.empty() )
    {
        cout << "Could not open or find the image!\n" << endl;
        parser.printMessage();
        return -1;
    }

    Mat temp;
    GaussianBlur( img1, temp, Size( 7, 7 ), 0, 0, BORDER_CONSTANT); img1 = temp.clone();
    GaussianBlur( img2, temp, Size( 7, 7 ), 0, 0, BORDER_CONSTANT); img2 = temp.clone();

    //-- Step 1: Detect the keypoints using SURF Detector, compute the descriptors
    int minHessian = 5000;

    Ptr<SURF> detector = SURF::create( minHessian );

    std::vector<KeyPoint> keypoints1, keypoints2;
    Mat descriptors1, descriptors2;

    detector->detectAndCompute( img1, noArray(), keypoints1, descriptors1 );
    detector->detectAndCompute( img2, noArray(), keypoints2, descriptors2 );

    //-- Step 2: Matching descriptor vectors with a FLANN based matcher
    // Since SURF is a floating-point descriptor NORM_L2 is used
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);
    std::vector< std::vector<DMatch> > knn_matches;

    matcher->knnMatch( descriptors1, descriptors2, knn_matches, 2 );

    std::cout << "Matches: " << knn_matches.size() << std::endl;

    //-- Filter matches using the Lowe's ratio test
    const float ratio_thresh = 0.6f;

    std::vector<DMatch> good_matches;
    std::vector<Point2f> goodPnt1, goodPnt2;
    for (size_t i = 0; i < knn_matches.size(); i++)
    {
        if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance)
        {
            good_matches.push_back(knn_matches[i][0]);
            goodPnt1.push_back( keypoints1.at( knn_matches[i][0].queryIdx ).pt );
            goodPnt2.push_back( keypoints2.at( knn_matches[i][0].trainIdx ).pt );
        }
    }

    std::cout << "Good matches: " << good_matches.size() << std::endl;

    //-- Draw matches
    Mat img_matches;

    drawMatches( img1, keypoints1, img2, keypoints2, good_matches, img_matches, Scalar::all(-1),
                 Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );

    //-- Show detected matches
    namedWindow("Good Matches", WINDOW_NORMAL);
    imshow("Good Matches", img_matches );

    // Homography.
    Mat H;
    H = findHomography( goodPnt2, goodPnt1, RANSAC );

    std::cout << "H = " << std::endl;
    std::cout << H << std::endl;

    // Decomposition of the homography matrix.
    Mat cameraMatrix = Mat::eye(3, 3, CV_64FC1);
    cameraMatrix.at<double>(0,0) = 5106.382;
    cameraMatrix.at<double>(1,1) = 5128.206;
    cameraMatrix.at<double>(0,2) = 3000;
    cameraMatrix.at<double>(1,2) = 2000;

    std::vector<Mat> DH_R, DH_T, DH_N;
    int DHSolutions = 0;

    DHSolutions = decomposeHomographyMat( 
        H, cameraMatrix, DH_R, DH_T, DH_N );

    for ( int i = 0; i < DHSolutions; i++ )
    {
        std::cout << "Solution " << i << std::endl;
        std::cout << "R = " << std::endl;
        std::cout << DH_R.at(i) << std::endl;
        std::cout << "T = " << std::endl;
        std::cout << DH_T.at(i) << std::endl;
        std::cout << "N = " << std::endl;
        std::cout << DH_N.at(i) << std::endl;
    }

    // Open the color images.
    img1 = imread( imgFn1 );
    img2 = imread( imgFn2 );

    // Warp.
    Mat queryImage, trainImage;
    Mat qImgH = Mat::eye(3,3,CV_64FC1);
    qImgH.at<double>(0, 2) += img1.cols;
    qImgH.at<double>(1, 2) += img1.rows / 2;

    warpPerspective( img1, queryImage,     qImgH, Size( img1.cols*2, img1.rows*2 ), INTER_LINEAR, BORDER_CONSTANT );
    warpPerspective( img2, trainImage, qImgH * H, Size( img1.cols*2, img1.rows*2 ), INTER_LINEAR, BORDER_CONSTANT );
    // Threshold.
    Mat trainImageThreshold, bitwiseNotThreshold;
    Mat trainImageGray;
    cvtColor( trainImage, trainImageGray, COLOR_BGR2GRAY );
    threshold( trainImageGray, trainImageThreshold, 0, 255, THRESH_BINARY );
    bitwise_not( trainImageThreshold, bitwiseNotThreshold );

    Mat enlargedImage = Mat::zeros( img1.rows*2, img1.cols*2, CV_8UC3 );

    std::cout << "Test" << std::endl;

    std::cout << "queryImage:    H" << queryImage.rows << ", W" << queryImage.cols << std::endl;
    std::cout << "enlargedImage: H" << enlargedImage.rows << ", W" << enlargedImage.cols << std::endl;
    std::cout << "bitwiseNotThreshold: C" << bitwiseNotThreshold.channels() 
              << ", H" << bitwiseNotThreshold.rows
              << ", W" << bitwiseNotThreshold.cols << std::endl;

    add( enlargedImage, queryImage, enlargedImage, bitwiseNotThreshold, CV_8U );
    add( enlargedImage, trainImage, enlargedImage, noArray(), CV_8U );

    namedWindow( "Warpped img2", WINDOW_NORMAL );
    imshow( "Warpped img2", enlargedImage );

    // Save the image.
    std::vector<int> jpegParams;
    jpegParams.push_back( IMWRITE_JPEG_QUALITY );
    jpegParams.push_back( 100 );

    imwrite( "result.jpg", enlargedImage, jpegParams );

    waitKey();
    return 0;
}