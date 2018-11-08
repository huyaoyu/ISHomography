
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
#include <opencv2/stitching/detail/blenders.hpp>

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

typedef double real;

std::vector<int> gJpegParams;

class PairwiseHomography
{
public:
    PairwiseHomography()
    : mFlagSmooth(true), mSmoothKernelSize(7),
      mMinHessian(5000), mMatchFilterRatio(0.6)
    { }

    ~PairwiseHomography() { }

    void compute(InputArray _src, std::vector<Mat>& HsVec)
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

public:
    bool mFlagSmooth;
    int mSmoothKernelSize;
    int mMinHessian;
    real mMatchFilterRatio;
};

void put_new_corners( InputArray _src, InputArray _H, OutputArray _corners )
{
    Mat src = _src.getMat();
    Mat H   = _H.getMat();
    
    // Get the four corners.
    Mat corners = Mat::ones( 3, 4, CV_64FC1 );

    corners.at<real>(0, 0) =        0; corners.at<real>(1, 0) = 0;
    corners.at<real>(0, 1) = src.cols; corners.at<real>(1, 1) = 0;
    corners.at<real>(0, 2) = src.cols; corners.at<real>(1, 2) = src.rows;
    corners.at<real>(0, 3) =        0; corners.at<real>(1, 3) = src.rows;

    // Warp the corners;
    _corners.create( Size( corners.cols, corners.rows), corners.type() );
    _corners.getMat() = H * corners;
}

void warp_and_blend_images( InputArray _base, InputArray _new, InputArray _HNew, OutputArray _warpped, InputOutputArray _HOrigin )
{
    Mat baseImg = _base.getMat();
    Mat newImg  = _new.getMat();
    Mat HNew    = _HNew.getMat();
    Mat HOrigin = _HOrigin.getMat();
    
    // Get the image coordinates of the warpped corners of the new image.
    Mat corners;
    put_new_corners( newImg, HOrigin * HNew, corners );

    cout << "The four corners of the new image are: " << endl;
    for ( int i = 0; i < corners.cols; i++ )
    {
        cout << "( " << corners.at<real>(0, i) << ", " << corners.at<real>(1, i) << " )" << endl;
    }

    // Dimensions of the base image.
    int W = baseImg.cols;
    int H = baseImg.rows;

    real shiftPositiveX = 0.0;
    real shiftPositiveY = 0.0;
    real enlargeX = 0.0;
    real enlargeY = 0.0;
    real tempX = 0.0;
    real tempY = 0.0;

    // Figure out the new image size and the associated homography matrix.
    for ( int i = 0; i < corners.cols; i++ )
    {
        tempX = corners.at<real>(0, i);
        if ( tempX < 0 && -tempX > shiftPositiveX )
        {
            shiftPositiveX = -tempX;
        }

        tempY = corners.at<real>(1, i);
        if ( tempY < 0 && -tempY > shiftPositiveY )
        {
            shiftPositiveY = -tempY;
        }
    }

    for ( int i = 0; i < corners.cols; i++ )
    {
        tempX = corners.at<real>(0, i);
        if ( tempX > W + enlargeX )
        {
            enlargeX = tempX - W;
        }

        tempY = corners.at<real>(1, i);
        if ( tempY > H + enlargeY )
        {
            enlargeY = tempY - H;
        }
    }

    cout << "ShiftPositiveX = " << shiftPositiveX << ", shiftPositiveY = " << shiftPositiveY << endl;
    // Update the original homography matrix of the base image.
    Mat shift = Mat::eye( 3, 3, CV_64FC1 );
    shift.at<real>(0, 2) = shiftPositiveX;
    shift.at<real>(1, 2) = shiftPositiveY;

    // New image size.
    W += shiftPositiveX + enlargeX;
    H += shiftPositiveY + enlargeY;

    // Create new image.
    Mat enlargedBase, warppedNew;

    // Move the base image.
    warpPerspective( baseImg, enlargedBase, shift, Size( W, H ), INTER_LINEAR, BORDER_CONSTANT );
    
    // Warp the new image.
    Mat shiftedH = shift * HOrigin * HNew;
    shiftedH /= shiftedH.at<real>(2, 2);
    warpPerspective( newImg, warppedNew, shiftedH, Size( W, H ), INTER_LINEAR, BORDER_CONSTANT );
    
    // Update the original homography matrix of the base image.
    // HOrigin = shiftedH;
    _HOrigin.assign( shiftedH );

    // // Test use.
    // std::vector<int> jpegParams;
    // jpegParams.push_back( IMWRITE_JPEG_QUALITY );
    // jpegParams.push_back( 100 );
    // namedWindow("MovedBaseImage", WINDOW_NORMAL);
    // imshow("MovedBaseImage", enlargedBase);
    // imwrite("EnlargedBaseImage.jpg", enlargedBase, jpegParams);
    // namedWindow("MovedNewImage", WINDOW_NORMAL);
    // imshow("MovedNewImage", warppedNew);
    // imwrite("WarppedNewImage.jpg", warppedNew, jpegParams);

    _warpped.create( H, W, CV_8UC3);
    Mat warpped = _warpped.getMat();

    Ptr<detail::MultiBandBlender> blender = new detail::MultiBandBlender();
    Point leftCorner = Point(0, 0);
    // std::vector<Point> leftCornerVec;
    // leftCornerVec.push_back( leftCorner );
    // leftCornerVec.push_back( leftCorner );

    // std::vector<Size> imgSizeVec;
    // imgSizeVec.push_back( Size( W, H ) );
    // imgSizeVec.push_back( Size( W, H ) );

    blender->prepare( Rect( 0, 0, W, H ) );

    // // Threshold.
    // Mat newImgThreshold, bitwiseNotThreshold;
    // Mat newImgGray;
    // cvtColor( warppedNew, newImgGray, COLOR_BGR2GRAY );
    // threshold( newImgGray, newImgThreshold, 0, 255, THRESH_BINARY );
    // bitwise_not( newImgThreshold, bitwiseNotThreshold );

    // cout << "warpped ( " << warpped.rows << ", " << warpped.cols << " )" << endl;
    // cout << "bitwiseNotThreshold ( " << bitwiseNotThreshold.rows << ", " << bitwiseNotThreshold.cols << " )" << endl;

    // add( warpped, enlargedBase, warpped, bitwiseNotThreshold, CV_8U );
    // add( warpped, warppedNew, warpped, noArray(), CV_8U );

    Mat grayBase, maskBase;
    cvtColor( enlargedBase, grayBase, COLOR_BGR2GRAY );
    threshold( grayBase, maskBase, 1, 255, THRESH_BINARY );

    blender->feed( enlargedBase, maskBase, leftCorner );

    Mat grayNew, maskNew;
    cvtColor( warppedNew, grayNew, COLOR_BGR2GRAY );
    threshold( grayNew, maskNew, 1, 255, THRESH_BINARY );

    blender->feed( warppedNew, maskNew, leftCorner );

    Mat warppedTemp, maskResult;
    blender->blend( warppedTemp, maskResult ); // Note: The first argument for blend() is of type InputOutputArray!

    warppedTemp.convertTo(warpped, (warpped.type() / 8) * 8);
}

int main_backup( int argc, char* argv[] );

int main( int argc, char* argv[] )
{
    gJpegParams.push_back( IMWRITE_JPEG_QUALITY );
    gJpegParams.push_back( 100 );

    CommandLineParser parser( argc, argv, keys );

    std::vector<std::string> imgFns;
    imgFns.push_back( "../Data/1458400804.000000.jpg" );
    imgFns.push_back( "../Data/1458400805.000000.jpg" );
    imgFns.push_back( "../Data/1458400806.000000.jpg" );
    imgFns.push_back( "../Data/1458400807.000000.jpg" );
    imgFns.push_back( "../Data/1458400808.000000.jpg" );
    imgFns.push_back( "../Data/1458400809.000000.jpg" );
    imgFns.push_back( "../Data/1458400810.000000.jpg" );
    imgFns.push_back( "../Data/1458400811.000000.jpg" );
    imgFns.push_back( "../Data/1458400812.000000.jpg" );

    std::vector<Mat> imgVec;
    std::vector<std::string>::iterator iterString;
    int imgCount = 0;
    for ( iterString = imgFns.begin(); iterString != imgFns.end(); iterString++ )
    {
        imgVec.push_back( imread( *iterString, IMREAD_GRAYSCALE ) );
        if ( imgVec.at(imgCount).empty() )
        {
            cout << "Could not open or find the image!\n" << endl;
            parser.printMessage();
            return -1;
        }

        imgCount++;
    }

    std::vector<Mat> Hs;

    PairwiseHomography ph;
    ph.mMinHessian = 5000;
    ph.compute( imgVec, Hs );
    imgVec.clear();

    cout << "Hs.size() = " << Hs.size() << endl;

    // Warp image.
    // Open the color images.
    std::vector<Mat> imgColorVec;
    for ( iterString = imgFns.begin(); iterString != imgFns.end(); iterString++ )
    {
        imgColorVec.push_back( imread( *iterString ) );
    }

    Mat HOrigin;
    Mat resImage = imgColorVec.at(0);

    HOrigin = Mat::eye(3, 3, CV_64FC1);

    for ( int i = 1; i < imgColorVec.size(); i++ )
    {
        warp_and_blend_images( resImage, imgColorVec.at(i), Hs.at(i-1), resImage, HOrigin );
        cout << "HOrigin = " << endl;
        cout << HOrigin << endl;
    }

    imwrite("ResImage.jpg", resImage, gJpegParams);

    waitKey();

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