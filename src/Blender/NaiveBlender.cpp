
#include "Blender/NaiveBlender.hpp"

using std::cout;
using std::endl;

using namespace cv;

NaiveBlender::NaiveBlender()
{

}

NaiveBlender::~NaiveBlender()
{

}

static
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

void NaiveBlender::warp_and_blend( InputArray _base, InputArray _new, InputArray _HNew, OutputArray _warpped, InputOutputArray _HOrigin )
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

    // This is found at
    // https://stackoverflow.com/questions/30044477/how-to-use-multi-band-blender-in-opencv
    warppedTemp.convertTo(warpped, (warpped.type() / 8) * 8);
}
