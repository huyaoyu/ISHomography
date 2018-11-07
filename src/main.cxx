
#include <iostream>

#include <string>
#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>

using namespace std;
using namespace cv;

int main(void)
{
    cout << "Hello IST." << endl;

    vector<string> imgFns;
    imgFns.push_back( "../Data/1458400805.000000.jpg" );
    imgFns.push_back( "../Data/1458400806.000000.jpg" );

    // Load the original images.
    vector<Mat> imgOri;
    vector<string>::iterator iter;
    for ( iter = imgFns.begin(); iter != imgFns.end(); iter++ )
    {
        cout << *iter << endl;
        imgOri.push_back( imread( (*iter) ) );
    }

    // Show the image sizes.
    vector<Mat>::iterator iterMat;
    for ( iterMat = imgOri.begin(); iterMat != imgOri.end(); iterMat++ )
    {
        cout << (*iterMat).size << endl;
    }

    // Convert the original images into grayscale ones.
    vector<Mat> imgGray;
    Mat tempGray;
    for ( iterMat = imgOri.begin(); iterMat != imgOri.end(); iterMat++ )
    {
        cvtColor( (*iterMat), tempGray, COLOR_BGR2GRAY );
        imgGray.push_back( tempGray.clone() );
    }
    // Show the first image.
    namedWindow( "First gray", WINDOW_NORMAL );
    imshow( "First gray", imgGray.at(0) );
    // waitKey(0);

    // Smooth the images.
    vector<Mat> imgBlurred;
    for ( iterMat = imgGray.begin(); iterMat != imgGray.end(); iterMat++ )
    {
        GaussianBlur( (*iterMat), tempGray, Size( 7, 7 ), 0, 0, BORDER_CONSTANT);
        imgBlurred.push_back( tempGray.clone() );
    }
    namedWindow( "Blurred", WINDOW_NORMAL );
    imshow( "Blurred", imgBlurred.at(0) );
    cout << imgBlurred.at(0).size << endl;
    waitKey(0);

    Ptr<xfeatures2d::SURF> surfDetector = xfeatures2d::SURF::create( 1000, 2, 2, false, false );

    // Features and descriptors.
    vector<KeyPoint> keyPointVec;
    vector< vector<KeyPoint> > kpvv;
    vector<Mat> descriptorVec;
    Mat tempDescriptor;
    Ptr<xfeatures2d::LATCH> latch = xfeatures2d::LATCH::create(32, true, 3, 0.0);

    for ( iterMat = imgBlurred.begin(); iterMat != imgBlurred.end(); iterMat++ )
    {
        surfDetector->detect( (*iterMat), keyPointVec);
        latch->compute( (*iterMat), keyPointVec, tempDescriptor);
        kpvv.push_back( keyPointVec );
        keyPointVec.clear();
        descriptorVec.push_back( tempDescriptor.clone() );
    }

    cout << "Descriptors: " << endl;
    for ( iterMat = descriptorVec.begin(); iterMat != descriptorVec.end(); iterMat++ )
    {
        cout << (*iterMat).size << endl;
    }

    // Matcher.
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create( DescriptorMatcher::FLANNBASED );
    vector< vector<DMatch> > knnMatches;

    matcher->knnMatch( descriptorVec, knnMatches, 2);

    cout << knnMatches.size() << endl;

    return 0;
}
