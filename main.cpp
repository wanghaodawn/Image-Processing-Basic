/*
 *
 * 15462 - Assignment 5 - Image Processing Basis
 *
 * Hao Wang
 * Andrew ID: haow2
 *
 * 
 * This assignment includes basic image processing algorithms.
 * The assignment needs opencv library to read images, I wrote algorithms without using libraries.
 *
 * Also compared the performance among algorithms with similar functions in the same image.
 *
 * 
 * Procedures to run the project:
 * 1) mkdir build && cd build
 * 2) cmake ..
 * 3) make
 * 4) ./asst5_imageprocessing ../images/image1.jpg
 *
 * After successfully run the image, just press any key to go to next scene.
 *
 *
 * Main Tasks:
 *
 * Task 1: RGB to Grey
 * Task 2: Filter - Median Filter
 * Task 3: Filter - Mean Filter
 * Task 4: Blur - Box Blur
 * Task 5: Blur - Gaussian Blur
 * Task 6: Edge Detection - Sobel
 * Task 7: Edge Detection - Prewitt
 * Task 8: Edge Detection - Laplacian
 * Task 9: Image Enhancement - Histrogram Equalization
 *
 *
 * PS: It seems that it has some problem to read .png files, but it works on JPG, jpg, bmp etc.
 *
 */


// Header
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <math.h>


// Only use the library to read and write the image
// All implementation are finished by C++ manually


// Name space
using namespace cv;
using namespace std;

#define blurOptionsNum 6



// Task 1: RGB to Grey
// Use this function to chagne RGB image to grey image
void rgb_to_grey(Mat& image, int height, int width, int channel) {
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            for (int k = 0; k < channel; k++) {
                float grey = 0.f;

                grey  = 0.299 * image.at<cv::Vec3b>(i,j)[0];
                grey += 0.587 * image.at<cv::Vec3b>(i,j)[1];
                grey += 0.114 * image.at<cv::Vec3b>(i,j)[2];

                image.at<cv::Vec3b>(i,j)[0] = grey;
                image.at<cv::Vec3b>(i,j)[1] = grey;
                image.at<cv::Vec3b>(i,j)[2] = grey;
            }
        }
    }
}



// Use this function to display an image
void show_image(Mat& image, String string) {
    // Create a window
    namedWindow(string, WINDOW_AUTOSIZE );
    // Display the original image in the window
    imshow(string, image);
    // Wait for keys to be presses
    waitKey(0);
    // Destroy the window
    destroyWindow(string);
}



// Helper function in task 2
// Get the median element from the given array
int get_median(int a[]) {
    // Sort the array
    int temp;
    bool flag = false;
    for (int i = 0;i < 9; i++) {
        flag = true;
        for (int j = 0; j < 8-i; j++) {
            if(a[j] > a[j+1]) {
                temp = a[j];
                a[j] = a[j+1];
                a[j+1] = temp;
                flag = false;
            }
        }
        if(flag) 
            break;
    }
    // return the median element
    return a[4];
}



// Task 2: Filter - Median Filter
void median_filter(Mat& image, int height, int width, int channel, bool showImage) {

    Mat image_median = image.clone();
    for (int i = 1; i < height-1; i++) {
        for (int j = 1; j < width-1; j++) {
            for (int k = 0; k < channel; k++) {
                int temp[9];
                temp[0] = image.at<cv::Vec3b>(i-1,j-1)[k];
                temp[1] = image.at<cv::Vec3b>(i-1,j)[k];
                temp[2] = image.at<cv::Vec3b>(i-1,j+1)[k];

                temp[3] = image.at<cv::Vec3b>(i,j-1)[k];
                temp[4] = image.at<cv::Vec3b>(i,j)[k];
                temp[5] = image.at<cv::Vec3b>(i,j+1)[k];

                temp[6] = image.at<cv::Vec3b>(i+1,j-1)[k];
                temp[7] = image.at<cv::Vec3b>(i+1,j)[k];
                temp[8] = image.at<cv::Vec3b>(i+1,j+1)[k];

                image_median.at<cv::Vec3b>(i,j)[k] = get_median(temp);
            }
        }
    }
    if (showImage) {
        // Show Image
        show_image(image_median, "Median Filter");
    } else {
        image = image_median;
    }
}



// Task 3: Filter - Mean Filter
void mean_filter(Mat& image, int height, int width, int channel, bool showImage) {
    
    Mat image_mean = image.clone();
    for (int i = 1; i < height-1; i++) {
        for (int j = 1; j < width-1; j++) {
            for (int k = 0; k < channel; k++) {
                float temp = 0.f;
                temp += image.at<cv::Vec3b>(i-1,j-1)[k];
                temp += image.at<cv::Vec3b>(i-1,j)[k];
                temp += image.at<cv::Vec3b>(i-1,j+1)[k];

                temp += image.at<cv::Vec3b>(i,j-1)[k];
                temp += image.at<cv::Vec3b>(i,j)[k];
                temp += image.at<cv::Vec3b>(i,j+1)[k];

                temp += image.at<cv::Vec3b>(i+1,j-1)[k];
                temp += image.at<cv::Vec3b>(i+1,j)[k];
                temp += image.at<cv::Vec3b>(i+1,j+1)[k];

                image_mean.at<cv::Vec3b>(i,j)[k] = (int)(temp / 9 + 0.5);
            }
        }
    }
    if (showImage) {
        // Show Image
        show_image(image_mean, "Mean Filter");
    } else {
        image = image_mean;
    }
}



// Task 4-1: Blur - Box Blur (5 * 5)
void box_blur_general(Mat& image, int height, int width, int channel, bool showImage) {

    Mat image_box_blur_g = image.clone();
    for (int i = 2; i < height-2; i++) {
        for (int j = 2; j < width-2; j++) {
            for (int k = 0; k < channel; k++) {
                float temp = 0.f;
                temp += image.at<cv::Vec3b>(i-2,j-2)[k];
                temp += image.at<cv::Vec3b>(i-2,j-1)[k];
                temp += image.at<cv::Vec3b>(i-2,j)[k];
                temp += image.at<cv::Vec3b>(i-2,j+1)[k];
                temp += image.at<cv::Vec3b>(i-2,j+2)[k];

                temp += image.at<cv::Vec3b>(i-1,j-2)[k];
                temp += image.at<cv::Vec3b>(i-1,j-1)[k];
                temp += image.at<cv::Vec3b>(i-1,j)[k];
                temp += image.at<cv::Vec3b>(i-1,j+1)[k];
                temp += image.at<cv::Vec3b>(i-1,j+2)[k];

                temp += image.at<cv::Vec3b>(i,j-2)[k];
                temp += image.at<cv::Vec3b>(i,j-1)[k];
                temp += image.at<cv::Vec3b>(i,j)[k];
                temp += image.at<cv::Vec3b>(i,j+1)[k];
                temp += image.at<cv::Vec3b>(i,j+2)[k];

                temp += image.at<cv::Vec3b>(i+1,j-2)[k];
                temp += image.at<cv::Vec3b>(i+1,j-1)[k];
                temp += image.at<cv::Vec3b>(i+1,j)[k];
                temp += image.at<cv::Vec3b>(i+1,j+1)[k];
                temp += image.at<cv::Vec3b>(i+1,j+2)[k];

                temp += image.at<cv::Vec3b>(i+2,j-2)[k];
                temp += image.at<cv::Vec3b>(i+2,j-1)[k];
                temp += image.at<cv::Vec3b>(i+2,j)[k];
                temp += image.at<cv::Vec3b>(i+2,j+1)[k];
                temp += image.at<cv::Vec3b>(i+2,j+2)[k];

                image_box_blur_g.at<cv::Vec3b>(i,j)[k] = (int)(temp / 25 + 0.5);
            }
        }
    }
    if (showImage) {
        // Show Image
        show_image(image_box_blur_g, "General Box Blur 5 * 5");
    } else {
        image = image_box_blur_g;
    }
}



// Task 4-2: Blur - Vertical Box Blur (15 * 3)
void box_blur_vertical(Mat& image, int height, int width, int channel, bool showImage) {

    Mat image_box_blur_v = image.clone();
    for (int i = 7; i < height-7; i++) {
        for (int j = 1; j < width-1; j++) {
            for (int k = 0; k < channel; k++) {
                float temp = 0.f;
                temp += image.at<cv::Vec3b>(i-7,j-1)[k];
                temp += image.at<cv::Vec3b>(i-7,j)[k];
                temp += image.at<cv::Vec3b>(i-7,j+1)[k];

                temp += image.at<cv::Vec3b>(i-6,j-1)[k];
                temp += image.at<cv::Vec3b>(i-6,j)[k];
                temp += image.at<cv::Vec3b>(i-6,j+1)[k];

                temp += image.at<cv::Vec3b>(i-5,j-1)[k];
                temp += image.at<cv::Vec3b>(i-5,j)[k];
                temp += image.at<cv::Vec3b>(i-5,j+1)[k];

                temp += image.at<cv::Vec3b>(i-4,j-1)[k];
                temp += image.at<cv::Vec3b>(i-4,j)[k];
                temp += image.at<cv::Vec3b>(i-4,j+1)[k];

                temp += image.at<cv::Vec3b>(i-3,j-1)[k];
                temp += image.at<cv::Vec3b>(i-3,j)[k];
                temp += image.at<cv::Vec3b>(i-3,j+1)[k];

                temp += image.at<cv::Vec3b>(i-2,j-1)[k];
                temp += image.at<cv::Vec3b>(i-2,j)[k];
                temp += image.at<cv::Vec3b>(i-2,j+1)[k];

                temp += image.at<cv::Vec3b>(i-1,j-1)[k];
                temp += image.at<cv::Vec3b>(i-1,j)[k];
                temp += image.at<cv::Vec3b>(i-1,j+1)[k];

                temp += image.at<cv::Vec3b>(i,j-1)[k];
                temp += image.at<cv::Vec3b>(i,j)[k];
                temp += image.at<cv::Vec3b>(i,j+1)[k];

                temp += image.at<cv::Vec3b>(i+1,j-1)[k];
                temp += image.at<cv::Vec3b>(i+1,j)[k];
                temp += image.at<cv::Vec3b>(i+1,j+1)[k];

                temp += image.at<cv::Vec3b>(i+2,j-1)[k];
                temp += image.at<cv::Vec3b>(i+2,j)[k];
                temp += image.at<cv::Vec3b>(i+2,j+1)[k];

                temp += image.at<cv::Vec3b>(i+3,j-1)[k];
                temp += image.at<cv::Vec3b>(i+3,j)[k];
                temp += image.at<cv::Vec3b>(i+3,j+1)[k];

                temp += image.at<cv::Vec3b>(i+4,j-1)[k];
                temp += image.at<cv::Vec3b>(i+4,j)[k];
                temp += image.at<cv::Vec3b>(i+4,j+1)[k];

                temp += image.at<cv::Vec3b>(i+5,j-1)[k];
                temp += image.at<cv::Vec3b>(i+5,j)[k];
                temp += image.at<cv::Vec3b>(i+5,j+1)[k];

                temp += image.at<cv::Vec3b>(i+6,j-1)[k];
                temp += image.at<cv::Vec3b>(i+6,j)[k];
                temp += image.at<cv::Vec3b>(i+6,j+1)[k];

                temp += image.at<cv::Vec3b>(i+7,j-1)[k];
                temp += image.at<cv::Vec3b>(i+7,j)[k];
                temp += image.at<cv::Vec3b>(i+7,j+1)[k];

                image_box_blur_v.at<cv::Vec3b>(i,j)[k] = (int)(temp / 45 + 0.5);
            }
        }
    }
    if (showImage) {
        // Show Image
        show_image(image_box_blur_v, "Vertical Box Blur 15 * 3");
    } else {
        image = image_box_blur_v;
    }
}



// Task 4-3: Blur - Horizontal Box Blur (3 * 15)
void box_blur_horizontal(Mat& image, int height, int width, int channel, bool showImage) {

    Mat image_box_blur_h = image.clone();
    for (int i = 1; i < height-1; i++) {
        for (int j = 7; j < width-7; j++) {
            for (int k = 0; k < channel; k++) {
                float temp = 0.f;
                temp += image.at<cv::Vec3b>(i-1,j-7)[k];
                temp += image.at<cv::Vec3b>(i-1,j-6)[k];
                temp += image.at<cv::Vec3b>(i-1,j-5)[k];
                temp += image.at<cv::Vec3b>(i-1,j-4)[k];
                temp += image.at<cv::Vec3b>(i-1,j-3)[k];
                temp += image.at<cv::Vec3b>(i-1,j-2)[k];
                temp += image.at<cv::Vec3b>(i-1,j-1)[k];
                temp += image.at<cv::Vec3b>(i-1,j)[k];
                temp += image.at<cv::Vec3b>(i-1,j+1)[k];
                temp += image.at<cv::Vec3b>(i-1,j+2)[k];
                temp += image.at<cv::Vec3b>(i-1,j+3)[k];
                temp += image.at<cv::Vec3b>(i-1,j+4)[k];
                temp += image.at<cv::Vec3b>(i-1,j+5)[k];
                temp += image.at<cv::Vec3b>(i-1,j+6)[k];
                temp += image.at<cv::Vec3b>(i-1,j+7)[k];

                temp += image.at<cv::Vec3b>(i,j-7)[k];
                temp += image.at<cv::Vec3b>(i,j-6)[k];
                temp += image.at<cv::Vec3b>(i,j-5)[k];
                temp += image.at<cv::Vec3b>(i,j-4)[k];
                temp += image.at<cv::Vec3b>(i,j-3)[k];
                temp += image.at<cv::Vec3b>(i,j-2)[k];
                temp += image.at<cv::Vec3b>(i,j-1)[k];
                temp += image.at<cv::Vec3b>(i,j)[k];
                temp += image.at<cv::Vec3b>(i,j+1)[k];
                temp += image.at<cv::Vec3b>(i,j+2)[k];
                temp += image.at<cv::Vec3b>(i,j+3)[k];
                temp += image.at<cv::Vec3b>(i,j+4)[k];
                temp += image.at<cv::Vec3b>(i,j+5)[k];
                temp += image.at<cv::Vec3b>(i,j+6)[k];
                temp += image.at<cv::Vec3b>(i,j+7)[k];

                temp += image.at<cv::Vec3b>(i+1,j-7)[k];
                temp += image.at<cv::Vec3b>(i+1,j-6)[k];
                temp += image.at<cv::Vec3b>(i+1,j-5)[k];
                temp += image.at<cv::Vec3b>(i+1,j-4)[k];
                temp += image.at<cv::Vec3b>(i+1,j-3)[k];
                temp += image.at<cv::Vec3b>(i+1,j-2)[k];
                temp += image.at<cv::Vec3b>(i+1,j-1)[k];
                temp += image.at<cv::Vec3b>(i+1,j)[k];
                temp += image.at<cv::Vec3b>(i+1,j+1)[k];
                temp += image.at<cv::Vec3b>(i+1,j+2)[k];
                temp += image.at<cv::Vec3b>(i+1,j+3)[k];
                temp += image.at<cv::Vec3b>(i+1,j+4)[k];
                temp += image.at<cv::Vec3b>(i+1,j+5)[k];
                temp += image.at<cv::Vec3b>(i+1,j+6)[k];
                temp += image.at<cv::Vec3b>(i+1,j+7)[k];

                image_box_blur_h.at<cv::Vec3b>(i,j)[k] = (int)(temp / 45 + 0.5);
            }
        }
    }
    if (showImage) {
        // Show Image
        show_image(image_box_blur_h, "Horizontal Box Blur 3 * 15");
    } else {
        image = image_box_blur_h;
    }
}



// Task 5: Blur - Gaussian Blur
void gaussian_blur(Mat& image, int height, int width, int channel, bool showImage) {

    float gaussian[] = {0.075, 0.124, 0.075,
                        0.124, 0.204, 0.124,
                        0.075, 0.124, 0.075};
    Mat image_gaussian = image.clone();
    for (int i = 1; i < height-1; i++) {
        for (int j = 1; j < width-1; j++) {
            for (int k = 0; k < channel; k++) {
                float temp = 0.f;
                temp += gaussian[0] * image.at<cv::Vec3b>(i-1,j-1)[k];
                temp += gaussian[1] * image.at<cv::Vec3b>(i-1,j)[k];
                temp += gaussian[2] * image.at<cv::Vec3b>(i-1,j+1)[k];

                temp += gaussian[3] * image.at<cv::Vec3b>(i,j-1)[k];
                temp += gaussian[4] * image.at<cv::Vec3b>(i,j)[k];
                temp += gaussian[5] * image.at<cv::Vec3b>(i,j+1)[k];

                temp += gaussian[6] * image.at<cv::Vec3b>(i+1,j-1)[k];
                temp += gaussian[7] * image.at<cv::Vec3b>(i+1,j)[k];
                temp += gaussian[8] * image.at<cv::Vec3b>(i+1,j+1)[k];

                image_gaussian.at<cv::Vec3b>(i,j)[k] = (int)(temp + 0.5);
            }
        }
    }
    if (showImage) {
        // Show Image
        show_image(image_gaussian, "Gaussian Blur");
    } else {
        image = image_gaussian;
    }
}



// Task 6: Edge Detection - Sobel Edge Detection
void sobel_edge_detection(Mat& image_old, int height, int width, int channel, int blurOption) {

    Mat image = image_old.clone();
    // Choose blur options
    switch (blurOption) {
        case 1:
            mean_filter(image, height, width, channel, false);
            break;
        case 2:
            median_filter(image, height, width, channel, false);
            break;
        case 3:
            box_blur_horizontal(image, height, width, channel, false);
            break;
        case 4:
            box_blur_vertical(image, height, width, channel, false);
            break;
        case 5:
            box_blur_general(image, height, width, channel, false);
            break;
        case 6:
            gaussian_blur(image, height, width, channel, false);
        default:
            break;
    }

    Mat image_sobel = image.clone();
    for (int i = 1; i < height-1; i++) {
        for (int j = 1; j < width-1; j++) {
            for (int k = 0; k < channel; k++) {
                float gx = 0.f, gy = 0.f;
                gx -= image.at<cv::Vec3b>(i-1,j-1)[k];
                gx -= 2 * image.at<cv::Vec3b>(i-1,j)[k];
                gx -= image.at<cv::Vec3b>(i-1,j+1)[k];

                gx += image.at<cv::Vec3b>(i+1,j-1)[k];
                gx += 2 * image.at<cv::Vec3b>(i+1,j)[k];
                gx += image.at<cv::Vec3b>(i+1,j+1)[k];

                gy -= image.at<cv::Vec3b>(i-1,j-1)[k];
                gy -= 2 * image.at<cv::Vec3b>(i,j-1)[k];
                gy -= image.at<cv::Vec3b>(i+1,j-1)[k];

                gy += image.at<cv::Vec3b>(i-1,j+1)[k];
                gy += 2 * image.at<cv::Vec3b>(i,j+1)[k];
                gy += image.at<cv::Vec3b>(i+1,j+1)[k];
                
                // Use abs(gx) + abs(gy) to approximate sqrt(gx^2 + gy^2)
                image_sobel.at<cv::Vec3b>(i,j)[k] = (int)(abs(gx) + abs(gy) + 0.5);
            }
        }
    }

    // RGB To Grey
    rgb_to_grey(image_sobel, height, width, channel);

    // Choose Image Display Options
    switch (blurOption) {
        case 1:
            show_image(image_sobel, "Sobel Edge Detection After Median Filter");
            break;
        case 2:
            show_image(image_sobel, "Sobel Edge Detection After Mean Filter");
            break;
        case 3:
            show_image(image_sobel, "Sobel Edge Detection After Horizontal Box Blur");
            break;
        case 4:
            show_image(image_sobel, "Sobel Edge Detection After Vertical Box Blur");
            break;
        case 5:
            show_image(image_sobel, "Sobel Edge Detection After General Box Blur");
            break;
        case 6:
            show_image(image_sobel, "Sobel Edge Detection After Gaussian Blur");
            break;
        default:
            show_image(image_sobel, "Sobel Edge Detection");
            break;
    }
}



// Task 7: Edge Detection - Prewitt Edge Detection
void prewitt_edge_detection(Mat& image_old, int height, int width, int channel, int blurOption) {

    Mat image = image_old.clone();
    // Choose blur options
    switch (blurOption) {
        case 1:
            mean_filter(image, height, width, channel, false);
            break;
        case 2:
            median_filter(image, height, width, channel, false);
            break;
        case 3:
            box_blur_horizontal(image, height, width, channel, false);
            break;
        case 4:
            box_blur_vertical(image, height, width, channel, false);
            break;
        case 5:
            box_blur_general(image, height, width, channel, false);
            break;
        case 6:
            gaussian_blur(image, height, width, channel, false);
        default:
            break;
    }

    Mat image_prewitt = image.clone();
    for (int i = 1; i < height-1; i++) {
        for (int j = 1; j < width-1; j++) {
            for (int k = 0; k < channel; k++) {
                float gx = 0.f, gy = 0.f;
                gx -= image.at<cv::Vec3b>(i-1,j-1)[k];
                gx -= image.at<cv::Vec3b>(i-1,j)[k];
                gx -= image.at<cv::Vec3b>(i-1,j+1)[k];

                gx += image.at<cv::Vec3b>(i+1,j-1)[k];
                gx += image.at<cv::Vec3b>(i+1,j)[k];
                gx += image.at<cv::Vec3b>(i+1,j+1)[k];

                gy -= image.at<cv::Vec3b>(i-1,j-1)[k];
                gy -= image.at<cv::Vec3b>(i,j-1)[k];
                gy -= image.at<cv::Vec3b>(i+1,j-1)[k];

                gy += image.at<cv::Vec3b>(i-1,j+1)[k];
                gy += image.at<cv::Vec3b>(i,j+1)[k];
                gy += image.at<cv::Vec3b>(i+1,j+1)[k];

                // Use abs(gx) + abs(gy) to approximate sqrt(gx^2 + gy^2)
                image_prewitt.at<cv::Vec3b>(i,j)[k] = (int)(abs(gx) + abs(gy) + 0.5);
            }
        }
    }

    // RGB To Grey
    rgb_to_grey(image_prewitt, height, width, channel);

    // Choose Image Display Options
    switch (blurOption) {
        case 1:
            show_image(image_prewitt, "Prewitt Edge Detection After Median Filter");
            break;
        case 2:
            show_image(image_prewitt, "Prewitt Edge Detection After Mean Filter");
            break;
        case 3:
            show_image(image_prewitt, "Prewitt Edge Detection After Horizontal Box Blur");
            break;
        case 4:
            show_image(image_prewitt, "Prewitt Edge Detection After Vertical Box Blur");
            break;
        case 5:
            show_image(image_prewitt, "Prewitt Edge Detection After General Box Blur");
            break;
        case 6:
            show_image(image_prewitt, "Prewitt Edge Detection After Gaussian Blur");
            break;
        default:
            show_image(image_prewitt, "Prewitt Edge Detection");
            break;
    }
}



// Task 8-1: Edge Detection - Laplacian Edge Detection Type A
// The parameter is: -1, -1, -1, -1, 4
void laplacian_edge_detection_1(Mat& image_old, int height, int width, int channel, int blurOption) {

    Mat image = image_old.clone();
    // Choose blur options
    switch (blurOption) {
        case 1:
            mean_filter(image, height, width, channel, false);
            break;
        case 2:
            median_filter(image, height, width, channel, false);
            break;
        case 3:
            box_blur_horizontal(image, height, width, channel, false);
            break;
        case 4:
            box_blur_vertical(image, height, width, channel, false);
            break;
        case 5:
            box_blur_general(image, height, width, channel, false);
            break;
        case 6:
            gaussian_blur(image, height, width, channel, false);
        default:
            break;
    }

    Mat image_laplacian_1 = image.clone();
    for (int i = 1; i < height-1; i++) {
        for (int j = 1; j < width-1; j++) {
            for (int k = 0; k < channel; k++) {
                int temp = 0;
                temp -= image.at<cv::Vec3b>(i-1,j)[k];
                temp -= image.at<cv::Vec3b>(i,j-1)[k];
                temp -= image.at<cv::Vec3b>(i,j+1)[k];
                temp -= image.at<cv::Vec3b>(i+1,j)[k];
                temp += 4 * image.at<cv::Vec3b>(i,j)[k];

                image_laplacian_1.at<cv::Vec3b>(i,j)[k] = temp;
            }
        }
    }

    // RGB To Grey
    rgb_to_grey(image_laplacian_1, height, width, channel);

    // Choose Image Display Options
    switch (blurOption) {
        case 1:
            show_image(image_laplacian_1, "Laplacian Edge Detection Type A After Median Filter");
            break;
        case 2:
            show_image(image_laplacian_1, "Laplacian Edge Detection Type A After Mean Filter");
            break;
        case 3:
            show_image(image_laplacian_1, "Laplacian Edge Detection Type A After Horizontal Box Blur");
            break;
        case 4:
            show_image(image_laplacian_1, "Laplacian Edge Detection Type A After Vertical Box Blur");
            break;
        case 5:
            show_image(image_laplacian_1, "Laplacian Edge Detection Type A After General Box Blur");
            break;
        case 6:
            show_image(image_laplacian_1, "Laplacian Edge Detection Type A After Gaussian Blur");
            break;
        default:
            show_image(image_laplacian_1, "Laplacian Edge Detection Type A");
            break;
    }
}



// Task 8-2: Edge Detection - Laplacian Edge Detection Type B
// The parameter is: 1, 1, 1, 1, -4
void laplacian_edge_detection_2(Mat& image_old, int height, int width, int channel, int blurOption) {
    
    Mat image = image_old.clone();
    // Choose blur options
    switch (blurOption) {
        case 1:
            mean_filter(image, height, width, channel, false);
            break;
        case 2:
            median_filter(image, height, width, channel, false);
            break;
        case 3:
            box_blur_horizontal(image, height, width, channel, false);
            break;
        case 4:
            box_blur_vertical(image, height, width, channel, false);
            break;
        case 5:
            box_blur_general(image, height, width, channel, false);
            break;
        case 6:
            gaussian_blur(image, height, width, channel, false);
        default:
            break;
    }

    Mat image_laplacian_2 = image.clone();
    for (int i = 1; i < height-1; i++) {
        for (int j = 1; j < width-1; j++) {
            for (int k = 0; k < channel; k++) {
                int temp = 0;
                temp += image.at<cv::Vec3b>(i-1,j)[k];
                temp += image.at<cv::Vec3b>(i,j-1)[k];
                temp += image.at<cv::Vec3b>(i,j+1)[k];
                temp += image.at<cv::Vec3b>(i+1,j)[k];
                temp -= 4 * image.at<cv::Vec3b>(i,j)[k];

                image_laplacian_2.at<cv::Vec3b>(i,j)[k] = temp;
            }
        }
    }

    // RGB To Grey
    rgb_to_grey(image_laplacian_2, height, width, channel);

    // Choose Image Display Options
    switch (blurOption) {
        case 1:
            show_image(image_laplacian_2, "Laplacian Edge Detection Type B After Median Filter");
            break;
        case 2:
            show_image(image_laplacian_2, "Laplacian Edge Detection Type B After Mean Filter");
            break;
        case 3:
            show_image(image_laplacian_2, "Laplacian Edge Detection Type B After Horizontal Box Blur");
            break;
        case 4:
            show_image(image_laplacian_2, "Laplacian Edge Detection Type B After Vertical Box Blur");
            break;
        case 5:
            show_image(image_laplacian_2, "Laplacian Edge Detection Type B After General Box Blur");
            break;
        case 6:
            show_image(image_laplacian_2, "Laplacian Edge Detection Type B After Gaussian Blur");
            break;
        default:
            show_image(image_laplacian_2, "Laplacian Edge Detection Type B");
            break;
    }
}



// Task 9: Image Enhancement - Histrogram Equalization
void histo_equal(Mat& image, int height, int width, int channel, bool showImage, bool toGrey) {
    
    Mat image_histo = image.clone();

    float r[256], g[256], b[256];
    float rr[256], gg[256], bb[256];
    int size = width * height;

    // Array Initialization
    for (int i = 0; i < 256; i++) {
        r[i] = 0.f;
        g[i] = 0.f;
        b[i] = 0.f;
    }

    // Compute the distribution of colors
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            r[image.at<cv::Vec3b>(i,j)[0]]++;
            g[image.at<cv::Vec3b>(i,j)[1]]++;
            b[image.at<cv::Vec3b>(i,j)[2]]++;
        }
    }

    // Change number to possibility
    for (int i = 0; i < 256; i++) {
        r[i] /= size;
        g[i] /= size;
        b[i] /= size;
    }

    // Compute Cumulative Possibility
    rr[0] = r[0];
    gg[0] = g[0];
    bb[0] = b[0];
    for (int i = 1; i < 256; i++) {
        rr[i] = rr[i-1] + r[i];
        gg[i] = gg[i-1] + g[i];
        bb[i] = bb[i-1] + b[i];
    }
    for (int i = 0; i < 256; i++) {
        rr[i] = (int) (255 * rr[i] + 0.5);
        gg[i] = (int) (255 * gg[i] + 0.5);
        bb[i] = (int) (255 * bb[i] + 0.5);
    }

    // Convert
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            image_histo.at<cv::Vec3b>(i,j)[0] = rr[image.at<cv::Vec3b>(i,j)[0]];
            image_histo.at<cv::Vec3b>(i,j)[1] = gg[image.at<cv::Vec3b>(i,j)[1]];
            image_histo.at<cv::Vec3b>(i,j)[2] = bb[image.at<cv::Vec3b>(i,j)[2]];
        }
    }

    // Different options
    if (toGrey) {
        // RGB To Grey
        rgb_to_grey(image_histo, height, width, channel);

        if (showImage) {
            // Show Image
            show_image(image_histo, "Histrogram Equalization in Grey");
        } else {
            image = image_histo;
        }
    } else {
        // RGB
        if (showImage) {
            // Show Image
            show_image(image_histo, "Histrogram Equalization in RGB");
        } else {
            image = image_histo;
        }
    }
}



int main(int argc, char** argv ) {   
    // Illegal Input
    if ( argc != 2 ) {
        printf("usage: DisplayImage.out <Image_Path>\n");
        return -1;
    }

    // Get the image
    Mat image;
    image = imread( argv[1], 1 );
    
    // If no image data
    if ( !image.data ) {
        printf("No image data \n");
        return -1;
    }

    int height = image.rows;
    int width  = image.cols;
    int channel = image.channels();
    cout << "height: " << height << endl;
    cout << "width: " << width << endl;

    // uchar* data = image.ptr<uchar>(0);

    // Show the original image first
    show_image(image, "Original Image");
    
    // Show the result of histrogram equalization with GRB
    histo_equal(image, height, width, channel, true, false);

    // RGB to Grey
    Mat image_grey = image.clone();
    rgb_to_grey(image_grey, height, width, channel);
    show_image(image_grey, "RGB to Grey");

    // Show the result of histrogram equalization with grey
    histo_equal(image, height, width, channel, true, true);

    // Filter
    median_filter(image, height, width, channel, true);
    mean_filter(image, height, width, channel, true);

    // Blur
    box_blur_horizontal(image, height, width, channel, true);
    box_blur_vertical(image, height, width, channel, true);
    box_blur_general(image, height, width, channel, true);
    gaussian_blur(image, height, width, channel, true);

    // Edge Detection
    for (int blurOption = 0; blurOption <= blurOptionsNum; blurOption++)
        sobel_edge_detection(image, height, width, channel, blurOption);

    for (int blurOption = 0; blurOption <= blurOptionsNum; blurOption++)
        prewitt_edge_detection(image, height, width, channel, blurOption);

    for (int blurOption = 0; blurOption <= blurOptionsNum; blurOption++)
        laplacian_edge_detection_1(image, height, width, channel, blurOption);

    for (int blurOption = 0; blurOption <= blurOptionsNum; blurOption++)
        laplacian_edge_detection_2(image, height, width, channel, blurOption);

    // Show the original image last
    show_image(image, "Original Image");

    return 1;
}

