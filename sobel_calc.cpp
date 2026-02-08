#include "opencv2/imgproc/imgproc.hpp"
#include "sobel_alg.h"
using namespace cv;

/*******************************************
 * Model: grayScale
 * Input: Mat img
 * Output: None directly. Modifies a ref parameter img_gray_out
 * Desc: This module converts the image to grayscale
 ********************************************/
void grayScale(Mat& img, Mat& img_gray_out)
{
  // double color;
  float color;
  
  // Convert to grayscale
  for (int i=0; i<img.rows; i++) {
    int i_step = STEP0*i;
    int j_step = 0;
    for (int j = 0; j<img.cols; j++, j_step += 3) {
      color = .114*img.data[i_step + j_step] +
              .587*img.data[i_step + j_step + 1] +
              .299*img.data[i_step + j_step + 2];
      img_gray_out.data[IMG_WIDTH*i + j] = color;
    }
  }
}

/*******************************************
 * Model: sobelCalc
 * Input: Mat img_in
 * Output: None directly. Modifies a ref parameter img_sobel_out
 * Desc: This module performs a sobel calculation on an image. It first
 *  converts the image to grayscale, calculates the gradient in the x
 *  direction, calculates the gradient in the y direction and sum it with Gx
 *  to finish the Sobel calculation
 ********************************************/

// EDITS: FUSED LOOP operations
void sobelCalc(Mat& img_gray, Mat& img_sobel_out)
{
  // Mat img_outx = img_gray.clone();
  // Mat img_outy = img_gray.clone();

  // Apply Sobel filter to black & white image
  unsigned short sobel_x, sobel_y, sobel_comb;

  // Calculate the x convolution
  for (int i=1; i<img_gray.rows; i++) {
    for (int j=1; j<img_gray.cols; j++) {
      sobel_x = abs(img_gray.data[IMG_WIDTH*(i-1) + (j-1)] -
		  img_gray.data[IMG_WIDTH*(i+1) + (j-1)] +
		  2*img_gray.data[IMG_WIDTH*(i-1) + (j)] -
		  2*img_gray.data[IMG_WIDTH*(i+1) + (j)] +
		  img_gray.data[IMG_WIDTH*(i-1) + (j+1)] -
		  img_gray.data[IMG_WIDTH*(i+1) + (j+1)]);

      sobel_x = (sobel_x > 255) ? 255 : sobel_x;
      // img_outx.data[IMG_WIDTH*(i) + (j)] = sobel_x;
      
      // y vonv
     sobel_y = abs(img_gray.data[IMG_WIDTH*(i-1) + (j-1)] -
		   img_gray.data[IMG_WIDTH*(i-1) + (j+1)] +
		   2*img_gray.data[IMG_WIDTH*(i) + (j-1)] -
		   2*img_gray.data[IMG_WIDTH*(i) + (j+1)] +
		   img_gray.data[IMG_WIDTH*(i+1) + (j-1)] -
		   img_gray.data[IMG_WIDTH*(i+1) + (j+1)]);
     sobel_y = (sobel_y > 255) ? 255 : sobel_y;

    //  img_outy.data[IMG_WIDTH*(i) + j] = sobel_y;
  
  // Combine the two convolutions into the output image
      sobel_comb = sobel_x + sobel_y;
      sobel_comb = (sobel_comb > 255) ? 255 : sobel_comb;
      img_sobel_out.data[IMG_WIDTH*(i) + j] = sobel_comb;
    }
  }
}
