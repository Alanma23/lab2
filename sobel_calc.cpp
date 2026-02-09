#include "opencv2/imgproc/imgproc.hpp"
#include "sobel_alg.h"
using namespace cv;

/*******************************************
 * Model: grayScale
 * Input: Mat img
 * Output: None directly. Modifies a ref parameter img_gray_out
 * Desc: This module converts the image to grayscale
 ********************************************/

 #include <arm_neon.h>
 
// void grayScale(Mat& img, Mat& img_gray_out)
// {
//   // double color;
//   float color;
  
//   // Convert to grayscale
//   for (int i=0; i<img.rows; i++) {
//     int i_step = STEP0*i;
//     int j_step = 0;
//     for (int j = 0; j<img.cols; j++, j_step += 3) {
//       color = .114*img.data[i_step + j_step] +
//               .587*img.data[i_step + j_step + 1] +
//               .299*img.data[i_step + j_step + 2];
//       img_gray_out.data[IMG_WIDTH*i + j] = color;
//     }
//   }
// }

void grayScale(Mat& img, Mat& img_gray_out, int start_row, int end_row)
{
  // process rows from start_row to end_row
  for (int i = start_row; i < end_row; i++) {
    int row_offset = i * img.cols;
    int rgb_offset = i * STEP0;
    int j;
    
    // Neon vectorized 8 pixels at a time
    for (j = 0; j < img.cols - 7; j += 8) {
      // Load 8 RGB triplets (24 bytes) using deinterleave
      uint8x8x3_t rgb = vld3_u8(&img.data[rgb_offset + j * 3]);
      
      uint16x8_t b16 = vmovl_u8(rgb.val[0]);
      uint16x8_t g16 = vmovl_u8(rgb.val[1]);
      uint16x8_t r16 = vmovl_u8(rgb.val[2]);
      
      uint16x8_t gray16 = vmulq_n_u16(b16, 29);
      gray16 = vmlaq_n_u16(gray16, g16, 150);  // mac
      gray16 = vmlaq_n_u16(gray16, r16, 77);
      uint8x8_t gray8 = vshrn_n_u16(gray16, 8);
      
      vst1_u8(&img_gray_out.data[row_offset + j], gray8);
    }
    
    // remaining pixels in this row (scalar)
    for (; j < img.cols; j++) {
      int color = (29 * img.data[rgb_offset + j*3] + 
                   150 * img.data[rgb_offset + j*3 + 1] + 
                   77 * img.data[rgb_offset + j*3 + 2]) >> 8;
      img_gray_out.data[row_offset + j] = color;
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
void sobelCalc(Mat& img_gray, Mat& img_sobel_out, int start_row, int end_row)
{
  // Apply Sobel filter to black & white image
  unsigned short sobel_x, sobel_y, sobel_comb;

  // Handle boundaries: Sobel needs i-1 and i+1
  int start_i = (start_row == 0) ? 1 : start_row;
  int end_i = (end_row >= img_gray.rows) ? img_gray.rows - 1 : end_row;

  // Calculate x and y convolutions, then combine (fused loop)
  for (int i = start_i; i < end_i; i++) {
    for (int j = 1; j < img_gray.cols - 1; j++) {
      // X gradient
      sobel_x = abs(img_gray.data[IMG_WIDTH*(i-1) + (j-1)] -
		  img_gray.data[IMG_WIDTH*(i+1) + (j-1)] +
		  2*img_gray.data[IMG_WIDTH*(i-1) + (j)] -
		  2*img_gray.data[IMG_WIDTH*(i+1) + (j)] +
		  img_gray.data[IMG_WIDTH*(i-1) + (j+1)] -
		  img_gray.data[IMG_WIDTH*(i+1) + (j+1)]);
      sobel_x = (sobel_x > 255) ? 255 : sobel_x;
      
      // Y gradient
      sobel_y = abs(img_gray.data[IMG_WIDTH*(i-1) + (j-1)] -
		   img_gray.data[IMG_WIDTH*(i-1) + (j+1)] +
		   2*img_gray.data[IMG_WIDTH*(i) + (j-1)] -
		   2*img_gray.data[IMG_WIDTH*(i) + (j+1)] +
		   img_gray.data[IMG_WIDTH*(i+1) + (j-1)] -
		   img_gray.data[IMG_WIDTH*(i+1) + (j+1)]);
      sobel_y = (sobel_y > 255) ? 255 : sobel_y;
  
      // Combine the two convolutions
      sobel_comb = sobel_x + sobel_y;
      sobel_comb = (sobel_comb > 255) ? 255 : sobel_comb;
      img_sobel_out.data[IMG_WIDTH*i + j] = sobel_comb;
    }
  }
}

// zhikai's code
void sobelCalc(Mat& img_gray, Mat& img_sobel_out)
{
  unsigned short sobelx, sobely, sobel;

  uchar* prev_row;
  uchar* curr_row;
  uchar* next_row;

// (i-1,j-1)  (i-1,j)  (i-1,j+1)
// (i  ,j-1)  (i  ,j)  (i  ,j+1)
// (i+1,j-1)  (i+1,j)  (i+1,j+1)

  // Optimized convolution
  for (int i=1; i<img_gray.rows-1; i++) {

    prev_row = image_gray.ptr<uchar>(i-1);
    curr_row = image_gray.ptr<uchar>(i);
    next_row = image_gray.ptr<uchar>(i+1);

    for (int j=1; j<img_gray.cols-1; j++) { // this loop hits an individual pixel
      // sobel math
      int right = j+1;
      int left = j-1;
      sobelx = abs(prev_row[right] -
		              prev_row[left] +
		            2*curr_row[right] -
		            2*curr_row[left] +
		              next_row[right] -
		              next_row[left]);

      sobely = abs(-prev_row[left] -
		              2*prev_row[j] -
		                prev_row[right] +
		                next_row[left] +
		              2*next_row[j] +
		                next_row[right]);
      
      sobel = sobelx + sobely;              // combine the two
      sobel = (sobel > 255) ? 255 : sobel;  // check upper bound
      img_sobel_out.data[IMG_WIDTH*(i) + j] = sobel;
    }
  }
}
