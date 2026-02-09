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
  // double color;
  int total_pixels = img.rows * img.cols;
  int i;
  
  // Weights -> integers (divide by 256) B = 29, G = 150, R = 77 
  for (int i=0; i< total_pixels - 7; i+= 8) {
    uint8x8x3_t rgb = vld3_u8(&img.data[i*3]); // 210-RGB

    uint16x8_t r16 = vmovl_u8(rgb.val[2]); // to prevent overflow 16
    uint16x8_t g16 = vmovl_u8(rgb.val[1]);
    uint16x8_t b16 = vmovl_u8(rgb.val[0]);

    uint16x8_t gray16 = vmulq_n_u16(b16, 29);
    gray16 = vmlaq_n_u16(gray16, g16, 150);  // MA Ops
    gray16 = vmlaq_n_u16(gray16, r16, 77);

    uint8x8_t gray8 = vshrn_n_u16(gray16, 8);

    vst1_u8(&img_gray_out.data[i], gray8);
  }

  for(int i = 0; i < total_pixels; i++) {
    int color = (29 * img.data[i*3] + 
      150 * img.data[i*3 + 1] + 
      77 * img.data[i*3 + 2]) >> 8;
    img_gray_out.data[i] = color;
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
