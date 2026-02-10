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

void sobelCalc(Mat& img_gray, Mat& img_sobel_out, int start_row, int end_row)
{
  // int start_i = (start_row == 0) ? 1 : start_row;
  // int end_i = (end_row >= img_gray.rows) ? img_gray.rows - 1 : end_row;
  int start_i = start_row;
  int end_i = end_row;

  // base pointers
  unsigned char* img_data = img_gray.data;
  unsigned char* out_data = img_sobel_out.data;

  // Process rows
  for (int i = start_i + 1; i < end_i - 1; i++) {
    unsigned char* prev_row = img_data + IMG_WIDTH * (i - 1);
    unsigned char* curr_row = img_data + IMG_WIDTH * i;
    unsigned char* next_row = img_data + IMG_WIDTH * (i + 1);
    unsigned char* out_row = out_data + IMG_WIDTH * i;
    
    int j;
    
    // 8 pixels at a time
    for (j = 1; j < img_gray.cols - 8; j += 8) {
      // Load 16 pixels from each row
      uint8x16_t prev = vld1q_u8(prev_row + j - 1);
      uint8x16_t curr = vld1q_u8(curr_row + j - 1);
      uint8x16_t next = vld1q_u8(next_row + j - 1);
      
      // Extract 8-byte windows
      uint8x8_t prev_l = vget_low_u8(prev);
      uint8x8_t prev_m = vext_u8(vget_low_u8(prev), vget_high_u8(prev), 1);
      uint8x8_t prev_r = vext_u8(vget_low_u8(prev), vget_high_u8(prev), 2);
      
      uint8x8_t curr_l = vget_low_u8(curr);
      uint8x8_t curr_r = vext_u8(vget_low_u8(curr), vget_high_u8(curr), 2);
      
      uint8x8_t next_l = vget_low_u8(next);
      uint8x8_t next_m = vext_u8(vget_low_u8(next), vget_high_u8(next), 1);
      uint8x8_t next_r = vext_u8(vget_low_u8(next), vget_high_u8(next), 2);
      
      //  sign-16-bit arithmetic
      int16x8_t prev_l16 = vreinterpretq_s16_u16(vmovl_u8(prev_l));
      int16x8_t prev_m16 = vreinterpretq_s16_u16(vmovl_u8(prev_m));
      int16x8_t prev_r16 = vreinterpretq_s16_u16(vmovl_u8(prev_r));
      int16x8_t curr_l16 = vreinterpretq_s16_u16(vmovl_u8(curr_l));
      int16x8_t curr_r16 = vreinterpretq_s16_u16(vmovl_u8(curr_r));
      int16x8_t next_l16 = vreinterpretq_s16_u16(vmovl_u8(next_l));
      int16x8_t next_m16 = vreinterpretq_s16_u16(vmovl_u8(next_m));
      int16x8_t next_r16 = vreinterpretq_s16_u16(vmovl_u8(next_r));
      
      // G_x = (prev_{r}-prev_{l}) + 2(curr_{r}-curr_{l}) + (next_{r}-next_{l})
      int16x8_t gx = vsubq_s16(prev_r16, prev_l16);
      gx = vaddq_s16(gx, vshlq_n_s16(vsubq_s16(curr_r16, curr_l16), 1));
      gx = vaddq_s16(gx, vsubq_s16(next_r16, next_l16));
      gx = vabsq_s16(gx);
      
      // G_y = (next_l-prev_l) + 2(next_m-prev_m) + (next_r-prev_r)
      int16x8_t gy = vsubq_s16(next_l16, prev_l16);
      gy = vaddq_s16(gy, vshlq_n_s16(vsubq_s16(next_m16, prev_m16), 1));
      gy = vaddq_s16(gy, vsubq_s16(next_r16, prev_r16));
      gy = vabsq_s16(gy);
      
      // comb
      int16x8_t mag = vaddq_s16(gx, gy);
      uint8x8_t result = vqmovun_s16(mag);
      
      // Store
      vst1_u8(out_row + j, result);
    }
    
    // Scalar for remaining pixels
    for (; j < img_gray.cols - 1; j++) {
      int gx = abs((int)prev_row[j+1] - (int)prev_row[j-1] +
                   2*((int)curr_row[j+1] - (int)curr_row[j-1]) +
                   (int)next_row[j+1] - (int)next_row[j-1]);
      
      int gy = abs((int)next_row[j-1] - (int)prev_row[j-1] +
                   2*((int)next_row[j] - (int)prev_row[j]) +
                   (int)next_row[j+1] - (int)prev_row[j+1]);
      
      int mag = gx + gy;
      out_row[j] = (mag > 255) ? 255 : mag;
    }
  }
}