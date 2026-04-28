/*
 * image_data.h
 * Auto-generated for STM32 X-CUBE-AI binary classifier (digit 3)
 */
#ifndef INC_IMAGE_DATA_H_
#define INC_IMAGE_DATA_H_

#define NUM_TEST_IMAGES  20
#define IMAGE_SIZE       784   /* 28 x 28 pixels */

/* 1 = IS a 3,  0 = NOT a 3 */
extern const int   test_labels[NUM_TEST_IMAGES];
extern const float test_images[NUM_TEST_IMAGES][IMAGE_SIZE];

#endif /* INC_IMAGE_DATA_H_ */
