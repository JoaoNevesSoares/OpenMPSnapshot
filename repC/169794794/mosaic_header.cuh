//
// Created by Hao Xu on 2019-03-07.
//
#pragma once

#ifndef PPM_H
#define PPM_H

#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <time.h>
#include <omp.h>
#include <string.h>

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

/** *********** Type Define *********** **/
typedef enum PPM_TYPE {
	PPM_BINARY, PPM_PLAIN_TEXT
} PPM;

typedef enum MODE {
	CPU, OPENMP, GPU, ALL
} MODE;

typedef struct Img {
	unsigned int width;
	unsigned int height;
	unsigned int color_value;
	unsigned int num_pixel;

	unsigned char * data;
} Img;

typedef struct Mosaic {
	unsigned int cell_size, cell_num, cell_main_num;
	unsigned int cell_num_height, cell_remain_height;
	unsigned int cell_num_weight, cell_remain_weight;
	unsigned int pixcel_num;
} Mosaic;

typedef int(*Func)(int, int, Img *, Mosaic*);
typedef void(*Process)(Img *, Mosaic *, int *, Func, int, int *);

/** *********** Utility Functions *********** **/
void error(char *message);
int is_exp_of_two(unsigned int x);
void print_help();
int process_command_line(int argc, char *argv[]);


/** *********** Pre-Process Data *********** **/
void read_ppm(const char * img_path, Img * image);
void write_ppm_binary(Img * img_path, char * file_name);
void write_ppm_text(Img * image, char * file_name);

void compute_mosaic_info(unsigned int cell_size, Img * image, Mosaic * mosic);


/** *********** Implementation *********** **/
void run_gpu(Img * image, Mosaic * mos, unsigned long long int * ave);
void run_gpu2(Img * image, Mosaic * mos, unsigned long long int * ave);

void run_cpu(Img * image, Mosaic * mos, Process ppp);

// cpu version
void process_mosaic_section_cpu(Img * image, Mosaic * mos, int limits[3], Func fff, int pixcel_num, int * total);

// gpu version
void process_mosaic_section_openmp2(Img * image, Mosaic * mos, int limits[3], Func fff, int pixcel_num, int * total);	// for inner loop


#endif /* PPM_H */