#include "mosaic_header.cuh"

#define FAILURE 0
#define SUCCESS !FAILURE

#define USER_NAME "acq18hx"

#define BUFFER_SIZE 100

// =========================== global variables ===========================
MODE execution_mode = CPU;
PPM write_type = PPM_BINARY;
char * in_file;
char * out_file;
char out_name[100];
int cell_size = 8;



// =========================== main ===========================
int 
main(int argc, char *argv[]) {
	//========================================================
	if (process_command_line(argc, argv) == FAILURE)
		return 1;
	//========================================================	

    // variable define and initialise
	Img * image;
	Mosaic * mos;
	unsigned long long int * ave;					// the average r, g, b over the whole image

    
	// allocate CPU memory
	image = (Img *)malloc(sizeof(Img));
	mos = (Mosaic *)malloc(sizeof(Mosaic));
	ave = (unsigned long long int *)malloc(3 * sizeof(unsigned long long int));

	
	// load image and get mosaic infomation
	read_ppm(in_file, image);
	compute_mosaic_info(cell_size, image, mos);


	// ------------- execute the mosaic filter based on the mode
	switch (execution_mode)
	{
	case CPU: {
	
		break;
	}

	case OPENMP: {
		break;
	}
		
	case GPU: {
		

		break;
	}

	case ALL: {
		/****** CPU ******/
		// start timing
		clock_t timer = clock();

		// process image
		run_cpu(image, mos, &process_mosaic_section_cpu);

		// compute and print time
		double cost = (double)(clock() - timer) / CLOCKS_PER_SEC;
		printf("CPU mode execution time took %d s and %.4fms\n\n", (int)cost, ((cost - (int)cost) * 1000));


		/****** OPENMP ******/
		// load image and get mosaic infomation
		read_ppm(in_file, image);
		compute_mosaic_info(cell_size, image, mos);

		// start timing
		double time_begin = omp_get_wtime();
		
		// process image
		run_cpu(image, mos, &process_mosaic_section_openmp2);

		// compute and print time
		double time_end = omp_get_wtime();
		cost = (double)(time_end - time_begin);
		printf("OPENMP mode execution time took %d s and %.4fms\n\n", (int)cost, (cost - (int)cost) * 1000);

		/****** GPU ******/
		// timing in the gpu image processing function

		// load image and get mosaic infomation
		read_ppm(in_file, image);
		compute_mosaic_info(cell_size, image, mos);

		// process image
		run_gpu(image, mos, ave);

		break;
	}

	default:
		break;
	}

	// ------------ write output image file (either binary or plain text ppm)
	sprintf(out_name, "%s_acq18hx.ppm", out_file);
	switch (write_type)
	{
	case PPM_PLAIN_TEXT: {
		write_ppm_text(image, out_name);
		break;
	}
	default: {
		write_ppm_binary(image, out_name);
		break;
	}
	}


	
	// free CPU memory
	free(ave);
	free(image->data);
	free(image);
	free(mos);

    return 0;
}

/************************ Various Utility Functions ****************************/
void 
print_help() {
	printf("mosaic_%s C M -i input_file -o output_file [options]\n", USER_NAME);

	printf("where:\n");
	printf("\tC              Is the mosaic cell size which should be any positive\n"
		"\t               power of 2 number \n");
	printf("\tM              Is the mode with a value of either CPU, OPENMP or\n"
		"\t               ALL. The mode specifies which version of the simulation\n"
		"\t               code should execute. ALL should execute each mode in\n"
		"\t               turn.\n");
	printf("\t-i input_file  Specifies an input image file\n");
	printf("\t-o output_file Specifies an output image file which will be used\n"
		"\t               to write the mosaic image\n");
	printf("[options]:\n");
	printf("\t-f ppm_format  PPM image output format either PPM_BINARY (default) or \n"
		"\t               PPM_PLAIN_TEXT\n ");
}

/* read and check the input parameters */
int 
process_command_line(int argc, char *argv[]) {
	if (argc < 7) {
		fprintf(stderr, "Error: Missing program arguments. Correct usage is...\n");
		print_help();
		return FAILURE;
	}
	//first argument is always the executable name

	//read in the non optional command line arguments
	cell_size = (unsigned int)atoi(argv[1]);
	if (!is_exp_of_two(cell_size)) {
		fprintf(stderr, "mosaic size should be exp of 2");
		print_help();
	}


	if (!strcmp(argv[2], "CPU")) { execution_mode = CPU; };
	if (!strcmp(argv[2], "OPENMP")) { execution_mode = OPENMP; };
	if (!strcmp(argv[2], "GPU")) { execution_mode = GPU; };
	if (!strcmp(argv[2], "ALL")) { execution_mode = ALL; };
	//TODO: read in the input image name
	in_file = argv[4];
	//TODO: read in the output image name
	out_file = argv[6];
	if (argc > 7) {
		//TODO: read in any optional part 3 arguments
		if (!strcmp(argv[8], "PPM_BINARY")) { write_type = PPM_BINARY; };
		if (!strcmp(argv[8], "PPM_PLAIN_TEXT")) { write_type = PPM_PLAIN_TEXT; };
	}

	return SUCCESS;
}

/* print error message and exit */
void
error(char *message) {
	fprintf(stderr, "ppm: %s\n", message);
	exit(1);
}

/** check if 2 ** n */
int
is_exp_of_two(unsigned int x) {
	unsigned int t = 1;
	while (1)
	{
		if (t == x) return 1;

		if (t > x) return 0;

		t = t * 2;
	}
}

/************************ PPM I/O Functions ****************************/
static void
skip_comment(FILE *fp) {
	int c = getc(fp);
	while (c == '#') {
		while (getc(fp) != '\n');
		c = getc(fp);
	}
	ungetc(c, fp);
}

/* read either p3 or p6, the output pixel data is a list of unsigned char */
void
read_ppm(const char *img_path, Img *image) {
	// open file
	FILE *fp = fopen(img_path, "rb");
	if (!fp) fprintf(stderr, "Image file cannot be accessed");

	// local variables
	unsigned int length;
	char buffer[BUFFER_SIZE];

	// get ppm type
	if (!fgets(buffer, BUFFER_SIZE, fp)) fprintf(stderr, "Cant get ppm type");
	PPM ppm_type = (buffer[1] == '6') ? PPM_BINARY : PPM_PLAIN_TEXT;

	// get comment and other three features
	skip_comment(fp);
	if (!fscanf(fp, "%d\n", &image->width)) fprintf(stderr, "fail to scan width");
	skip_comment(fp);
	if (!fscanf(fp, "%d\n", &image->height)) fprintf(stderr, "fail to scan height");
	skip_comment(fp);
	if (!fscanf(fp, "%d\n", &image->color_value)) fprintf(stderr, "fail to scan color range");


	image->num_pixel = image->width * image->height;
	length = 3 * image->num_pixel;

	// get pixel data
	switch (ppm_type) {
	case (PPM_BINARY): {
		image->data = (unsigned char *)malloc(length);
		if (!image->data) fprintf(stderr, "data cannot be allocated on memory");

		if (fread((void *)image->data, 1, (size_t)length, fp) != length)
			fprintf(stderr, "cannot read image data from file");
		//            printf("\n");
		//
		//            for(int i = 0; i<100; i++)
		//                printf("%hhu ", image->data[i]);
		break;
	}
	case (PPM_PLAIN_TEXT): {
		int temp_int;
		unsigned char * temp_data = image->data = (unsigned char *)malloc(length);
		int data_index = 0;
		while (fscanf(fp, "%d", &temp_int) == 1) {
			temp_data[data_index++] = (unsigned char)temp_int;
			//printf("%hhu %d \n", temp_data[data_index - 1], i++);
		}
		break;
	}
	}

	fclose(fp);
}

void
write_ppm_binary(Img * image, char * file_name) {
	FILE *fp = fopen(file_name, "wb");
	if (!fp) fprintf(stderr, "file cannot be created or opened");

	// write header
	fprintf(fp, "P6\n%d\n%d\n%d\n", image->width, image->height, image->color_value);

	// write pixel data
	unsigned int length = 3 * image->width * image->height;
	if (fwrite((void *)image->data, 1, (size_t)length, fp) != length)
		fprintf(stderr, "data cannot be written to file");

	// close file
	fclose(fp);
}

void
write_ppm_text(Img * image, char * file_name) {
	FILE *fp = fopen(file_name, "w");

	//  write header
	fprintf(fp, "P3\n%d %d\n%d\n", image->width, image->height, image->color_value);

	// format all data into string and write it into file 
	unsigned int index = 0;
	for (unsigned int i = 0; i < image->height; i++) {
		index = i * image->width * 3;
		for (unsigned int j = 0; j < (image->width - 1) * 3; j += 3) {
			fprintf(fp, "%d %d %d\t", image->data[index + j], image->data[index + j + 1], image->data[index + j + 2]);
		}

		index += (image->width - 1) * 3;
		if (i == image->height - 1) {
			fprintf(fp, "%d %d %d", image->data[index], image->data[index + 1], image->data[index + 2]);
		}
		else {
			fprintf(fp, "%d %d %d\n", image->data[index], image->data[index + 1], image->data[index + 2]);
		}

	}
	/**/

	fclose(fp);
}

/************************ Mosaic Function ****************************/
void
compute_mosaic_info(unsigned int cell_size, Img * image, Mosaic * mos) {
	mos->cell_size = cell_size;

	// check the cell size
	if (cell_size > image->height || cell_size > image->width)
	{
		printf("Warning: invalid mosaic size input!");
		exit(1);
	}

	// the numbers of mosaic cell in each row and column, as well as their remain pixels
	mos->cell_num_height = image->height / cell_size;
	mos->cell_remain_height = image->height % cell_size;
	mos->cell_num_weight = image->width / cell_size;
	mos->cell_remain_weight = image->width % cell_size;

	// calculate the total number of mosaic cell
	mos->cell_main_num = mos->cell_num_height * mos->cell_num_weight;
	mos->cell_num = mos->cell_main_num;
	if (mos->cell_remain_weight != 0) mos->cell_num += mos->cell_num_height;
	if (mos->cell_remain_height != 0) mos->cell_num += mos->cell_num_weight + 1;

	// calculate the number of pixel in each mosaic cell
	mos->pixcel_num = cell_size * cell_size;

}

/************************ GPU Mosaic Function ****************************/
__device__ unsigned long long int dev_ave[3];

 //========== kernel functions ============
__global__ void
averageFilterMain(unsigned int width, Mosaic * dev_mos, unsigned char * dev_data) {
	__shared__ unsigned char r, g, b;					// average over mosaic cell
	__shared__ unsigned int cell_r, cell_g, cell_b;		// sum over mosaic cell
	cell_r = 0; cell_g = 0; cell_b = 0;

	register int row_r = 0, row_g = 0, row_b = 0;		// sum over row in the cell


	register int index = ((blockIdx.x / dev_mos->cell_num_weight * dev_mos->cell_size + threadIdx.x) * width + (blockIdx.x % dev_mos->cell_num_weight) * dev_mos->cell_size) * 3;

	for (int i = 0; i < dev_mos->cell_size; i++) {
		row_r += (int)dev_data[index + i * 3];
		row_g += (int)dev_data[index + i * 3 + 1];
		row_b += (int)dev_data[index + i * 3 + 2];
	}

	// sum over the mosaic cell
	atomicAdd(&cell_r, row_r);
	atomicAdd(&cell_g, row_g);
	atomicAdd(&cell_b, row_b);

	//printf("from block %d, sum over image is %u, %u, %u \n", blockIdx.x, cell_r, cell_g, cell_b);
	//synchronize the local threads writing for average calculation
	__syncthreads();



	// calculate the average over the mosaic cell
	if (threadIdx.x == 0) {
		//printf("from block %d, thread %d, sum over row is %u, %u, %u \n", 
		//blockIdx.x, threadIdx.x, cell_r, cell_g, cell_b);

		unsigned int temp = dev_mos->cell_size * dev_mos->cell_size;
		r = (unsigned char)(cell_r / temp);
		g = (unsigned char)(cell_g / temp);
		b = (unsigned char)(cell_b / temp);

		// sum over the blocks
		atomicAdd(&dev_ave[0], (int)r);
		atomicAdd(&dev_ave[1], (int)g);
		atomicAdd(&dev_ave[2], (int)b);
	}

	//synchronize the local threads writing for average calculation
	__syncthreads();

	// assign averaged value back to image data
	for (int i = 0; i < dev_mos->cell_size; i++) {
		dev_data[index + i * 3] = r;
		dev_data[index + i * 3 + 1] = g;
		dev_data[index + i * 3 + 2] = b;
	}

	/*if (threadIdx.x == 1)
	{
	printf("from block %d, thread 1 is %u, %u, %u \n", blockIdx.x, dev_data[index], dev_data[index +1], dev_data[index+2]);
	}*/
}

__global__ void
averageFilterMain2(unsigned int width, Mosaic * dev_mos, uchar3 * dev_data) {
	__shared__ unsigned char r, g, b;					// average over mosaic cell
	__shared__ unsigned int cell_r, cell_g, cell_b;		// sum over mosaic cell
	cell_r = 0; cell_g = 0; cell_b = 0;

	register int row_r = 0, row_g = 0, row_b = 0;		// sum over row in the cell
	register uchar3 pixcel;

	register int index = (blockIdx.x / dev_mos->cell_num_weight * dev_mos->cell_size + threadIdx.x) * width + (blockIdx.x % dev_mos->cell_num_weight) * dev_mos->cell_size;

	for (int i = 0; i < dev_mos->cell_size; i++) {
		pixcel = dev_data[index + i];
		row_r += (int)pixcel.x;
		row_g += (int)pixcel.y;
		row_b += (int)pixcel.z;
	}

	// sum over the mosaic cell
	atomicAdd(&cell_r, row_r);
	atomicAdd(&cell_g, row_g);
	atomicAdd(&cell_b, row_b);

	//printf("from block %d, sum over image is %u, %u, %u \n", blockIdx.x, cell_r, cell_g, cell_b);
	//synchronize the local threads writing for average calculation
	__syncthreads();



	// calculate the average over the mosaic cell
	if (threadIdx.x == 0) {
		//printf("from block %d, thread %d, sum over row is %u, %u, %u \n", 
		//blockIdx.x, threadIdx.x, cell_r, cell_g, cell_b);

		unsigned int temp = dev_mos->cell_size * dev_mos->cell_size;
		r = (unsigned char)(cell_r / temp);
		g = (unsigned char)(cell_g / temp);
		b = (unsigned char)(cell_b / temp);

		// sum over the blocks
		atomicAdd(&dev_ave[0], (int)r);
		atomicAdd(&dev_ave[1], (int)g);
		atomicAdd(&dev_ave[2], (int)b);
	}

	//synchronize the local threads writing for average calculation
	__syncthreads();

	// assign averaged value back to image data
	for (int i = 0; i < dev_mos->cell_size; i++) {
		pixcel.x = r;
		pixcel.y = g;
		pixcel.z = b;
		dev_data[index + i] = pixcel;
	}

	/*if (threadIdx.x == 1)
	{
	printf("from block %d, thread 1 is %u, %u, %u \n", blockIdx.x, dev_data[index], dev_data[index +1], dev_data[index+2]);
	}*/
}

// ========== run functions ============
void
run_gpu(Img * image, Mosaic * mos, unsigned long long int * ave) {
	// process: average filter on GPU
	// change:  outer variable: image
	// return:  cudaStatus

	// variable declear and initialise
	cudaError_t cudaStatus = cudaSuccess;
	cudaEvent_t start, stop;
	float ms;						// timing

	Img * dev_image;
	Mosaic * dev_mos;
	unsigned char * dev_data;		// array of r, g, b in device


	// create timer
	cudaEventCreate(&start);
	cudaEventCreate(&stop);


	// starting timing here
	cudaEventRecord(start, 0);


	// allocate GPU memory
	cudaStatus = cudaMalloc((void **)&dev_image, sizeof(Img));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc image failed!");
		exit(1);
	}

	cudaStatus = cudaMalloc((void **)&dev_mos, sizeof(Mosaic));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc mosaic info failed!");
		cudaFree(dev_image);
		exit(1);
	}

	cudaStatus = cudaMalloc((void **)&dev_data, (3 * image->num_pixel));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc data failed!");
		cudaFree(dev_image);
		cudaFree(dev_mos);
		exit(1);
	}


	// copy data from host to device
	cudaStatus = cudaMemcpy(dev_image, image, sizeof(Img), cudaMemcpyHostToDevice);

	cudaStatus = cudaMemcpy(dev_mos, mos, sizeof(Img), cudaMemcpyHostToDevice);

	cudaStatus = cudaMemcpy(dev_data, image->data, 3 * image->num_pixel * sizeof(unsigned char), cudaMemcpyHostToDevice);

	// ======================= process mosaic =======================
	// -> main section <-
	unsigned int blocksPerGrid = mos->cell_num_weight * mos->cell_num_height;
	unsigned int threadsPerBlock = mos->cell_size;
	averageFilterMain << <blocksPerGrid, threadsPerBlock >> > (image->width, dev_mos, dev_data);

	cudaDeviceSynchronize();
	cudaStatus = cudaGetLastError();

	// copy image data from device to host
	cudaStatus = cudaMemcpy(image->data, dev_data, 3 * image->num_pixel * sizeof(unsigned char), cudaMemcpyDeviceToHost);
	cudaStatus = cudaMemcpyFromSymbol(ave, dev_ave, 3 * sizeof(unsigned long long int));


	// print the average r, g and b over image
	printf("Average image colour red = %llu, green = %llu, blue = %llu \n",
		(ave[0] / mos->cell_num), (ave[1] / mos->cell_num), (ave[2] / mos->cell_num));

	
	// end timing and print
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&ms, start, stop);
	printf("CUDA mode execution time took %d s and %f ms\n", (int)ms / 1000, ms);


	// free cuda memory
	cudaFree(dev_image);
	cudaFree(dev_mos);
	cudaFree(dev_data);


	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
	}

}


void
run_gpu2(Img * image, Mosaic * mos, unsigned long long int * ave) {
	// process: average filter on GPU
	// change:  outer variable: image
	// return:  cudaStatus

	// variable declear and initialise
	cudaError_t cudaStatus = cudaSuccess;
	cudaEvent_t start, stop;
	float ms;						// timing

	Img * dev_image;
	Mosaic * dev_mos;
	uchar3 * data3;
	uchar3 * dev_data;		// array of pixcel with r, g, b dimensions in device

	// change the data from 1d to 3d
	data3 = (uchar3*)malloc(image->num_pixel * sizeof(uchar3));
	for (unsigned int i = 0; i < image->num_pixel; i++) {
		data3[i].x = image->data[i * 3];
		data3[i].y = image->data[i * 3 + 1];
		data3[i].z = image->data[i * 3 + 2];
	}


	// create timer
	cudaEventCreate(&start);
	cudaEventCreate(&stop);


	// starting timing here
	cudaEventRecord(start, 0);


	// allocate GPU memory
	cudaStatus = cudaMalloc((void **)&dev_image, sizeof(Img));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc image failed!");
		exit(1);
	}

	cudaStatus = cudaMalloc((void **)&dev_mos, sizeof(Mosaic));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc mosaic info failed!");
		cudaFree(dev_image);
		exit(1);
	}

	cudaStatus = cudaMalloc((void **)&dev_data, image->num_pixel * sizeof(uchar3));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc data failed!");
		cudaFree(dev_image);
		cudaFree(dev_mos);
		exit(1);
	}


	// copy data from host to device
	cudaStatus = cudaMemcpy(dev_image, image, sizeof(Img), cudaMemcpyHostToDevice);

	cudaStatus = cudaMemcpy(dev_mos, mos, sizeof(Img), cudaMemcpyHostToDevice);

	cudaStatus = cudaMemcpy(dev_data, data3, image->num_pixel * sizeof(uchar3), cudaMemcpyHostToDevice);
	// ======================= process mosaic =======================
	// -> main section <-
	unsigned int blocksPerGrid = mos->cell_num_weight * mos->cell_num_height;
	unsigned int threadsPerBlock = mos->cell_size;
	averageFilterMain2 << <blocksPerGrid, threadsPerBlock >> > (image->width, dev_mos, dev_data);

	cudaDeviceSynchronize();
	cudaStatus = cudaGetLastError();

	// copy image data from device to host
	cudaStatus = cudaMemcpy(dev_data, data3, image->num_pixel * sizeof(uchar3), cudaMemcpyDeviceToHost);
	cudaStatus = cudaMemcpyFromSymbol(ave, dev_ave, 3 * sizeof(unsigned long long int));


	// print the average r, g and b over image
	printf("Average image colour red = %llu, green = %llu, blue = %llu \n",
		(ave[0] / mos->cell_num), (ave[1] / mos->cell_num), (ave[2] / mos->cell_num));


	// end timing and print
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&ms, start, stop);
	printf("CUDA mode execution time took %d s and %f ms\n", (int)ms / 1000, ms);


	// chage the data type back to 1d
	for (unsigned int i = 0; i < image->num_pixel; i++) {
		image->data[i * 3] = data3[i].x;
		image->data[i * 3 + 1] = data3[i].y;
		image->data[i * 3 + 2] = data3[i].z;
	}


	// free cpu memory
	free(data3);

	// free cuda memory
	cudaFree(dev_image);
	cudaFree(dev_mos);
	cudaFree(dev_data);


	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
	}

}

/************************ Index Function ****************************/
int
index_main(int cell, int row, Img * image, Mosaic* mos) {
	return ((cell / mos->cell_num_weight * mos->cell_size + row) * image->width + (cell % mos->cell_num_weight) * mos->cell_size) * 3;
}

int
index_column_edge(int cell, int row, Img * image, Mosaic * mos) {
	return (cell * image->width * mos->cell_size + row * image->width + mos->cell_num_weight * mos->cell_size) * 3;
}

int
index_row_edge(int cell, int row, Img * image, Mosaic * mos) {
	return (image->width * (mos->cell_num_height * mos->cell_size + row) + cell * mos->cell_size) * 3;
}

int
index_end_case(int cell, int row, Img * image, Mosaic * mos) {
	return ((mos->cell_num_height * mos->cell_size + row) * image->width + mos->cell_num_weight * mos->cell_size) * 3;
}


/************************ CPU Mosaic Function ****************************/

void
process_mosaic_section_cpu(Img * image, Mosaic * mos, int limits[3], Func fff, int pixcel_num, int * total) {
	/** limits[0]: the number of mosaic in this mosaic section
	limits[1]: the number of pixcel rows in the current mosaic
	limits[2]: the number of pixcel columns in the current mosaic*/

	for (int i = 0; i < limits[0]; i++)	// the ith cell
	{
		/****** variables declaraction ******/
		unsigned char r, g, b;
		int r_sum = 0, g_sum = 0, b_sum = 0;
		int index;

		for (int j = 0; j < limits[1]; j++)	// the jth row of ith cell
		{
			// get the index in photo data that strar the jth row of the ith cell
			index = fff(i, j, image, mos);

			for (int k = 0; k < limits[2]; k++)	// the kth element of jth row
			{
				r_sum += (int)image->data[index + k * 3];
				g_sum += (int)image->data[index + k * 3 + 1];
				b_sum += (int)image->data[index + k * 3 + 2];
			}
		}

		// calculate the average
		r = (unsigned char)(r_sum / pixcel_num);
		g = (unsigned char)(g_sum / pixcel_num);
		b = (unsigned char)(b_sum / pixcel_num);

		// add to total
		total[0] += r; total[1] += g; total[2] += b;

		// mosaic the original
		for (int j = 0; j < limits[1]; ++j) {      // the jth row of ith cell
												   // index in data that start the l row in ith cell
			index = fff(i, j, image, mos);

			for (int k = 0; k < limits[2]; ++k) {       // the kth element of jth row
				image->data[index + k * 3] = r;
				image->data[index + k * 3 + 1] = g;
				image->data[index + k * 3 + 2] = b;
			}
		}
	}
}

void
run_cpu(Img * image, Mosaic * mos, Process ppp) {
	/****** variables declaraction ******/
	int * limits = (int *)malloc(3);
	int pixcel_num;
	int * total = (int *)malloc(3);
	total[0] = 0; total[1] = 0; total[2] = 0;


	/****** process main mosaic section ******/
	limits[0] = mos->cell_main_num;
	limits[1] = mos->cell_size;
	limits[2] = mos->cell_size;

	pixcel_num = mos->cell_size * mos->cell_size;
	ppp(image, mos, limits, &index_main, pixcel_num, total);

	/****** process column-edge mosaic section ******/
	if (mos->cell_remain_weight != 0)
	{
		limits[0] = mos->cell_num_height;
		limits[1] = mos->cell_size;
		limits[2] = mos->cell_remain_weight;

		pixcel_num = mos->cell_size * mos->cell_remain_weight;

		ppp(image, mos, limits, &index_column_edge, pixcel_num, total);
	}

	/****** process row-edge mosaic section ******/
	if (mos->cell_remain_weight != 0)
	{
		limits[0] = mos->cell_num_weight;
		limits[1] = mos->cell_remain_height;
		limits[2] = mos->cell_size;

		pixcel_num = mos->cell_size * mos->cell_remain_height;
		ppp(image, mos, limits, &index_row_edge, pixcel_num, total);

		/****** process end-corner mosaic section ******/
		limits[0] = 1;
		limits[1] = mos->cell_remain_height;
		limits[2] = mos->cell_remain_weight;

		pixcel_num = mos->cell_remain_height * mos->cell_remain_weight;

		ppp(image, mos, limits, &index_end_case, pixcel_num, total);
		//process_mosaic_section_cpu(image, mos, limits, &index_end_case, pixcel_num, total);		// for CPU version 2.0

	}

	// Output the average colour value for the image
	printf("Average image colour red = %d, green = %d, blue = %d \n",
		total[0] / mos->cell_num, total[1] / mos->cell_num, total[2] / mos->cell_num);
}

/************************ OpenMP Mosaic Function ****************************/
void
process_mosaic_section_openmp2(Img * image, Mosaic * mos, int limits[3], Func fff, int pixcel_num, int * total) {

	/** limits[0]: the number of mosaic in this mosaic section
	limits[1]: the number of pixcel rows in the current mosaic
	limits[2]: the number of pixcel columns in the current mosaic*/
	int i = 0;
	omp_set_num_threads(16);

	// construct array of int for r, g, b, each thread can only access an element dependent on its thread number.
	int rrr[16] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
	int ggg[16] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
	int bbb[16] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };

	//#pragma omp parallel for private(i) schedule(runtime)
	//#pragma omp parallel for private(i) schedule(static, 1)
#pragma omp parallel for private(i) schedule(static)
	//#pragma omp parallel for private(i) schedule(dynamic)
	for (i = 0; i < limits[0]; i++)	// the ith cell
	{
		/****** variables declaraction ******/
		unsigned char r, g, b;
		int r_sum = 0, g_sum = 0, b_sum = 0;
		int index;


		for (int j = 0; j < limits[1]; j++)	// the jth row of ith cell
		{
			// get the index in photo data that strar the jth row of the ith cell
			index = fff(i, j, image, mos);

			for (int k = 0; k < limits[2]; k++)	// the kth element of jth row
			{
				r_sum += (int)image->data[index + k * 3];
				g_sum += (int)image->data[index + k * 3 + 1];
				b_sum += (int)image->data[index + k * 3 + 2];
			}
		}

		// calculate the average
		r = (unsigned char)(r_sum / pixcel_num);
		g = (unsigned char)(g_sum / pixcel_num);
		b = (unsigned char)(b_sum / pixcel_num);

		// add to total
		rrr[omp_get_thread_num()] += r;
		ggg[omp_get_thread_num()] += g;
		bbb[omp_get_thread_num()] += b;

		// mosaic the original
		for (int j = 0; j < limits[1]; ++j) {      // the jth row of ith cell
												   // index in data that start the l row in ith cell
			index = index = fff(i, j, image, mos);

			for (int k = 0; k < limits[2]; ++k) {       // the kth element of jth row
				image->data[index + k * 3] = r;
				image->data[index + k * 3 + 1] = g;
				image->data[index + k * 3 + 2] = b;
			}
		}

	}

	for (int m = 0; m < 16; m++)
	{
		total[0] += rrr[m];
		total[1] += ggg[m];
		total[2] += bbb[m];
	}
}