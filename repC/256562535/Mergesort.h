/*
The two functions merge and sort are from https://www.geeksforgeeks.org/iterative-merge-sort/.
*/
#pragma once
#include <cstdint>


#define min(x, y) (x < y) ? x : y;
#define MAX_PRINT_SIZE 20

class Mergesort
{
private:
	static void merge(int32_t *arr, int l, int m, int r);
public:
	// iterative version of merge sort
	static void sort(int32_t* arr, int size);
	// parallel iterative version of merge sort
	static void sortParallel(int32_t* arr, int size);

	// pseudo-random number generator
	static void fillWithRandomNumbers(int32_t* arr, int size);
	static void print(const int32_t* arr, int size);
};



