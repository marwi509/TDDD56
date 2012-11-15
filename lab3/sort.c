/*
 * sort.c
 *
 *  Created on: 5 Sep 2011
 *  Copyright 2011 Nicolas Melot
 *
 * This file is part of TDDD56.
 * 
 *     TDDD56 is free software: you can redistribute it and/or modify
 *     it under the terms of the GNU General Public License as published by
 *     the Free Software Foundation, either version 3 of the License, or
 *     (at your option) any later version.
 * 
 *     TDDD56 is distributed in the hope that it will be useful,
 *     but WITHOUT ANY WARRANTY; without even the implied warranty of
 *     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *     GNU General Public License for more details.
 * 
 *     You should have received a copy of the GNU General Public License
 *     along with TDDD56. If not, see <http://www.gnu.org/licenses/>.
 * 
 */

// Do not touch or move these lines
#include <stdio.h>
#include "disable.h"

#ifndef DEBUG
#define NDEBUG
#endif

#include "array.h"
#include "sort.h"
#include "simple_quicksort.h"
#include <string.h>
#include <limits.h>
#include "thread_args.h"
#include "thread_handler.h"
#include "insertion_sort.h"
#include "merge_sort.h"

//#include "custom_sort.h"



void quick_sort(void* args);
void quick_sort_impl(struct array* theArray, int start, int end);
void merge_sort(struct array* theArray, int start, int end);
void stupid_sort(struct array* theArray, int nrOfThreads);
void sample_sort(struct array* theArray, int start, int end, int nrOfThreads);

int
sort(struct array * array)
{
	
	//quick_sort(array, 0, array -> length - 1);
	/*
	struct array* newArray = array_alloc(array -> length);
	newArray -> length = array -> length;
	int i;
	for(i = 0; i < array -> length; i ++)
	{
		newArray -> data[i] = array -> data[i];
	}
	*/
	//merge_sort(array, 0, array -> length - 1, 0);
	//stupid_sort(array, 2);
	//memcpy(array -> data, newArray -> data, sizeof(value) * array -> length);
	struct thread_args* theStartingArguments = malloc(sizeof(struct thread_args));
	theStartingArguments -> theArray = array;
	theStartingArguments -> start = 0;
	theStartingArguments -> end = array -> length - 1;
	
	thread_handler_init();
	int threadCreated = get_available_thread();
	
	if(threadCreated == -1)
		quick_sort_impl(array, 0, array -> length - 1);
	else
		pthread_create(&threads[threadCreated], NULL, (void*)quick_sort, (void*) theStartingArguments);
	wait_for_thread(threadCreated);
	//merge_sort_void(theStartingArguments);
	//simple_quicksort_ascending(array);
	return 0;
}



#include <stdlib.h>
#include <string.h>
#include <pthread.h>

#ifndef NB_THREADS
	#define NB_THREADS 6
#endif
//pthread_t threads[6];
//int busy_threads[NB_THREADS];
pthread_mutex_t theThreadMutex;


int check_sorted(struct array* theArray, int start, int end)
{
	int i;
	for(i = start; i < end; i ++)
	{
		if(theArray -> data[i] > theArray -> data[i + 1])
			return 0;
	}
	return 1;
}

int check_desc_sorted(struct array* theArray, int start, int end)
{
	int i;
	for(i = start; i < end; i ++)
	{
		if(theArray -> data[i] < theArray -> data[i + 1])
			return 0;
	}
	return 1;
}
void swap(value* a, value* b);
void reverse_array(struct array* theArray, int start, int end)
{
	int i;
	for(i = start; i <= end/2; i ++)
	{
		swap(&theArray -> data[i], &theArray -> data[end - i]);
	}
}





void swap(value* a, value* b)
{
	value temp = *a;
	memcpy(a, b, sizeof(value));
	memcpy(b, &temp, sizeof(value));
}

void median_of_three(value *left, value *middle, value *right)
{
	if(*left >= *right && *left <= *middle)
	{
		swap(left, middle);
		swap(right, left);
	}
	else if(*left <= *right && *left >= *middle)
	{
		swap(left, middle);
	}
	else if(*right >= *left && *right <= *middle)
	{
		 swap(right, middle);
	}
	else if	(*right <= *left && *right >= *middle)
	{
		swap(right, middle);
		swap(left, right);
	}
	else if(*middle >= *left && *middle <= *right);
	else
	{
		swap(right, left);
	}
}

int partition(struct array* theArray, int start, int end, int piv)
{
	value x = theArray -> data[piv];
	theArray -> data[piv] = theArray -> data[end];
	theArray -> data[end] = x;
	int i=start,j=end-1;
	value temp;
	while(i < j)
	{
		if(theArray -> data[i] >= x)
		{
			if(theArray -> data[j] <= x)
			{
				temp = theArray -> data[i];
				theArray -> data[i++] = theArray -> data[j];
				theArray -> data[j--] = temp;
			}
			else
			{
				j--;
			}
		}
		else if(theArray -> data[j] <= x)
		{
			i++;
		}
		else
		{
			i++;
			j--;
		}
	}
	for(;theArray -> data[i] < x; i++);
	theArray -> data[end] = theArray -> data[i];
	theArray -> data[i] = x;
	return i;
}
/*
int partition(struct array* theArray, int start, int end, int pivot_index)
{
	value pivot_value = theArray -> data[pivot_index];
	swap(&theArray -> data[pivot_index], &theArray -> data[end]);
	int left_index = start, right_index = end -1;
	int i;
	
	for(i = start; i < right_index; i ++)
	{
		if(theArray -> data[i] <= pivot_value)
		{
			swap(&theArray -> data[i], &theArray -> data[left_index++]);
			right_index--;
		}
	}
	
	int last_right = end;
	value* a = theArray -> data;
	while(right_index > left_index)
	{
		if(a[left_index] >= pivot_value && a[right_index] <= pivot_value)
		{
			swap(&a[left_index++], &a[right_index--]);
			last_right = right_index + 1;
		}
		else if(a[left_index] < pivot_value)
			left_index++;
		else if(a[right_index] > pivot_value)
		{
			last_right = right_index;
			right_index--;
		}
		
	}
	swap(&theArray -> data[end], &theArray -> data[last_right]);
	return last_right;
}
*/


void quick_sort(void* args)
{
	/*
	if(check_sorted(theArray, start, end))
	{
		return;
	}
	if(check_desc_sorted(theArray, start, end))
	{
		reverse_array(theArray, start, end);
		return;
	}
	*/
	struct thread_args * theArray = args;
	quick_sort_impl(theArray -> theArray, theArray -> start, theArray -> end);
}

void quick_sort_impl(struct array* theArray, int start, int end)
{
	if(end - start < 20)
	{
		insertion_sort(theArray, start, end);
	}
	else
	{
		int length = end - start;
		value* left_val = &theArray -> data[start], *right_val = &theArray -> data[end], *middle_val = &theArray -> data[start + length/2];
		median_of_three(left_val, middle_val, right_val);
		int pivot_final = partition(theArray, start + 1 , end - 1, start + length / 2);
		
		if(1 || (end == theArray -> length - 1 && start == 0))
		{
			struct thread_args* theArguments = malloc(sizeof(struct thread_args));
			theArguments -> theArray = theArray;
			theArguments -> start = start;
			theArguments -> end = pivot_final - 1;
			int threadCreated = get_available_thread();
			if(threadCreated == -1)
				quick_sort_impl(theArray, start, pivot_final - 1);
			else
				pthread_create(&threads[threadCreated], NULL, (void*) (quick_sort), (void*)theArguments);
			//quick_sort_impl(theArray, start, pivot_final - 1);
			
			struct thread_args* theArguments2 = malloc(sizeof(struct thread_args));
			theArguments2 -> theArray = theArray;
			theArguments2 -> start = pivot_final + 1;
			theArguments2 -> end = end;
			
			//pthread_create(&threads[1], NULL, (void*) (quick_sort), (void*)theArguments2);
			//quick_sort_impl(theArray, theArguments2 -> start, theArguments2 -> end );
			quick_sort(theArguments2);
			//pthread_join(threads[1], NULL);
			//pthread_join(threads[0], NULL);
			if(end == theArray -> length - 1 && start == 0)
			{
				printf("The length of the first half: %d\n", theArguments -> end - theArguments -> start);
			}
			wait_for_thread(threadCreated);
			free(theArguments);
			free(theArguments2);
			return;
		}
		quick_sort_impl(theArray, start, pivot_final - 1);
		quick_sort_impl(theArray, pivot_final + 1, end );
	}
}






void stupid_sort(struct array* theArray, int nrOfThreads)
{
	/*
	if(nrOfThreads < 2)
	{
		struct thread_args* theArgs = malloc(sizeof(struct thread_args));
		theArgs -> theArray = theArray;
		theArgs -> start = 0;
		theArgs -> end = theArray -> length - 1;
		quick_sort(theArgs);
		return;
	}
	*/
	int i;
	int elementsPerThread = theArray -> length / nrOfThreads;
	int totalElements = 0;
	for(i = 0; i < nrOfThreads; i ++)
	{
		struct thread_args* theArgs = malloc(sizeof(struct thread_args));
		theArgs -> theArray = theArray;
		theArgs -> start = i * elementsPerThread;
		if(i != nrOfThreads - 1)
		{
			theArgs -> end = (i + 1) * elementsPerThread - 1;
		}
		else
		{
			theArgs -> end = theArray -> length - 1;
		}
		pthread_create(&threads[i], NULL, (void*)(quick_sort), (void*)theArgs);
		//printf("Total number of elements for stupid_sort(): %d\n", theArgs -> end - theArgs -> start + 1);
	}
	for(i = 0; i < nrOfThreads; i ++)
	{
		pthread_join(threads[i], NULL);
	}
	//quick_sort_impl(theArray, 0, theArray -> length - 1);
	
	for(i = 1; i < nrOfThreads; i ++)
	{
		if(i != nrOfThreads - 1)
			merge_lists(theArray, 0, i * elementsPerThread - 1, i * elementsPerThread, (i + 1) * elementsPerThread - 1);
		else
			merge_lists(theArray, 0, i * elementsPerThread - 1, i * elementsPerThread, theArray -> length - 1);
	}
	
	//merge_lists(theArray, 0, theArray -> length / 2, theArray -> length / 2 + 1, theArray -> length - 1);
	
}


void sample_sort(struct array* theArray, int start, int end, int nrOfThreads)
{
	if(nrOfThreads < 2)
	{
		quick_sort_impl(theArray, start, end);
		return;
	}
	int elementsPerThread = (end - start + 1) / theArray -> length;
	int *thePivots = malloc(sizeof(int) * nrOfThreads);
	int *thePivotIndices = malloc(sizeof(int) * nrOfThreads - 1);
	int *tempList = malloc(sizeof(int) * theArray -> length);
	printf("Innan pivot choice\n");
	int i;
	for(i = 0; i < nrOfThreads; i ++)
	{
		int currentStart = elementsPerThread * i;
		int currentStop = elementsPerThread * (i + 1);
		thePivots[i] = theArray -> data[start + (currentStop - currentStart) / 2];
	}
	
	printf("Innan partition\n");
	int p;
	int current_index = start;
	struct array *thePivotArray = malloc(sizeof(struct array));
	thePivotArray -> data = thePivots;
	thePivotArray -> length = nrOfThreads - 1;
	insertion_sort(thePivotArray, 0, nrOfThreads - 1);
	int nrOfPivots = nrOfThreads - 1;
	for(p = -1; p < nrOfPivots; p ++)
	{
		for(i = 0; i < theArray -> length; i ++)
		{
			if(p == -1)
			{
				if(theArray -> data[i] < thePivots[0])
				{
					if(theArray -> data[i] != INT_MIN)
					{
						tempList[current_index++] = theArray -> data[i];
						theArray -> data[i] = INT_MIN;
					}
				}
				if(i == theArray -> length - 1)
				{
					tempList[current_index] = thePivots[p + 1];
					thePivotIndices[p + 1] = current_index++;
				}
			}
			else if(p == nrOfPivots - 1)
			{
				if(theArray -> data[i] > thePivots[p])
				{
					if(theArray -> data[i] != INT_MIN)
					{
						tempList[current_index++] = theArray -> data[i];
						theArray -> data[i] = INT_MIN;
					}
				}
				if(i == theArray -> length - 1)
				{
					tempList[current_index] = thePivots[p + 1];
					thePivotIndices[p + 1] = current_index++;
				}
			}
			else
			{
				if(theArray -> data[i] >= thePivots[p] && theArray -> data[i] <= thePivots[p + 1])
				{
					if(theArray -> data[i] != INT_MIN)
					{
						tempList[current_index++] = theArray -> data[i];
						theArray -> data[i] = INT_MIN;
					}
				}
			}
		}
	}
	memcpy(theArray -> data, tempList, sizeof(int) * theArray -> length);
	printf("Innan sort, currentindex: %d\n", current_index);
	for(p = 0; p < nrOfThreads; p ++)
	{
		struct thread_args *theArguments = malloc(sizeof(struct thread_args));
		theArguments -> theArray = theArray;
		theArguments -> start = p > 0 ? thePivotIndices[p - 1] : 0;
		theArguments -> end = p < nrOfThreads - 1 ? thePivotIndices[p] - 1 : theArray -> length - 1;
		printf("Start: %d, Stop: %d\n", theArguments -> start, theArguments -> end);
		pthread_create(&threads[p], NULL, (void*)(quick_sort), (void*)(theArguments));
	}
	
	printf("Innan join\n");
	for(p = 0; p < nrOfThreads; p ++)
		pthread_join(threads[p], NULL);
	printf("Efter join\n");
}



















