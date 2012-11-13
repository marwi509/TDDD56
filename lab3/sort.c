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
//#include "custom_sort.h"

void quick_sort(struct array* theArray, int start, int end);
int
sort(struct array * array)
{
	
	quick_sort(array, 0, array -> length - 1);
	
	return 0;
}



#include <stdlib.h>
#include <string.h>
#include <pthread.h>


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

void insertion_sort(struct array* theArray, int start, int end)
{
	int i;
	for(i = start + 1; i <= end; i ++)
	{
		value item = theArray -> data[i];
		int iHole = i;
		while(iHole > start && theArray -> data[iHole - 1] > item)
		{
			theArray -> data[iHole] = theArray -> data[iHole - 1];
			iHole = iHole - 1;
		}
		theArray -> data[iHole] = item;
	}
}

void merge_lists(struct array* theArray, int start_left, int end_left, int start_right, int end_right)
{
	int current_left = start_left;
	int current_right = start_right;
	int current_index = 0;
	int total_size = (end_right - start_left + 1);
	value* tempArray = malloc(sizeof(value) * total_size);
	
	// Loop until both lists are done
	while(1)
	{
		if(	(current_left != start_right &&
			theArray -> data[current_left] <= theArray -> data[current_right]) ||
			current_right == end_right + 1
			)
		{
			tempArray[current_index] = theArray -> data[current_left];
			current_index++;
			current_left++;
		}
		else
		{
			tempArray[current_index] = theArray -> data[current_right];
			current_index++;
			current_right++;
		}
		if((current_left == start_right && current_right == end_right + 1) || current_index == total_size)
			break;
	}
	memcpy(theArray -> data + start_left, tempArray, sizeof(value) * total_size);
	free(tempArray);
}

void merge_sort(struct array* theArray, int start, int end, int rec)
{
	if(end <= start)
		return;
	int length = end - start;
	if(rec > 4)
	{
		quick_sort(theArray, start, end);
		return;
	}
	merge_sort(theArray, start, start + length / 2, rec + 1);
	merge_sort(theArray, start + length / 2 + 1, end, rec + 1);
	merge_lists(theArray, start, start + length / 2 ,start + length / 2 + 1,  end);
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
		quick_sort_impl(theArray, start, pivot_final - 1);
		quick_sort_impl(theArray, pivot_final + 1, end );
	}
}

void quick_sort(struct array* theArray, int start, int end)
{
	if(check_sorted(theArray, start, end))
	{
		return;
	}
	if(check_desc_sorted(theArray, start, end))
	{
		reverse_array(theArray, start, end);
		return;
	}
	quick_sort_impl(theArray, start, end);
}



