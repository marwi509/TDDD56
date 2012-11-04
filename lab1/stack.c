#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <malloc.h>
#include "stack.h"
#include <pthread.h>

int currentHead = -1;
pthread_mutex_t thePushPopMutex;
typedef struct mandelbrot_param_offset mandelbrot_param_offset;
typedef struct mandelbrot_param mandelbrot_param;
mandelbrot_param_offset* theStackArray[500];

void initStack()
{
	pthread_mutex_init(&thePushPopMutex, NULL);
}

mandelbrot_param_offset* popStack()
{
	pthread_mutex_lock(&thePushPopMutex);
	mandelbrot_param_offset* returnValue = NULL;
	if(currentHead > -1)
	{
		returnValue = theStackArray[currentHead--];
	}
	pthread_mutex_unlock(&thePushPopMutex);
	return returnValue;
}

void pushStack(mandelbrot_param* parametersIn, int heightOffset)
{
	pthread_mutex_lock(&thePushPopMutex);  // lock mutex
	mandelbrot_param_offset* theParam = malloc(sizeof(mandelbrot_param_offset));
	theParam -> theParam = *parametersIn;
	theParam -> heightOffset = heightOffset;
	theStackArray[++currentHead] = theParam;
	
	pthread_mutex_unlock(&thePushPopMutex); //unlock mutex
}

void clearStack()
{
	currentHead = -1;
}
