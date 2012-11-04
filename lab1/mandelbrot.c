#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <malloc.h>

#include "mandelbrot.h"
#include "ppm.h"
#include "stack.h"

#ifdef MEASURE
#include <time.h>
// If we measure, we don't debug as assert() and printf() seriously affect performance
#undef DEBUG
#endif

// Disable assertion code if DEBUG is not defined
#ifndef DEBUG
#define NDEBUG
#endif

#if NB_THREADS > 0
#include <pthread.h>
#endif

#if LOADBALANCE == 1
	const int ROWS_PER_TASK = 1;
#endif

#if LOADBALANCE == 2
	const int ROWS_PER_TASK = 5;
	#define LOADBALANCE 1
#endif

int * color = NULL;

#if NB_THREADS > 0
// Compiled only when several threads are used
struct mandelbrot_thread
{
	int id;
#ifdef MEASURE
struct mandelbrot_timing timing;
#endif
};

int thread_stop;
pthread_barrier_t thread_pool_barrier;
pthread_barrier_t thread_para_barrier;

pthread_t thread[NB_THREADS];
struct mandelbrot_thread thread_data[NB_THREADS];
#else
#ifdef MEASURE
struct mandelbrot_timing sequential;
#endif
#endif

#ifdef MEASURE
struct mandelbrot_timing **timing;
#endif

struct mandelbrot_param mandelbrot_param;

static int num_colors(struct mandelbrot_param* param)
{
	return param->maxiter + 1;
}

/**
 * Calculates if the complex number (Cre, Cim)
 * belongs to the Mandelbrot set
 *
 * @param Cre: Real part
 *
 * @param Cim: Imaginary part
 *
 * @return : MAXITER if (Cre, Cim) belong to the
 * mandelbrot set, else the number of iterations
 */
static int
is_in_Mandelbrot(float Cre, float Cim, int maxiter)
{
	int iter;
	float x = 0.0, y = 0.0, xto2 = 0.0, yto2 = 0.0, dist2;

	for (iter = 0; dist2 < 4 && iter <= maxiter; iter++)
	{
		y = x * y;
		y = y + y + Cim;
		x = xto2 - yto2 + Cre;
		xto2 = x * x;
		yto2 = y * y;

		dist2 = xto2 + yto2;
	}
	return iter;
}

/***** You may modify this portion *****/
static void
compute_chunk(struct mandelbrot_param *args, int offseti, int offsetj)
{
	int i, j, val;
	float Cim, Cre;
	color_t pixel;
	
	//#pragma omp parallel for schedule(dynamic, 1) private(i,j,pixel,Cim,Cre,val)  
	for (i = 0; i < args->height; i++)
	{
		for (j = 0; j < args->width; j++)
		{
			// Convert the coordinate of the pixel to be calculated to both
			// real and imaginary parts of the complex number to be checked
			Cim = (float) i / args->height * (args->upper_i - args->lower_i)
			    + args->lower_i;
			Cre = (float) j / args->width * (args->upper_r - args->lower_r)
			    + args->lower_r;

			// Gets the value returned by is_in_mandelbrot() and scale it
			// from 0 to 255, or -1 if (Cre, Cim) is in the mandelbrot set.
			val = is_in_Mandelbrot(Cre, Cim, args->maxiter);

			// Change a negative value to 0 in val to make mandelbrot
			// elements to appear black in the final picture.
			val = val > args->maxiter ? args->mandelbrot_color : color[val
			    % num_colors(args)];

			// Permute green, red and blue to get different fancy colors
			pixel.green = val >> 16 & 255;
			pixel.red = val >> 8 & 255;
			pixel.blue = val & 255;
			ppm_write(args->picture, j + offsetj, i + offseti, pixel);
		}
	}
}

void
init_round()
{
	// Initialize or reinitialize here variables before any thread starts or restarts computation

}

#if NB_THREADS > 0
void
parallel_mandelbrot(struct mandelbrot_thread *args, struct mandelbrot_param *parameters)
{
#if LOADBALANCE == 1 // Use a stack
	
	if(args -> id == 0)
	{
		initStack();
	}
	pthread_barrier_wait(&thread_para_barrier);
	
	int i;
	int RowsPerThread = ROWS_PER_TASK;
	float threadsInv = (float)ROWS_PER_TASK / (float)mandelbrot_param.height;
	float irange = mandelbrot_param.upper_i - mandelbrot_param.lower_i;
	int tasksPerThread = (mandelbrot_param.height / ROWS_PER_TASK) / NB_THREADS;
	
	for(i = tasksPerThread * args -> id; i < tasksPerThread * (args -> id + 1); i ++)
	{
		struct mandelbrot_param tempParam = mandelbrot_param;
		tempParam.height = ROWS_PER_TASK;
		tempParam.lower_i = irange * threadsInv * (float)i + mandelbrot_param.lower_i;
		tempParam.upper_i = irange * threadsInv * (float)(i + 1) + mandelbrot_param.lower_i;
		pushStack(&tempParam, ROWS_PER_TASK * i);
	}
#endif
	
#if LOADBALANCE == 0
	int RowsPerThread = parameters -> height / NB_THREADS;
	float range = (float)RowsPerThread / (float)parameters -> height;
	
	float threadsInv = 1.0f / (float)NB_THREADS;
	float irange = parameters -> upper_i - parameters -> lower_i;
	
	struct mandelbrot_param tempParam = *parameters;
	tempParam.height = parameters -> height / NB_THREADS;
	tempParam.lower_i = irange * threadsInv * (float)args -> id + parameters -> lower_i;
	tempParam.upper_i = irange * threadsInv * (float)(args -> id + 1) + parameters -> lower_i;
	
	compute_chunk(&tempParam, RowsPerThread * args -> id, 0);
#endif
#if LOADBALANCE == 1
	while (1)
	{
		struct mandelbrot_param_offset* theParamOffset = popStack();
		if(theParamOffset == NULL)
			break;
		compute_chunk(&(theParamOffset -> theParam), theParamOffset -> heightOffset, 0);
	}
#endif
}
#else
void
sequential_mandelbrot(struct mandelbrot_param *parameters)
{
	compute_chunk(parameters, 0, 0);
}
#endif
/***** end *****/

// Thread code, compiled only if we use threads
#if NB_THREADS > 0
static void *
run_thread(void * buffer)
{
	struct mandelbrot_thread *args;
	args = (struct mandelbrot_thread*) buffer;

	// Notify the master this thread is spawned
	pthread_barrier_wait(&thread_pool_barrier);

	// Reinitialize vars before any thread restart
	init_round();

	// Wait for the first computation order
	pthread_barrier_wait(&thread_pool_barrier);

	while (thread_stop == 0)
	{
#ifdef MEASURE
		clock_gettime(CLOCK_MONOTONIC, &args->timing.start);
#endif

	parallel_mandelbrot(args, &mandelbrot_param);

#ifdef MEASURE
		clock_gettime(CLOCK_MONOTONIC, &args->timing.stop);
#endif

		// Notify the master thread of completion
		pthread_barrier_wait(&thread_pool_barrier);

		// Reinitialize vars before any thread restart
		init_round();

		// Wait for the next work signal
		pthread_barrier_wait(&thread_pool_barrier);
	}

	// Notify the master thread of work completion
	pthread_barrier_wait(&thread_pool_barrier);

	return NULL;
}
#endif

void
init_ppm(struct mandelbrot_param* param)
{
	if(param->picture->data != NULL)
	{
		free(param->picture->data);
		param->picture->data = NULL;
	}

	param->picture->data = malloc(ppm_align(sizeof(color_t) * param->width, PPM_ALIGNMENT) * param->height);
	param->picture->height = param->height;
	param->picture->width = param->width;
}

void
update_colors(struct mandelbrot_param* param)
{
	// Gradient color
	int start_r, start_g, start_b, end_r, end_g, end_b;
	// Other control variables
	int i;

	if(color != NULL)
	{
		free(color);
	}
	color = malloc(sizeof(int) * num_colors(param));

	// Start color
	start_r = 219;
	start_g = 57;
	start_b = 0;

	// Stop color
	end_r = 0;
	end_g = 0;
	end_b = 0;

	// Initialize the color vector
	for (i = 0; i < num_colors(param); i++)
	{
		color[i]
		    = (((int) ((end_g - start_g) * ((double) i / num_colors(param)) + start_g)
		        & 255) << 16) + (((int) ((end_r - start_r) * ((double) i
		        / num_colors(param)) + start_r) & 255) << 8) + (((int) ((end_b - start_b)
		        * ((double) i / num_colors(param)) + start_b) & 255) << 0);
	}
}

void
init_mandelbrot(struct mandelbrot_param *param)
{
	// Initialize the picture container, but not its buffer
	param->picture = ppm_alloc(0, 0);
	param->picture->height = param->height;
	param->picture->width = param->width;

#if GLUT != 1
	// GLUT will do it when creating or resizing its window
	init_ppm(param);
#endif
	// initialize the color vector
	update_colors(param);

#if NB_THREADS > 0
	// Thread-based variant

	pthread_attr_t thread_attr;
	int i;

	// Initialise thread poll / master thread synchronisation
	pthread_barrier_init(&thread_pool_barrier, NULL, NB_THREADS + 1);
	pthread_barrier_init(&thread_para_barrier, NULL, NB_THREADS);

	// Initialize attributes
	pthread_attr_init(&thread_attr);
	pthread_attr_setdetachstate(&thread_attr, PTHREAD_CREATE_JOINABLE);

	// Enables thread running
	thread_stop = 0;

#ifdef MEASURE
	// Measuring record structures
	timing = malloc(sizeof(struct timing*) * NB_THREADS);
#endif

	// Create a thread pool
	for (i = 0; i < NB_THREADS; i++)
	{
		thread_data[i].id = i;

#ifdef MEASURE
		timing[i] = &thread_data[i].timing;
#endif

		// Check the good behavior or pthread_create; must be disabled while measuring for performance reasons
#ifdef DEBUG
		assert(pthread_create(&thread[i], &thread_attr, &run_thread, &thread_data[i]) == 0);
#else
		pthread_create(&thread[i], &thread_attr, &run_thread, &thread_data[i]);
#endif
	}

	// Wait for the thread to be fully spawned before returning
	pthread_barrier_wait(&thread_pool_barrier);
#else
#ifdef MEASURE
	// Measuring record structures
	timing = malloc(sizeof(struct timing*));
	timing[0] = &sequential;
#endif
#endif
}

#ifdef MEASURE
struct mandelbrot_timing**
#else
void
#endif
compute_mandelbrot(struct mandelbrot_param param)
{
#if NB_THREADS > 0
	mandelbrot_param = param;

	// Trigger threads' resume
	pthread_barrier_wait(&thread_pool_barrier);

	// Wait for the threads to be done
	pthread_barrier_wait(&thread_pool_barrier);
#else
#ifdef MEASURE
	clock_gettime(CLOCK_MONOTONIC, &sequential.start);
#endif

	init_round();
	sequential_mandelbrot(&param);

#ifdef MEASURE
	clock_gettime(CLOCK_MONOTONIC, &sequential.stop);
#endif
#endif

#ifdef MEASURE
	return timing;
#endif
}

void
destroy_mandelbrot(struct mandelbrot_param param)
{
#if NB_THREADS > 0
	int i;

	// Initiate a stop order and resume threads in the thread pool
	thread_stop = 1;
	compute_mandelbrot(param);

	// Wait for the threads to finish
	for (i = 0; i < NB_THREADS; i++)
	{
#ifdef DEBUG
		assert(pthread_join(thread[i], NULL) == 0);
#else
		pthread_join(thread[i], NULL);
#endif
	}

	pthread_barrier_destroy(&thread_pool_barrier);

#else
	// Sequential version, nothing to destroy
#endif

#ifdef MEASURE
	free(timing);
#endif

	free(color);
	ppm_free(param.picture);
}
