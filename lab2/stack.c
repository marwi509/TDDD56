/*
 * stack.c
 *
 *  Created on: 18 Oct 2011
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
 *     but WITHOUT ANY WARRANTY without even the implied warranty of
 *     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *     GNU General Public License for more details.
 * 
 *     You should have received a copy of the GNU General Public License
 *     along with TDDD56. If not, see <http://www.gnu.org/licenses/>.
 * 
 */

#ifndef DEBUG
#define NDEBUG
#endif

#include <assert.h>
#include <pthread.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>

#include "stack.h"
#include "non_blocking.h"

#if NON_BLOCKING == 0
#warning Stacks are synchronized through locks
#else
#if NON_BLOCKING == 1 
#warning Stacks are synchronized through lock-based CAS
#else
#warning Stacks are synchronized through hardware CAS
#endif
#endif

stack_t *
stack_alloc()
{
  // Example of a task allocation with correctness control
  // Feel free to change it
  stack_t *res;

  res = malloc(sizeof(stack_t));
  assert(res != NULL);

  if (res == NULL)
    return NULL;
  res -> head = NULL;
  
// You may allocate a lock-based or CAS based stack in
// different manners if you need so
#if NON_BLOCKING == 0
	pthread_mutex_init(&res -> theMutex, NULL);
  // Implement a lock_based stack
#else
  // Implement a CAS-based stack
#endif

  return res;
}

int
stack_init(stack_t *stack, size_t size)
{
  assert(stack != NULL);
  assert(size > 0);
  stack -> sizeOfElement = size;
#if NON_BLOCKING == 0
  
#else
  // Implement a CAS-based stack
#endif

  return 0;
}

int
stack_check(stack_t *stack)
{
  assert(stack != NULL);

  return 0;
}

int
stack_push_safe(stack_t *stack, void* buffer)
{
#if NON_BLOCKING == 0
  // Lock-based push
  pthread_mutex_lock(&stack -> theMutex);


	struct element* theNewElement = malloc(sizeof(struct element));
	theNewElement -> theData = malloc(stack -> sizeOfElement);
	memcpy(theNewElement -> theData, buffer, stack -> sizeOfElement);
	theNewElement -> next = stack -> head;
	stack -> head = theNewElement;
	
  pthread_mutex_unlock(&stack -> theMutex);
#else
	// Implement a CAS-based stack
	struct element* theNewElement;
	theNewElement = malloc(sizeof(struct element));
	theNewElement -> theData = malloc(stack ->sizeOfElement);
	memcpy(theNewElement -> theData,buffer,stack -> sizeOfElement);
	do
	{
		struct element* old = stack->head;
		theNewElement -> next = old;
		
	}while(!cas(stack->head,old,theNewElement));
	//stack -> head = theNewElement;
#endif

  return 0;
}

int
stack_pop_safe(stack_t *stack, void* buffer)
{
#if NON_BLOCKING == 0
  // Implement a lock_based stack
  pthread_mutex_lock(&stack -> theMutex);
	if(stack -> head != NULL)
	{
		struct element* theOldHead = stack -> head;
		stack -> head = stack -> head -> next;
		memcpy(buffer, &theOldHead -> theData, stack -> sizeOfElement);
		free(theOldHead -> theData);
		free(theOldHead);
	}
	else
		return -1;
  pthread_mutex_unlock(&stack -> theMutex);
#else
  // Implement a CAS-based stack
  struct element* newHead;
  struct element* theOldHead;
	do
	{
		theOldHead = stack -> head;
		memcpy(buffer, &stack -> head -> theData, stack -> sizeOfElement);
		newHead = theOldHead -> next;
	}while(!cas(stack -> head,theOldHead,newHead));
	free(theOldHead -> theData);
	free(theOldHead);
	
	
#endif

  return 0;
}

