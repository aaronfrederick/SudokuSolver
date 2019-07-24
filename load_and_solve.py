import numpy as np
import time
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import cv2
from PIL import Image
import pytesseract
import tensorflow as tf
import sys

#the global...
subtract_set = {1,2,3,4,5,6,7,8,9}

def instantiate_model(filename='image_to_number_model.hdf5'):
	model = tf.keras.models.Sequential()
	model.add(tf.keras.layers.Conv2D(254, kernel_size=(3,3), input_shape=(28,28, 1)))
	model.add(tf.keras.layers.MaxPool2D((2,2)))
	model.add(tf.keras.layers.Conv2D(128, kernel_size=(3,3)))
	model.add(tf.keras.layers.MaxPool2D((2,2)))
	model.add(tf.keras.layers.Flatten())
	model.add(tf.keras.layers.Dense(140, activation='relu'))
	model.add(tf.keras.layers.Dropout(0.2))
	model.add(tf.keras.layers.Dense(80, activation='relu'))
	model.add(tf.keras.layers.Dropout(0.2))
	model.add(tf.keras.layers.Dense(units=10, activation='sigmoid'))

	model.load_weights(filename)
	return model

def load_image(filename):
	#import and preprocess image
	img = Image.open(filename).convert('LA')
	array = np.array(img)[:,:,0]
	array = 255-array
	divisor = array.shape[0]//9

	puzzle = []
	for i in range(9):
	    row = []
	    for j in range(9):
	        #slice image, reshape it to 28x28 (mnist reader size)
	        row.append(cv2.resize(array[i*divisor:(i+1)*divisor,
	                                    j*divisor:(j+1)*divisor][3:-3, 3:-3], 
	                              dsize=(28,28), 
	                              interpolation=cv2.INTER_CUBIC))
	    puzzle.append(row)

	model = instantiate_model()

	template = [
	    [0 for _ in range(9)] for _ in range(9)
	]

	for i, row in enumerate(puzzle):
	    for j, spot in enumerate(row):
	        if np.mean(spot) > 6:
	            template[i][j] = model.predict_classes(spot.reshape(1,28,28,1).astype('float32')/255)[0]
	return template

###  Defining Solving Functions  ###
def check_horizontal(i,j):
    return subtract_set - set(container[i])

def check_vertical(i,j):
    ret_set = []
    for x in range(9):
        ret_set.append(container[x][j])
    return subtract_set - set(ret_set)

def check_square(i,j):
    first = [0,1,2]
    second = [3,4,5]
    third = [6,7,8]
    find_square = [first,second,third]
    for l in find_square:
        if i in l:
            row = l
        if j in l:
            col = l
    ret_set = []
    for x in row:
        for y in col:
            ret_set.append(container[x][y])
    return subtract_set - set(ret_set)

def get_poss_vals(i,j):
    poss_vals = list(check_square(i,j).intersection(check_horizontal(i,j)).intersection(check_vertical(i,j)))
    return poss_vals

def explicit_solver(container):
    stump_count = 1
    for i in range(9):
        for j in range(9):
            if container[i][j] == 0:
                poss_vals = get_poss_vals(i,j)
                if len(poss_vals) == 1:
                    container[i][j] = list(poss_vals)[0]
                    #print_container(container)
                    stump_count = 0
    return container, stump_count

def implicit_solver(i,j,container):
    if container[i][j] == 0:
        poss_vals = get_poss_vals(i,j)
        
        #check row
        row_poss = []
        for y in range(9):
            if y == j:
                continue
            if container[i][y] == 0:
                for val in get_poss_vals(i,y):
                    row_poss.append(val)
        if len(set(poss_vals)-set(row_poss)) == 1:
            container[i][j] = list(set(poss_vals)-set(row_poss))[0]
            #print_container(container)
        
        #check column
        col_poss = []
        for x in range(9):
            if x == i:
                continue
            if container[x][j] == 0:
                for val in get_poss_vals(x,j):
                    col_poss.append(val)
        if len(set(poss_vals)-set(col_poss)) == 1:
            container[i][j] = list(set(poss_vals)-set(col_poss))[0]
            #print_container(container)
                
        #check square
        first = [0,1,2]
        second = [3,4,5]
        third = [6,7,8]
        find_square = [first,second,third]
        for l in find_square:
            if i in l:
                row = l
            if j in l:
                col = l
        square_poss = []
        for x in row:
            for y in col:
                if container[x][y] == 0:
                    for val in get_poss_vals(x,y):
                        square_poss.append(val)
        if len(set(poss_vals)-set(square_poss)) == 1:
            container[i][j] = list(set(poss_vals)-set(square_poss))[0]
            #print_container(container)
    return container

def print_container(container):
    for i, row in enumerate(container):
        for j, val in enumerate(row):
            if (j)%3 == 0 and j<8 and j>0:
                print("|",end=' ')
            print(val,end=' ')
        print()
        if (i-2)%3 == 0 and i<8:
            print("_____________________", end='')
            print()
        print()
    print()
    print("||||||||||||||||||||||")
    print()

if __name__=='__main__':
	# using explicit solver
	start = time.time()
	container = load_image(sys.argv[1])
	zero_count = 0
	for l in container:
	    for v in l:
	        if v == 0:
	            zero_count += 1
	            
	print(f'There are {zero_count} moves I have to make!')
	print()
	print_container(container)
	solving = True

	while solving:
	    #Solver Portion
	    container, stump_count = explicit_solver(container)
	    
	    #Loop-Breaking Portion
	    zero_count = 0
	    for l in container:
	        for v in l:
	            if v == 0:
	                zero_count += 1
	    if zero_count==0:
	        # print_container(container)
	        solving=False
	    if stump_count > 0:
	        for i in range(9):
	            for j in range(9):
	                container = implicit_solver(i,j,container)
	print()
	print_container(container)
	print('That took', round(time.time()-start, 3), 'seconds!')
