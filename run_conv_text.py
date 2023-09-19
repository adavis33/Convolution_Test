# The script that runs the convolution tests

import numpy as np
import time as time
import tabulate as tabulate
import matplotlib.pyplot as plt
import os
import convolution as conv


##############################################################################
#### Options ####

# The dimensions of the test data fields. An array of this size will be filled
# with random numbers from [0, 1).
Y, X = 60, 60

# If True, Pad the data fields with np.nan. If False, no padding is added.
Pad = True

# How much padding to add to each data field
y, x = 5, 5

# Size of the Gaussian filter in a square domain (should be an odd int)
g = 15

# The number of data fields you want to test the convolution on
Num_img = 100

# Save some of the convolution plots
save_plots = False


##############################################################################
#### Input ####

# Path to the Figures dir
cwp = os.getcwd()
img_path = cwp + '/Figures/'

# Convolution methods
methods = ['exp_conv', 'list_comp_conv', 'fft_conv', 'prime_fact_fft', 'scipy']


##############################################################################
#### Code Body ####

# Create Num_img 2D arrays of random numbers of size Y, X
test_data = []
for i in range(Num_img):
    img = np.random.rand(Y, X)
    if Pad == True:
        img = np.pad(img, (y, x), constant_values = np.nan)
    test_data.append(img)

# Create an instance of the convolution class
conv1 = conv.convolution(DATA = test_data[0], FILTER_WIDTH = g)


#### Run the tests ####

# Create a dict to store the filtered fields
filtered_data = {}
for key in methods:
    filtered_data[key] = None

times = []
# Loop over the different methods
for m in methods:
    fd = []
    
    start = time.time() # Start the clock
    for i, data in enumerate(test_data):
        fd.append(conv1.Convolution2D(DATA = data,
                                      METHOD = m,
                                      PADDED = Pad)) 
    end = time.time() # Stop the clock
    
    times.append(round(end - start, 6))
    filtered_data[m] = fd



##############################################################################
#### Display the Results ####

#### Parameters ####
print(' ')
print('Computation Parameters:')
print(f'g = {g}, (X, Y) = ({X}, {Y})')
if Pad == True:
    print(f'Padding = {Pad}, (x, y) = ({x}, {y})')
print(f'Number of data fields = {Num_img}')

#### Time ####

# Display the times as terminal output table
Methods = methods.copy()
Methods.insert(0, 'Method:')
times.insert(0, 'Time (s):')

# Make the table for tabulate
table = [Methods, times]
# Print the table
print(' ')
print(f'Method times:')
print(tabulate.tabulate(table))
print(' ')


#### Acuracy ####
control_method = 'exp_conv'
control = filtered_data[control_method]

# Calculate the mean square error
mse = []
for m in methods:
    if m != control_method:
        est = filtered_data[m]
        mse.append(round(conv1.MSE(CONTROL = control,
                                   EST = est,
                                   IMG_SET = True), 8))
    else:
        # The method is the control method
        mse.append('NA')

# Make the table for tabulate
mse.insert(0, 'MSE:')
table = [Methods, mse]
# Print the table
print(' ')
print(f'Mean Square Error:')
print(tabulate.tabulate(table))
print(' ')

# Make some plots of the convolved fields
if save_plots == True:
    # Plot the filtered fields
    for m in methods:
        img = filtered_data[m][0]
        plt.imshow(img, cmap = 'plasma')
        plt.colorbar()
        plt.title(f'{m}')
        plt.xlabel('x grid')
        plt.ylabel('y grid')
        plt.savefig(img_path + f'{m}.png')
        plt.close()

    # Plot a unfiltered field
    plt.imshow(test_data[0], cmap = 'plasma')
    plt.colorbar()
    plt.title('Raw Field')
    plt.xlabel('x grid')
    plt.ylabel('y grid')
    plt.savefig(img_path + 'RawField.png')
    plt.close()
    

