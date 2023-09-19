# This file contains the convolution class and the support tools needed to
# test the convolution methods.

#################################################################################
#### Imports ####
import numpy as np
import scipy as sp
from scipy import signal 
from math import sqrt
from itertools import count, islice


#################################################################################
#### Convolution Class ####


class convolution:

    def __init__(self, DATA, FILTER_WIDTH):
        '''
        The convolution class demonstrates different convolution methods.

    	Create an intance of the class by passing it a representative data 
        field and a filter kernel width.
    	EX:
        	import convolution as conv
        	conv1 = conv.convolution(DATA = my_data, 
                                         FILTER_WIDTH = 15)

    	You can calculate the convolution of multiple data fields with one
    	instance of the class provided all of the data fields have the same
    	shape and padding.
    	EX:
        	import convolution as conv
        	DATA = [img1, img2, img3]
        	conv1 = conv.convolution(DATA = my_data[0], 
                                         FILTER_WIDTH = 15)
        	C = [conv1.Convolution2D(DATA = img) for img in my_data]

    	Input:
    	DATA [2D numpy array or 2D list]: The data field to be filtered.
    	FILTER_WIDTH [odd int]: The side length of the 2D Gaussian filter 
            kernel that will be convolved with DATA. This should be an odd 
            integer so that the filter kernel can be centered on a grid point 
            in the DATA array. If FILTER_WIDTH is even, the Gaussian() method 
            will increase it by 1.

    	Instance Variables:
    	A [2D numpy array]: The 2D data field to be filtered.
    	G [2D numpy array]: The filter or convolution kernel.
    	y [int]: Number of grid points in the vertical direction of DATA.
    	x [int]: Number of grid points in the horizontal direction of DATA.
    	m [int]: Number of grid points in the vertical direction of FILTER.
    	n [int]: Number of grid points in the horizontal direction of FILTER.
    	M [int]: Half the number of grid points in the vertical direction of
        	FILTER rounded down.
    	N [int]: Half the number of grid points in the horizontal direction of
        	FILTER rounded down.
        '''
        self.A = np.array(DATA)
        self.G = convolution.Gaussian(WIDTH = FILTER_WIDTH)
        self.y, self.x = self.A.shape
        self.m, self.n = self.G.shape
        self.M = int(self.m/2)
        self.N = int(self.n/2)



    def exp_conv(self, DATA, PADDED = False):
        '''
        Compute the explicit convolution with no algorithmic improvements or
        attempts at speeding the computation up.

        Input:
        DATA [2D numpy array]: The data field you want filtered.
        PADDED [Bool]: If True, DATA is nan padded. If False, DATA is not nan
            padded.
   	 
        Output:
        C [2D numpy array]: The filtered field. The convolution of DATA with
            the filter kernel, self.G, using an explicit double for loop method.
        '''

        # If DATA is padded, find the edges and extract the data
        if PADDED == True:
            # Remove the nan padding
            A_data, y_Adata_max, y_Adata_min, x_Adata_max, x_Adata_min \
                = convolution.find_index(DATA)
        else:
            # DATA is not padded so set the upper left corner of the field
            # within the padded array to 0, 0 (ie. no padding)
            y_Adata_min = 0
            x_Adata_min = 0
            A_data = DATA

        # Now pad the extracted data by half the filter width on each side.
        # This is padded by the data on the edge of the data field.
        A_pad = np.pad(A_data, (self.M, self.N), 'symmetric')
        h, k = A_data.shape

        # Calculate the convolution 
        C = np.full((self.y, self.x), np.nan)
        # Loop over the unpadded data plus half the filter width
        for j in range(self.M, h + self.M):
            for i in range(self.N, k + self.N):
                # Extract a region of the data the same size as the filter
                A_sub = A_pad[j - self.M : j + self.M + 1,
                              i - self.N : i + self.N + 1]
                
                # Store the convolved value back in C
                C[j + y_Adata_min - self.M][i + x_Adata_min - self.N] = np.sum(
                    np.multiply(A_sub, self.G))

        return C



    def list_comp_conv(self, DATA):
        '''
        !!! This method is for demostration only. Do not use this method. !!!
        
        An example of what not to do. This method uses Pythons list comprehension 
        to loop over the data fields and compute an explicit convolution. List 
        comprehension is usful and fast for simple conditions, but can get 
        overly complicated very quickly.

        For the cases tested in run_conv_test.py, this method is also slower 
        than the exp_conv() method.

        Input:
        DATA [2D numpy array]: The field you want filtered.

        Output:
        C [2D numpy array]: The filtered field. The convolution of DATA with 
            the filter kernel, self.G, using list comprehension.

        '''
        
        # Extract the data. For non-padded arrays this gives the shape
        A_data, y_Adata_max, y_Adata_min, x_Adata_max, x_Adata_min \
            = convolution.find_index(DATA)
        A_pad = np.pad(A_data, (self.M, self.N), 'symmetric') # Pad the data

        #### Do Not Write List Code Like This ####
        # Calculate the convolution... The following line is to complex to
        # read and should not be used
        C = np.asarray([[np.sum(A_pad[j-y_Adata_min:j-y_Adata_min+2*self.M+1, 
            i-x_Adata_min:i-x_Adata_min+2*self.N+1] * self.G) if x_Adata_min <= 
            i <= x_Adata_max and y_Adata_min <= j <= y_Adata_max else 
            np.nan for i in range(self.x)] for j in range(self.y)])
        #### Do Not Write List Code Like This ####

        return C



    def fft_conv(self, DATA, PADDED = False):
        '''
        Compute the convolution using a fast Fourier transform 
        (numpy.fft.fft2()).

        Input:
        DATA [2D numpy array or 2D list]: The field you want filtered.
        PADDED [bool]: If True, DATA is nan padded. If False, DATA is not nan
            padded.

        Output:
        C [2D numpy array]: The filtered field. The convolution of DATA with 
            the filter kernel, self.G, using a fast Fourier transform.
        '''
        
        if PADDED == True:
            # Remove nan padding and extract data
            A_data, y_Adata_max, y_Adata_min, x_Adata_max, x_Adata_min \
                = convolution.find_index(DATA)
        else:
            A_data = DATA

        # Pad the data field
        A_pad = np.pad(A_data, (self.M, self.N), 'symmetric')
        h, k = A_pad.shape

        # Pad the filter with 0's to the same size as the data
        Ly_bound = int((h - self.m)/2.)
        Lx_bound = int((k - self.n)/2.)

        # Check to see if data bounds are odd or even. This fixes an n or 
        # n+1 shift in the numpy.fft.fftshift() method
        if (h - self.m) % 2 == 1:
            Ry_bound = Ly_bound + 1
        else:
            Ry_bound = Ly_bound

        if (k - self.n) % 2 == 1:
            Rx_bound = Lx_bound + 1
        else:
            Rx_bound = Lx_bound

        # Zero pad the filter
        bounds = [(Ly_bound, Ry_bound), (Lx_bound, Rx_bound)]
        G_pad = np.pad(self.G, bounds, mode = 'constant')

        # Do the convolution with fft
        # rfft (which is faster) creates diffrent n or n+1 shifts here
        FTC = np.fft.fft2(A_pad) * np.fft.fft2(G_pad)
        FTP = np.fft.fftshift(np.fft.ifft2(FTC))
        C_unpadded = np.sign(FTP.real)*np.absolute(FTP)

        if PADDED == True:
            # Pad the edges with nan's again
            C_cut = C_unpadded[self.M - 1 : h - self.M - 1,
                               self.N - 1 : k - self.N - 1]
            y_max = self.y - y_Adata_max - 1
            x_max = self.x - x_Adata_max - 1
            C = np.pad(C_cut,
                       [(y_Adata_min, y_max), (x_Adata_min, x_max)],
                       mode = 'constant',
                       constant_values = np.nan)
        else:
            # Cut the data back to the original size
            C = C_unpadded[self.M - 1: h - self.M - 1,
                           self.N - 1: k - self.N - 1]

        return C



    def prime_fact_fft(self, DATA, PADDED = False):
        '''
        When the side lengths of the unpadded data field plus the filter kernel 
        side length is prime, the numpy fft slows down significantly. This 
        method checks the side lengths and adds one row if the sum is prime. 
        This speed-up a convolution but introduces a small error to the result.

        This method is useful for filtering large sets of images where the 
        accuracy of the final result is less important than speed, such as 
        blurring many images.

        Input:
        DATA [2D numpy array]: The field you want filtered.
        PADDED [bool]: If True, DATA is nan padded. If False, DATA is not nan
        	padded.

        Output:
        C [2D numpy array]: The filtered field. The convolution of DATA with
        	the filter kernel, self.G, using a prime factor fast Fourier
        	transform.
        '''

        def isprime(NUM):
            '''
            Check if NUM is prime.

            Input:
            NUM [int]: An integer.

            Output:
            [bool]: Returns True if num is prime, otherwise False.
            '''
            # Returns True if NUM is prime
            if NUM < 2:
                return False
            for number in islice(count(2), int(sqrt(NUM) - 1)):
                if NUM % number == 0:
                    return False
            return True

        if PADDED == True:
            # Remove nan padding and extract data
            A_data, y_Adata_max, y_Adata_min, x_Adata_max, x_Adata_min \
                = convolution.find_index(DATA)
        else:
            A_data = DATA
    
        # Check to see if a side lengths of the array are prime
        v, u = A_data.shape # Shape of the trimmed array
        if isprime(v + 2*self.M):
            # If prime, we pad the edge of the data array by one row/column
            pad_M_edge = True
            M_edge = self.M + 1
        else:
            pad_M_edge = False
            M_edge = self.M

        if isprime(u + 2*self.N):
            pad_N_edge = True
            N_edge = self.N + 1 
        else:
            pad_N_edge = False
            N_edge = self.N

        # Pad the data field
        A_pad = np.pad(A_data, [(self.M, M_edge), (self.N, N_edge)], 'symmetric')
        h, k = A_pad.shape

        # Pad the filter with 0's to the same size as the data
        Ly_bound = int((h - self.m)/2.)
        Lx_bound = int((k - self.n)/2.)

        # Check to see if data bounds are odd or even. This fixes an n or 
        # n+1 shift in the numpy.fft.fftshift() method
        if (h - self.m) % 2 == 1:
            Ry_bound = Ly_bound + 1
        else:
            Ry_bound = Ly_bound

        if (k - self.n) % 2 == 1:
            Rx_bound = Lx_bound + 1
        else:
            Rx_bound = Lx_bound

        # Zero pad the filter
        bounds = [(Ly_bound, Ry_bound), (Lx_bound, Rx_bound)]
        G_pad = np.pad(self.G, bounds, mode = 'constant', constant_values = 0.)

        # Do the convolution with fft
        FTC = np.fft.fft2(A_pad) * np.fft.fft2(G_pad)
        FTP = np.fft.fftshift(np.fft.ifft2(FTC))
        C_unpadded = np.sign(FTP.real) * np.absolute(FTP)

        # Account for the prime array length correction
        if pad_M_edge == True:
            M_shft = 1
        else:
            M_shft = 0

        if pad_N_edge == True:
            N_shft = 1
        else:
            N_shft = 0

        if PADDED == True:
            C_cut = C_unpadded[self.M - M_shft - 1 : h - M_edge - M_shft - 1,
                               self.N - N_shft - 1 : k - N_edge - N_shft - 1]
            # nan pad the data and return it with the same shape as A
            y_max = self.y - y_Adata_max - 1
            x_max = self.x - x_Adata_max - 1
            C = np.pad(C_cut,
                       [(y_Adata_min, y_max), (x_Adata_min, x_max)],
                       mode = 'constant',
                       constant_values = np.nan)
        else:
            # Cut the data back to the original size
            C = C_unpadded[M_edge - 1 : h - self.M - 1,
                           N_edge - 1 : k - self.N - 1]

        return C



    def scipy_pad_convolve(self, DATA, PADDED = False):
        '''
        Compute the convolution of a nan padded 2D numpy array using 
        scipy.signal.convolve2d().

        Input:
        DATA [2D numpy array]: The field you want filtered.
        PADDED [bool]: If True, DATA is nan padded. If False, DATA is not nan
            padded.

        Output:
        C [2D numpy array]: The filtered field. The convolution of DATA with
            the filter kernel, self.G, using scipys convolve2d method.
        '''
        
        # Check if DATA is nan padded
        if PADDED == True:
            # Remove the nan padding
            A_data, y_Adata_max, y_Adata_min, x_Adata_max, x_Adata_min \
                = convolution.find_index(DATA)
        else:
            A_data = DATA

        # Compute the convolution
        C_unpadded = signal.convolve2d(A_data,
                                       self.G, # The filter kernel
                                       mode = 'same', 
                                       boundary = 'symm')

        if PADDED == True:
            # Add the nan padded edges back
            y_max = self.y - y_Adata_max - 1
            x_max = self.x - x_Adata_max - 1
            C = np.pad(C_unpadded,
                       [(y_Adata_min, y_max), (x_Adata_min, x_max)],
                       mode = 'constant',
                       constant_values = np.nan)
        else:
            C = C_unpadded

        return C



    def Convolution2D(self,
                      DATA,
                      METHOD = 'scipy',
                      PADDED = False):
        '''
        Gives the 2D convolution of a 2D array with the Gaussian filter kernel, 
        self.G, using the specified method.

        Input:
        DATA [2D list or numpy array]: The input data to be filtered.
        METHOD [string]: What convolution method to use. 
            Options:
                'exp_conv': See exp_conv() method
                'list_comp_conv': See list_comp_conv() method
                'fft_conv': See fft_conv() method
                'prime_fact_fft': See prime_fact_fft() method
                'scipy': See scipy_pad_convolve() method
        PADDED [bool]: If True, DATA is nan padded. If False, DATA is not nan
            padded.

        Output:
        C [2D numpy array]: The filtered field. The convolution of DATA with
            the filter kernel, self.G, using the given method.

        '''

        if METHOD == 'exp_conv':
            # Calculate the explicit convolution
            C = self.exp_conv(DATA, PADDED)

        elif METHOD == 'list_comp_conv':
            #### For demonstration only ####
            # Same as exp_conv but using list comprehension
            C = self.list_comp_conv(DATA)
            #### For demonstration only ####

        elif METHOD == 'fft_conv':
            # Calculates the convolution with a fast Fourer transform 
            C = self.fft_conv(DATA, PADDED)

        elif METHOD == 'prime_fact_fft':
            # Calculates the fft convolution with a speed up when the side
            # length of the data field + side length of the filter is prime
            C = self.prime_fact_fft(DATA, PADDED)

        elif METHOD == 'scipy':
            # Calculate scipys convolve2d
            C = self.scipy_pad_convolve(DATA, PADDED)
            
        return C



    @staticmethod
    def Gaussian(WIDTH = 101):
        '''
        Returns a square 2D Gaussian filter kernel normalized over a given 
        grid (the filter domain). This grid will contain a 2sigma Gaussian 
        where 2sigma is the distance from the central peak to the midpoint 
        of all sides.
        EX.
                   | - - - WIDTH - - -|
                    __________________
                   |         |        |
                   |         2s       |
                   |         |        |
                   | - 2s -  C - 2s - |
                   |         |        |
                   |         2s       |
                   |_________|________|

        Where C = the central peak, and 2s = 2sigma

        Input:
        WIDTH [odd int]: The width of the square filter donain. This should be 
            an odd integer so that the filter kernel can be centered on a 
            grid point. If odd, the method will increase it by 1.

        Output:
        G [2D numpy array]: The normalized Gaussian filter kernel in a square 
            2D array.

        ''' 
        # Check if the width is even
        if WIDTH % 2 == 0:
            WIDTH += 1

        mean = int(WIDTH/2) # The central point 
        mu_x = mean
        mu_y = mean
        sigma_x = float(WIDTH)/4.
        sigma_y = float(WIDTH)/4.
        G = np.zeros((WIDTH, WIDTH))

        # Compute the Gaussian
        for j in range(WIDTH):
            for i in range(WIDTH):
                G[j][i] = np.exp(-0.5 * ((np.divide((i-mu_x)**2, sigma_x**2)) 
                            + ((np.divide((j-mu_y)**2, sigma_y**2)))))

        # Normalize the Gaussian over the filter domain
        G = np.multiply(np.divide(1.0, np.sum(G)), G)

        return G



    @staticmethod
    def find_index(DATA):
        '''
        This method trims a 2D array (image) of any nan or zero padding around 
        the edges. It returns the trimmed array and the indexed position of 
        the trimmed region relative to the full array.

        Input:
        DATA [2D numpy array]: The 2D array you want to trim. Note: if the data 
            field within the array has many zero values, this method will not 
            work properly.

        Output:
        data [2D numpy array]: The trimmed 2D numpy array.
        y_max [int]: Index of where the array was trimmed (vertical, bottom).
        y_min [int]: Index of where the image was trimmed (vertical, top).
        x_max [int]: Index of where the image was trimmed (horizontal, right).
        x_max [int]: Index of where the image was trimmed (horizontal, left).

        '''

        # Find the edges of the data
        if DATA[0][0] == 0.0:
            # Find where the data is not 0.0
            index = np.where(DATA != 0.0)
        else:
            # Find where the data is not nan
            index = np.where(np.isnan(DATA) == False)

        # Find the boundaries
        y_max = np.max(index[0])
        y_min = np.min(index[0])
        x_max = np.max(index[1])
        x_min = np.min(index[1])

        # Trim the input
        data = DATA[y_min:y_max+1, x_min:x_max+1].copy()

        return data, y_max, y_min, x_max, x_min



    @staticmethod
    def MSE(CONTROL, EST, IMG_SET = False):
        '''
        The method calculates the mean square error (MSE) between a filtered 
        field and a control field (another filtered field). Any nan padding is 
        removed from the fields before the MES is computed so care must be taken 
        to make sure the non-nan padded data is the same shape.

        Mean Square Error:

                MSE = 1/n sum([x_i - X_i]^2)

        Where n = the number of samples, 
              x_i = the control value, 
              X_i = the sample value

        Input:
        CONTROL [2D numpy array]: A 2D filtered data field given by one of the 
            convolution methods within this class.
        EST [2D numpy array]: Another 2D filtered data field with the same 
            data (non nan) shape as the CONTROL field.
        IMG_SET [bool]: Set to True if both CONTROL and EST are lists of 2D 
            arrays. Set to False if CONTROL and EST are single 2D data arrays.

        Output:
        mse [float]: The mean square error between the two data fields.
        '''

        if IMG_SET == False:
            # Trim the padding from the fields
            cont, y_max, y_min, x_max, x_min = convolution.find_index(CONTROL)
            est, y_max, y_min, x_max, x_min = convolution.find_index(EST)
            
            # Number of elements
            n = cont.size
            
            # Find the squared difference
            diff = (cont - est)**2

        else:
            # Loop over the list of data fields
            diff = []
            for i, img in enumerate(CONTROL):
                cont, y_max, y_min, x_max, x_min = convolution.find_index(img)
                est, y_max, y_min, x_max, x_min = convolution.find_index(EST[i])

                # Find the squared difference
                diff.append((cont - est)**2)

            # Number of elements
            n = len(diff) * (diff[0].size)

        # Calculate the MSE
        mse = np.sum(diff) / n

        return mse

