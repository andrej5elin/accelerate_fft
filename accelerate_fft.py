"""
Wrapper of apple's accelerate (vDSP) FFT routines. It implements multiple-signal
version of ffts. 

Note that vDSP FFT only works for input data of length that is a power of 2,
e.g. ... 256, 512, 1024 ... Input data has to be contiguous. Transformation
is performed over the last axis (1d fft) and last two axes (2d fft).

Also note that real transform in accelerate has a wierd format, and is not 
the same as numpy's rfft and rfft2. Use :func:`unpack` or :func:`unpack2` to 
convert to numpy-like format.

This module has a very primitive multi-threading support. Useful only for 
large-size multi-signal 2d FFT. Input data has to be multi-dimensional, and it 
must have a size that is a multiple of nthread  * nfft, where
nthread is number of threads used, and nfft is the size of the fft. 
"""

__version__ = "0.2.0"

from _accelerate_fft_cffi import ffi, lib
import numpy as np
#from multiprocessing.pool import ThreadPool as Pool
import threading
#from multiprocessing import Pool
from functools import reduce
#import pyfftw
#empty = pyfftw.empty_aligned
empty = np.empty
fft_config = {"nthreads" : 1} 

lock = threading.Lock()

#--------------------------
# Higl-level user functions
#--------------------------

def test():
    """Runs unittests"""
    from unittest import main
    main(module='accelerate_fft_test', exit=False)

def set_nthreads(nthreads):
    """Sets number of threads used in calculations. Setting  nthreads > 1
    will trigger a ThreadPool and use multiple threads when computing.
    
    Note that this only works for multi-dimensional data (with three or more dimensions).
    Useful for multi-signal 2d ffts.
    """
    current_value = fft_config["nthreads"]
    fft_config["nthreads"] = max(1,int(nthreads))
    #we will have to rebuild the setups, so clear it.
    FFTSETUP_DATA.clear()
    return current_value

def fft(a, n = None, axis = -1, norm = None, overwrite_x = False, split_in = False, split_out = False):
    """Returns a dicrete Fourier transform.
    
    Parameters
    ----------
    a : ndarray or (ndarray,ndarray)
        Input complex array (if split_in = False) or a tuple of float arrays
        (if split_in = True). 
    axis : int, optional
        Axis over which to compute the FFT.  If not given, the last axis is
        used.
    overwrite_x  : bool
        Specifies whether input data can be overwritten. Whether data is actually
        overwritten depends on the `split_in` and `split_out` arguments.
    split_in : bool
        If set, input data is treated as split-complex data. A tuple of two
        real arrays describing the real and imginary parts of the data.
    split_out : bool
        If set, output data is treated as split-complex data. A tuple of two
        real arrays describing the real and imginary parts of the data.
        
    Returns
    -------
    out : ndarray or (ndarray, ndarray)
        Output complex array (if split_out = False) or a tuple of float arrays
        (if split_out = True). 
    """
    return generalized_fft(a, n = n, axis = axis, norm = norm, overwrite_x = overwrite_x, split_in = split_in, split_out = split_out, direction = +1)

def ifft(a, n = None, axis = -1, norm = None, overwrite_x = False, split_in = False, split_out = False):
    """Returns an inverse of the dicrete Fourier transform.
    
    Note that a == ifft(fft(a))/a.shape[-1]
    
    Parameters
    ----------
    a : ndarray or (ndarray,ndarray)
        Input complex array (if split_in = False) or a tuple of float arrays
        (if split_in = True). 
    axis : int, optional
        Axis over which to compute the FFT.  If not given, the last axis is
        used.
    overwrite_x  : bool
        Specifies whether input data can be overwritten. Whether data is actually
        overwritten depends on the `split_in` and `split_out` arguments.
    split_in : bool
        If set, input data is treated as split-complex data. A tuple of two
        real arrays describing the real and imginary parts of the data.
    split_out : bool
        If set, output data is treated as split-complex data. A tuple of two
        real arrays describing the real and imginary parts of the data.
        
    Returns
    -------
    out : ndarray or (ndarray, ndarray)
        Output complex array (if split_out = False) or a tuple of float arrays
        (if split_out = True). 
    """
    return generalized_fft(a, n = n, axis = axis, norm = norm, overwrite_x = overwrite_x, split_in = split_in, split_out = split_out, direction = -1)
    
def fft2(a, s = None, axes = (-2,-1), norm = None, overwrite_x = False, split_in = False, split_out = False):
    """Returns a dicrete 2D Fourier transform.
        
    Parameters
    ----------
    a : ndarray or (ndarray,ndarray)
        Input complex array (if split_in = False) or a tuple of float arrays
        (if split_in = True). 
    axes : sequence of ints, optional
        Axes over which to compute the FFT.  If not given, the last two
        axes are used. Only length 2 axes is supported.
    overwrite_x  : bool
        Specifies whether input data can be overwritten. Whether data is actually
        overwritten depends on the `split_in` and `split_out` arguments.
    split_in : bool
        If set, input data is treated as split-complex data. A tuple of two
        real arrays describing the real and imginary parts of the data.
    split_out : bool
        If set, output data is treated as split-complex data. A tuple of two
        real arrays describing the real and imginary parts of the data.
        
    Returns
    -------
    out : ndarray or (ndarray, ndarray)
        Output complex array (if split_out = False) or a tuple of float arrays
        (if split_out = True). 
    """
    return generalized_fft(a, s = s, axes = axes, norm = norm, overwrite_x = overwrite_x, split_in = split_in, split_out = split_out, direction = +1, dim = 2)

def ifft2(a, s = None, axes = (-2,-1), norm = None, overwrite_x = False, split_in = False, split_out = False):
    """Returns an inverse of the dicrete 2D Fourier transform.
    
    Note that a == ifft2(fft2(a))/a.shape[-1]/a.shape[-2]
    
    Parameters
    ----------
    a : ndarray or (ndarray,ndarray)
        Input complex array (if split_in = False) or a tuple of float arrays
        (if split_in = True). 
    axes : sequence of ints, optional
        Axes over which to compute the FFT.  If not given, the last two
        axes are used. Only length 2 axes is supported.
    overwrite_x  : bool
        Specifies whether input data can be overwritten. Whether data is actually
        overwritten depends on the `split_in` and `split_out` arguments.
    split_in : bool
        If set, input data is treated as split-complex data. A tuple of two
        real arrays describing the real and imginary parts of the data.
    split_out : bool
        If set, output data is treated as split-complex data. A tuple of two
        real arrays describing the real and imginary parts of the data.
        
    Returns
    -------
    out : ndarray or (ndarray, ndarray)
        Output complex array (if split_out = False) or a tuple of float arrays
        (if split_out = True). 
    """
    return generalized_fft(a, s = s, axes = axes, norm = norm, overwrite_x = overwrite_x, split_in = split_in, split_out = split_out, direction = -1, dim = 2)
  
def rfft(a, n = None, axis = -1, norm = None, split_out = False):
    """Returns a dicrete Fourier transform for real input data.
    
    Note that you should call :func:`unpack`on the computed result to obtain
    a numpy-like representation of the computed data.
     
    Parameters
    ----------
    a : ndarray
        Input real array. 
    axis : int, optional
        Axis over which to compute the FFT.  If not given, the last axis is
        used.
    split_out : bool
        If set, output data is treated as split-complex data. A tuple of two
        real arrays describing the real and imginary parts of the data.
        
    Returns
    -------
    out : ndarray or (ndarray, ndarray)
        Output complex array (if split_out = False) or a tuple of float arrays
        (if split_out = True). 
    """
    return generalized_fft(a, n = n, axis = axis, norm = norm, split_out = split_out, direction = +1, dim = 1, real_transform = True)

def irfft(a, n = None, axis = -1, norm = None, split_in =  False):
    """Returns an inverse of the dicrete Fourier transform for real input data.
    
    Note that a == irfft(rfft(a))/a.shape[-1]/2 
    
    Parameters
    ----------
    a : ndarray or (ndarray,ndarray)
        Input complex array (if split_in = False) or a tuple of float arrays
        (if split_in = True). 
    axis : int, optional
        Axis over which to compute the FFT.  If not given, the last axis is
        used.
    split_in : bool
        If set, input data is treated as split-complex data. A tuple of two
        real arrays describing the real and imginary parts of the data.

    Returns
    -------
    out : ndarray 
        Output float array.
    """
    return generalized_fft(a, n = n, axis = axis, norm = norm,  split_in = split_in, direction = -1, dim = 1, real_transform = True)

def rfft2(a, s = None, axes = (-2,-1), norm = None, split_out = False):
    """Returns a dicrete Fourier transform for 2D real input data.
    
    Note that you should call :func:`unpack2`on the computed result to obtain
    a numpy-like representation of the computed data.
     
    Parameters
    ----------
    a : ndarray
        Input real array. 
    axes : sequence of ints, optional
        Axes over which to compute the FFT.  If not given, the last two
        axes are used. Only length 2 axes is supported.
    split_out : bool
        If set, output data is treated as split-complex data. A tuple of two
        real arrays describing the real and imginary parts of the data.
        
    Returns
    -------
    out : ndarray or (ndarray, ndarray)
        Output complex array (if split_out = False) or a tuple of float arrays
        (if split_out = True). 
    """
    return generalized_fft(a, s = s, axes = axes, norm = norm, split_out = split_out, direction = +1, dim = 2, real_transform = True)

def irfft2(a, s = None, axes = (-2,-1), norm = None,split_in = False):
    """Returns an inverse of the dicrete Fourier transform for real 2D input data.
    
    Note that a == irfft2(rfft2(a))/a.shape[-1]/a.shape[-2]/2 
    
    Parameters
    ----------
    a : ndarray or (ndarray,ndarray)
        Input complex array (if split_in = False) or a tuple of float arrays
        (if split_in = True). 
    axes : sequence of ints, optional
        Axes over which to compute the FFT.  If not given, the last two
        axes are used. Only length 2 axes is supported.
    split_in : bool
        If set, input data is treated as split-complex data. A tuple of two
        real arrays describing the real and imginary parts of the data.

    Returns
    -------
    out : ndarray 
        Output float array.
    """
    return generalized_fft(a, s = s, axes = axes, norm = norm, split_in = split_in, direction = -1, dim = 2, real_transform = True)

def prepare_input_1d(a,axis = -1,n = None,norm = None):
    if n is not None:
        raise ValueError("The 'n' argument is not yet supported!")
    if norm is not None:
        raise ValueError("The 'norm' argument is not yet supported!")
    a = np.asarray(a)
    return a
    return np.swapaxes(a,-1,-1)

def prepare_input_2d(a,axes = (-2,-1),s = None,norm = None):
    return a
    if s is not None:
        raise ValueError("The 's' argument is not yet supported!")
    if norm is not None:
        raise ValueError("The 'norm' argument is not yet supported!")
    a = np.asarray(a)
    try:
        first, last = [int(i) for i in axes]    
    except:
        raise ValueError("Invalid axes argument. Must be a tuple of of length 2")
    return np.moveaxis(a,(first,last), (-2,-1))


def unpack(a, axis = -1, inplace = False):
    """Unpack vDSP rfft data format of shape (...,n/2) into regular
    numpy-like rfft of shape (...,n/2 + 1).
    
    Optionally, this can be done inplace without calculating the last element.
    """
    a = np.swapaxes(np.asarray(a),axis,-1)
    
    if inplace == False:
        out = np.empty(a.shape[:-1] + (a.shape[-1]+1,), dtype = a.dtype)
    else:
        out = a
    if inplace == False:
        out[...,-1] = a[...,0].imag
        out[...,1:-1] = a[...,1:]
    out[...,0] = a[...,0].real
    
    return np.swapaxes(out,-1,axis)
    
def unpack2(a, axes = (-2,-1),inplace = False):
    """Unpack vDSP rfft2 data format of shape (..., n, n/2) into regular
    numpy-like rfft2 of shape (..., n, n/2 + 1).
    
    Optionally, this can be done inplace without calculating the last column.
    """
    a = np.moveaxis(np.asarray(a),axes,(-2,-1))
    
    n0,n1 = a.shape[-2:] #number of rows
    n0_half = n0//2

    if inplace == False:
        out = np.empty(a.shape[:-1] + (a.shape[-1]+1,), dtype = a.dtype)
    else:
        out = a
    if inplace == True:
        #we must copy first column, because we will write to it.
        column = a[...,:,0].copy()
    else:
        column = a[...,:,0]
    
    #unpack first column
    first_column = out[...,:,0]
    first_column[...,0] = column[...,0].real
    first_column[...,n0_half] = column[...,1].real
    first_column[...,1:n0_half] = (column[...,2::2].real + 1j* column[...,3::2].real)
    first_column[...,n0_half+1:] = first_column[...,n0_half-1:0:-1].conj()
    
    if inplace == False:
        #unpack last column
        last_column = out[...,:,n1]
        last_column[...,0] = column[...,0].imag
        last_column[...,n0_half] = column[...,1].imag
        last_column[...,1:n0_half] = column[...,2::2].imag + 1j* column[...,3::2].imag
        last_column[...,n0_half+1:] = last_column[...,n0_half-1:0:-1].conj()
        
        #unpack rest of the data
        out[...,:,1:n1] = a[...,:,1:]
        
    return np.moveaxis(out,(-2,-1),axes)

def create_fftsetup(logn, double = True):
    """Creates and returns a fftsetup of a given logsize and precision.
    Setup is allocated only if needed. This function must be called whenever
    a setup is required.
    
    Parameters
    ----------
    logn : int
        A np.log2 of the size of the arrays that you will be using. E.g. for
        an array of size 1024 use logn = 10.
    double : bool
        Indicates whether we are working with double precision (default) or not.
    """
    #this function must be thread-safe, so we acquire a lock here
    with lock:
        if double == True:
            return fftsetupD.create(logn)
        else:
            return fftsetup.create(logn)

def destroy_fftsetup():
    """Destroys all (double and single precision) setups."""
    with lock:
        FFTSETUP_DATA.clear()
        fftsetup.destroy() 
        fftsetupD.destroy() 
    
#-------------------------------
# Non-user functions and objects
#-------------------------------

class FFTSetupPointer:
    """fftsetup object. This is meant to be initiated only once. 
    Use :func:`create_fftsetup` to build setups."""

    logsize = 0
    setups = []
    
    def create(self,logsize):
        radix = 2
        if logsize > self.logsize:
            self.logsize = logsize
            self.pointer = lib.vDSP_create_fftsetup(logsize, radix)
            self.setups.append(self.pointer)
        elif logsize <= 0:
            raise ValueError("Invalid logsize")
        return self.pointer
        
    def destroy(self):
        for pointer in self.setups:
            lib.vDSP_destroy_fftsetup(pointer)
        if self.setups != []:
            del self.pointer
            self.setups.clear()
        self.logsize = 0

    def __del__(self):
        self.destroy()

class FFTSetupDPointer:
    """fftsetupD object. This is meant to be initiated only once.
    Use :func:`create_fftsetup` to build setups.""" 
    
    logsize = 0
    setups = []
    
    def create(self,logsize):
        radix = 2
        if logsize > self.logsize:
            self.logsize = logsize
            self.pointer = lib.vDSP_create_fftsetupD(logsize, radix)
            self.setups.append(self.pointer)
        elif logsize <= 0:
            raise ValueError("Invalid logsize")
        return self.pointer
        
    
    def destroy(self):
        for pointer in self.setups:
            lib.vDSP_destroy_fftsetupD(pointer)
        if self.setups != []:
            del self.pointer
            self.setups.clear()
        self.logsize = 0
        
    def __del__(self):
        self.destroy()
        
#:fftsetup singletons, we have at most two setup objects at runtime, a single and double precision.       
fftsetup = FFTSetupPointer() 
fftsetupD = FFTSetupDPointer()        
                      
#: a placehold for FFT setups and temporary data. This gets populated when calling fft functions
FFTSETUP_DATA = {}


class FFTSetupData():
    """FFTSetup data and info about the transform is stored here.
    """
    
    def __init__(self, in_shape, out_shape, split_in = False, split_out = False, dim = 1, double = True, direction = +1, real_transform = False):
        # check if shapes are OK
        for shape in (in_shape[-dim:], out_shape[-dim:]):
            for count in shape:
                size = int(np.log2(count))
                if count != 2**size:
                    raise ValueError("Size of the input/output arrays must be a power of 2 for each of the dimensions.")
        
        #for real transforms input and output arrays differ in size, the highest value is the one we need.
        shape_high = tuple((max(a,b) for a,b in zip(in_shape, out_shape)))
        
        # how many elements in data
        self.count = reduce((lambda x,y: x*y),shape_high)
        if real_transform == True:
            self.count //=2
        
        # array size in log size parameters. 
        self.size = tuple(reversed([int(np.log2(count)) for count in shape_high]))
        
        self.in_shape = in_shape
        self.out_shape = out_shape
        
        self.double = double

        if self.double:
            self.cast_name = "DSPDoubleComplex*"
            #self.buffer_real = empty((2**max(self.size),),"float64")
            #self.buffer_imag = empty((2**max(self.size),),"float64")
        else:
            self.cast_name = "DSPComplex*"
            #self.buffer_real = empty((2**max(self.size),),"float32")
            #self.buffer_imag = empty((2**max(self.size),),"float32")
                 
        #buffer for temporary data
        #self.buffer_pointer = _create_split_complex_pointer(self.buffer_real, self.buffer_imag, double = self.double)
            
    def allocate_memory(self):
        
        if self.double:
            self.array_real = empty(self.out_shape,"float64")
            self.array_imag = empty(self.out_shape,"float64")
        else:
            self.array_real = empty(self.out_shape,"float32")
            self.array_imag = empty(self.out_shape,"float32")            
                    
        self._split_complex_pointer = _create_split_complex_pointer(self.array_real, self.array_imag, double = self.double)
    
    @property            
    def split_complex_pointer(self):
        """Returns a valid splitcomplex pointer for output data."""
        try:
            return self._split_complex_pointer
        except AttributeError:
            self.allocate_memory()
            return self._split_complex_pointer
        
    @property
    def pointer(self):
        return create_fftsetup(max(self.size), double = self.double)

def _optimal_workers(size, nthreads):
    """determines optimal number of workers for ThreadPool."""
    if size%nthreads == 0:
        return nthreads
    else:
        return _optimal_workers(size, nthreads-1)

def _optimal_flattened_shape(original_shape, dim = 1):
    """Finds a shape for a flattened array, so that we can then iterate over it"""
    nthreads = fft_config["nthreads"]
    if len(original_shape) == 1:
        #Even for 1D transforms data has to be 3D data. 
        # algorithm iterates over the first axis, transform is done on 2D data.
        return (1,1) + original_shape
    
    elif len(original_shape) == 2:
        if dim == 1:
            # we need to reduce the shape
            newshape = reduce((lambda x,y: x*y), original_shape[:-1])
            n = _optimal_workers(newshape,nthreads)
            return (n,newshape//n) + original_shape[:-1]
        else:
            return (1,) + original_shape
    else:
        newshape = reduce((lambda x,y: x*y), original_shape[:-2])
        return (newshape,) + original_shape[-2:]

      
def _create_split_complex_pointer(array_real, array_imag, double = False):
    """Creates a split-complex data pointer from two real numpy arrays"""
    _array_real = ffi.from_buffer(array_real)
    _array_imag = ffi.from_buffer(array_imag)
         
    if double:
        split_complex_pointer = ffi.new("DSPDoubleSplitComplex*") 
        split_complex_pointer.realp = ffi.cast("double*",_array_real)
        split_complex_pointer.imagp = ffi.cast("double*",_array_imag)
    else:
        split_complex_pointer = ffi.new("DSPSplitComplex*") 
        split_complex_pointer.realp = ffi.cast("float*",_array_real)
        split_complex_pointer.imagp = ffi.cast("float*",_array_imag)
    return split_complex_pointer

        
def _get_in_out_dtype(in_dtype, split_in = False, split_out = False, direction = +1, real_transform = False):
    """Determines input and output arrays dtype from the calculation parameters"""
    
    double = in_dtype in ("float64",  "complex128")
    f,c = (np.dtype("float64"), np.dtype("complex128")) if double else (np.dtype("float32"), np.dtype("complex64"))
    if real_transform == True:
        if direction == +1:
            return (f,f) if split_out else (f,c) 
        else:
            return (f,f) if split_in else (c,f)
    else:
        return f if split_in else c , f if split_out else c
    
def _get_out_shape(in_shape, dim = 1, direction = +1, real_transform = False):
    """Determines output array shape from the calculation parameters"""
    if real_transform:
        if direction == +1:
            return in_shape[:-1] + (in_shape[-1]//2,)
        else:
            return in_shape[:-1] + (in_shape[-1] * 2,)
    else:
        return in_shape
     
def _init_setup_and_arrays(a, n = None, s = None, axis = -1, axes = (-2,-1), norm = None, overwrite_x = False, dim = 1, split_in = False, split_out = False, direction = +1, real_transform = False):
    """checks input parameters and creates a valid fft setup and input/output arrays"""
    
    if n is not None:
        raise ValueError("The 'n' argument is not yet supported!")
    if s is not None:
        raise ValueError("The 's' argument is not yet supported!")
    if norm is not None:
        raise ValueError("The 'norm' argument is not yet supported!")
    
    # make an array and read dtype, and shape
    if split_in:
        a0,a1 = np.asarray(a[0]),np.asarray(a[1])
        if dim == 1:
            a0 = np.swapaxes(a0,axis,-1)
            a1 = np.swapaxes(a1,axis,-1)
        else:
            a0 = np.moveaxis(a0,axes,(-2,-1))
            a1 = np.moveaxis(a1,axes,(-2,-1))            
        dtype = a0.dtype
        shape = a0.shape 
        new_shape = _optimal_flattened_shape(a0.shape, dim = dim)
        a0, a1 = (a0.reshape(new_shape), a1.reshape(new_shape))
        if not a0.data.c_contiguous:
            a0 = a0.copy()
        if not a1.data.c_contiguous:
            a1 = a1.copy()
        a =  (a0, a1)
    else:
        a = np.asarray(a)
        if dim == 1:
            a = np.swapaxes(a,axis,-1)
        else:
            a = np.moveaxis(a,axes,(-2,-1))  

        dtype = a.dtype
        shape = a.shape 
        new_shape = _optimal_flattened_shape(a.shape, dim = dim)
        a = a.reshape(new_shape)  
        if not a.data.c_contiguous:
            a = a.copy()


    # determine the allowed type and shapes of the input and output arrays
    in_dtype, out_dtype = _get_in_out_dtype(dtype, split_in, split_out, direction, real_transform)
    out_shape = _get_out_shape(new_shape, dim, direction, real_transform)

    # make sure it is right type
    a = tuple((np.asarray(d,in_dtype) for d in a)) if split_in else np.asarray(a,in_dtype)  
    #create output array(s)
    if overwrite_x == True:
        if split_in == split_out and real_transform == False:
            out = a
        else:
            out = (np.empty(out_shape,out_dtype), np.empty(out_shape,out_dtype)) if split_out else np.empty(out_shape,out_dtype) 
    else:
        out = (np.empty(out_shape,out_dtype), np.empty(out_shape,out_dtype)) if split_out else np.empty(out_shape,out_dtype) 
    #determine if we need to use double precision    
    double = dtype in ("float64","complex128") 
    key = (new_shape[-2:], out_shape[-2:],split_in, split_out, dim, double,direction,real_transform)
    try:
        #if setup already exists, just take one.
        setup = FFTSETUP_DATA[key] 
    except KeyError:
        #else create one. For each thread, there must be a new setup, so we buld a list of setups here.
        setup = [FFTSetupData(*key) for i in range(fft_config["nthreads"])]    
        FFTSETUP_DATA[key] = setup
    
    return shape, setup, a, out

#-----------------
# Worker functions
#-----------------


POOL = {}

from threading import Thread
from queue import Queue

class PoolWorker():
    """Mimics the object returned by ThreadPool.apply_async method."""
    def __init__(self, queue):
        self.queue = queue
        
    def get(self):
        return self.queue.get()

class Pool:
    """A multiprocessing.pool.ThreadPool -like object.
    
    Implements only necessary part of ThreadPool API.
    """
    def __init__(self,nthreads):
        def worker(i, inputs, results):
            #print("Thread ", i, "started")
            while True:
                data = inputs.get()
                if data is None:
                    results.put(None)
                    break
                else:
                    func, args, kwargs = data
                    out = func(*args,**kwargs)
                    results.put((i,out))
            #print("Thread ", i, "stopped")
            
            
        self.nthreads = nthreads
        self.results = [Queue() for i in range(nthreads)]
        self.inputs = [Queue() for i in range(nthreads)]
        self.threads = [Thread(target = worker, args = (i,self.inputs[i],self.results[i]), daemon = True) for i in range(nthreads)]
        [t.start() for t in self.threads]
        self.nruns = 0
        
    def apply_async(self,func, args = (), kwds = {}):
        index = self.nruns % self.nthreads
        self.inputs[index].put((func,args,kwds))
        self.nruns += 1
        return PoolWorker(self.results[index])
              
    def close(self):
        for q in self.inputs:
            q.put(None)
        for q in self.results:
            while q.get(timeout = 1) != None:
                pass
        for t in self.threads:
            t.join()
                 
    def __del__(self):
        self.close()

def clear_pool():
    """Clears thread pool. Deletes all Pool objects, which terminates all
    running background threads."""
    
    POOL.clear()

def _sequential_call(fftfunc,setup, *args,**kwargs):
    # a simple sequential runner 
    [fftfunc(setup,*arg,**kwargs) for arg in zip(*args)] 
   
def _calculate_fft(fftfunc,*args,**kwds):
    """Runs fft function with given arguments (optionally in parallel 
    using a ThreadPool)"""
    nthreads = fft_config["nthreads"]

    if nthreads > 1:
        try:
            pool = POOL[nthreads]
        except KeyError:
            pool = Pool(nthreads)
            POOL[nthreads] = pool
        #pool = Pool(nthreads)
        setup = args[0]
        args = args[1:]
        workers = [pool.apply_async(fftfunc, args = (setup[i%nthreads],) + arg, kwds = kwds) for i,arg in enumerate(zip(*args))] 
        _ = [w.get() for w in workers]
        #pool.close()
    else:
        setup = args[0]
        #setup is a list. For single-threaded run, take the first and only element of the list.
        args = (setup[0],) + args[1:]
        _sequential_call(fftfunc,*args, **kwds)

def _get_vDSP_fft_inplace(double = False, dim = 1, real_transform = False):
    if double:
        if dim == 1:
            _func = lib.vDSP_fftm_zripD if real_transform else lib.vDSP_fftm_zipD    
        else:
            _func = lib.vDSP_fft2d_zripD if real_transform else lib.vDSP_fft2d_zipD  
    else:
        if dim == 1:
            _func = lib.vDSP_fftm_zrip if real_transform else lib.vDSP_fftm_zip  
        else:
            _func = lib.vDSP_fft2d_zrip if real_transform else lib.vDSP_fft2d_zip  
    return _func

def _get_vDSP_fft_outplace(double = False, dim = 1,real_transform = False):
    if double:
        if real_transform == False:
            _func = lib.vDSP_fftm_zopD if dim == 1 else lib.vDSP_fft2d_zopD  
        else:
            _func = lib.vDSP_fftm_zropD if dim == 1 else lib.vDSP_fft2d_zropD 
    else:
        if real_transform == False:
            _func = lib.vDSP_fftm_zop  if dim == 1 else lib.vDSP_fft2d_zop 
        else:
            _func = lib.vDSP_fftm_zrop  if dim == 1 else lib.vDSP_fft2d_zrop 
    return _func
        
def _fft(setup, a, out, n = 1, dim = 1, direction = +1, real_transform = False):
    """1D or 2D fft transform, real or complex"""

    #initialize
    _a = ffi.from_buffer(a)
    _pa = ffi.cast(setup.cast_name,_a) 
    _ps = setup.split_complex_pointer 
    _out = ffi.from_buffer(out) 
    _pout = ffi.cast(setup.cast_name,_out) 
    
    #define functions
    _ctoz, _ztoc = (lib.vDSP_ctozD, lib.vDSP_ztocD) if setup.double else (lib.vDSP_ctoz, lib.vDSP_ztoc)
    _func = _get_vDSP_fft_inplace(double = setup.double, dim = dim, real_transform = real_transform)
   
    #perform calculations
    _ctoz(_pa,2,_ps,1, setup.count)
  
    if dim == 1:
        _func(setup.pointer, _ps, 1, setup.count//n, setup.size[0], n , direction)
    else:
        _func(setup.pointer, _ps, 1, 0, setup.size[0], setup.size[1], direction)
    _ztoc(_ps,1,_pout,2, setup.count)
    
def _ffts(setup, a, outr, outi, n = 1, dim = 1, direction = +1, real_transform = False):
    """Same as _fft, but it leaves the data in split-complex format"""
 
    #initialize
    _a = ffi.from_buffer(a) 
    _pa = ffi.cast(setup.cast_name,_a)
    _ps = _create_split_complex_pointer(outr, outi, double = setup.double)
    
    #define functions
    _ctoz= lib.vDSP_ctozD if setup.double else lib.vDSP_ctoz
    _func = _get_vDSP_fft_inplace(double = setup.double, dim = dim, real_transform = real_transform)

    #perform calculations
    _ctoz(_pa,2,_ps,1, setup.count)
    if dim == 1:
        _func(setup.pointer, _ps, 1, setup.count//n, setup.size[0], n, direction)
    else:
        _func(setup.pointer, _ps, 1,0, setup.size[0], setup.size[1],direction)

def _sffti(setup, ar, ai, out, n = 1, dim = 1, direction = +1, real_transform = False):
    """1D or 2D fft transform, real or complex for split-complex input data - inplace"""

    #initialize
    _ps = _create_split_complex_pointer(ar, ai, double = setup.double)
    _out = ffi.from_buffer(out) 
    _pout = ffi.cast(setup.cast_name,_out) 
    
    #define functions
    _ztoc = lib.vDSP_ztocD if setup.double else lib.vDSP_ztoc
    _func = _get_vDSP_fft_inplace(double = setup.double, dim = dim, real_transform = real_transform)
   
    #perform calculations  
    if dim == 1:
        _func(setup.pointer, _ps, 1, setup.count//n, setup.size[0], n , direction)
    else:
        _func(setup.pointer, _ps, 1, 0, setup.size[0], setup.size[1], direction)
    _ztoc(_ps,1,_pout,2, setup.count)        

def _sffto(setup, ar, ai, out, n = 1, dim = 1, direction = +1, real_transform = False):
    """1D or 2D fft transform, real or complex for split-complex input data - out of place"""

    _ps = _create_split_complex_pointer(ar, ai, double = setup.double)
    _out = ffi.from_buffer(out) 
    _pout = ffi.cast(setup.cast_name,_out)   
    _pouts  = setup.split_complex_pointer 
    _func = _get_vDSP_fft_outplace(double = setup.double, dim = dim, real_transform = real_transform)
    if dim == 1:
        _func(setup.pointer, _ps, 1, setup.count//n,_pouts, 1, setup.count//n, setup.size[0], n, direction)
    else:
        _func(setup.pointer, _ps, 1,0, _pouts, 1,0,setup.size[0], setup.size[1],direction)

    _ztoc = lib.vDSP_ztocD if setup.double else lib.vDSP_ztoc
    _ztoc(_pouts,1,_pout,2, setup.count)     

def _sfftis(setup, ar,ai, n = 1, dim = 1, direction = +1):
    """Same as _ffts (complex transform only), but with input data as split complex data. Inplace transform"""

    _ps = _create_split_complex_pointer(ar, ai, double = setup.double)
    _func = _get_vDSP_fft_inplace(double = setup.double, dim = dim)
    if dim == 1:
        _func(setup.pointer, _ps, 1, setup.count//n, setup.size[0], n, direction)
    else:
        _func(setup.pointer, _ps, 1, 0, setup.size[0], setup.size[1],direction)

def _sfftos(setup, ar,ai, outr, outi, n = 1, dim = 1, direction = +1):
    """out-of-place version of _sfftis"""

    _pins = _create_split_complex_pointer(ar, ai, double = setup.double)
    _pouts = _create_split_complex_pointer(outr, outi, double = setup.double)
    _func = _get_vDSP_fft_outplace(double = setup.double, dim = dim)
    if dim == 1:
        _func(setup.pointer, _pins, 1, setup.count//n,_pouts, 1, setup.count//n, setup.size[0], n, direction)
    else:
        _func(setup.pointer, _pins, 1,0, _pouts, 1,0,setup.size[0], setup.size[1],direction)

def generalized_fft(a,n = None, s = None, axis = -1, axes = (-2,-1), norm = None, dim = 1, overwrite_x = False, split_in = False, split_out = False, direction = +1, real_transform = False):
    original_shape, setup, a, out = _init_setup_and_arrays(a, n = n, s = s, axis = axis, axes = axes, overwrite_x = overwrite_x, dim = dim, split_in = split_in, split_out = split_out, direction = direction, real_transform = real_transform)
    
    if split_in == True:
        #a must be a tuple of length twp for the two arrays
        a_real, a_imag = a

        n = a_real.shape[-2]
       
        if split_out == True:
            if overwrite_x == False:
                #out of place transform
                out_real, out_imag = out
                _calculate_fft(_sfftos , setup, a_real,a_imag,out_real,out_imag, n = n, dim = dim, direction = direction)

            else:
                #inplace transform
                _calculate_fft(_sfftis, setup, a_real,a_imag, n = n, dim = dim, direction = direction)
        else:
            if overwrite_x == True:
                _calculate_fft(_sffti, setup, a_real,a_imag,out, n = n, dim = dim, direction = direction, real_transform = real_transform)
            else:
                _calculate_fft(_sffto, setup, a_real,a_imag,out, n = n, dim = dim, direction = direction, real_transform = real_transform)

    else:

        n = a.shape[-2]
                  
        if split_out == False:
            _calculate_fft(_fft, setup, a,out, n = n, dim = dim, direction = direction, real_transform = real_transform)
        else:
            out_real, out_imag =  out
            _calculate_fft(_ffts, setup, a,out_real,out_imag, n = n, dim = dim, direction = direction, real_transform = real_transform)
    if split_out:
        out_shape = original_shape[:-1] + out[0].shape[-1:]
        x,y =  out[0].reshape(out_shape), out[1].reshape(out_shape)
        if dim == 2:
            return np.moveaxis(x,(-2,-1),axes),np.moveaxis(y,(-2,-1),axes)
        else:
            return np.swapaxes(x,-1,axis),np.swapaxes(y,-1,axis)
        
    else:
        out_shape = original_shape[:-1] + out.shape[-1:]
        z = out.reshape(out_shape)  
        if dim == 2:
            return np.moveaxis(z,(-2,-1),axes)
        else:
            return np.swapaxes(z,-1,axis)      





    
