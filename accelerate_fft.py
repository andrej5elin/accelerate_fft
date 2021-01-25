__version__ = "0.1.0"

from _accelerate_fft_cffi import ffi, lib
import numpy as np
from multiprocessing.pool import ThreadPool as Pool
#from multiprocessing import Pool
from functools import reduce
#import pyfftw
#empty = pyfftw.empty_aligned
empty = np.empty

fft_config = {"nthreads" : 1} 

def set_nthreads(nthreads):
    """Sets number of threads used in calculations. Setting this to nthreads > 1
    will trigger a ThreadPool and use multiple threads when computing.
    """
    current_value = fft_config["nthreads"]
    fft_config["nthreads"] = max(1,int(nthreads))
    #we will have to rebuild the setups, so clear it.
    FFTSETUP_DATA.clear()
    return current_value

class FFTSetupPointer:
    """fftsetup object. This is meant to be initiated only once. 
    Use :func:`create_fftsetup` to build setups."""

    logsize = 0
    
    def create(self,logsize):
        self.destroy()
        radix = 2
        self.logsize = logsize
        self.pointer = lib.vDSP_create_fftsetup(logsize, radix)
    def destroy(self):
        try:
            lib.vDSP_destroy_fftsetup(self.pointer)
            del self.pointer
            del self.logsize
        except AttributeError:
            pass

class FFTSetupDPointer:
    """fftsetupD object. This is meant to be initiated only once.
    Use :func:`create_fftsetup` to build setups.""" 
    
    logsize = 0
    
    def create(self,logsize):
        self.destroy()
        radix = 2
        self.logsize = logsize
        self.pointer = lib.vDSP_create_fftsetupD(logsize, radix)
    
    def destroy(self):
        try:
            lib.vDSP_destroy_fftsetupD(self.pointer)
            del self.pointer
            del self.logsize
        except AttributeError:
            pass
        
    def __del__(self):
        self.destroy()
        
#:fftsetup singletons, we have at most two setup objects at runtime, a single and double precision.       
_fftsetup = FFTSetupPointer() 
_fftsetupD = FFTSetupDPointer()        
        
def create_fftsetup(logn, double = True):
    """Creates and returns a fftsetup of a given logsize and precision."""
    if double == True:
        if _fftsetupD.logsize < logn:
            _fftsetupD.create(logn)
        return _fftsetupD.pointer
    else:
        if _fftsetup.logsize < logn:
            _fftsetup.create(logn)
        return _fftsetup.pointer  
    
def destroy_fftsetup():
    """Destroys all (double and single precision) setups."""
    FFTSETUP_DATA.clear()
    _fftsetup.destroy() 
    _fftsetupD.destroy() 
              
#: a placehold for FFT setups and temporary data. This gets populated when calling fft()
FFTSETUP_DATA = {}

def _optimal_workers(size, nthreads):
    """determines optimal number of workers for ThreadPool."""
    if size%nthreads == 0:
        return nthreads
    else:
        return _optimal_workers(size, nthreads-1)

def _optimal_flattened_shape(original_shape,  dim = 1):
    nthreads = fft_config["nthreads"]
    newshape = reduce((lambda x,y: x*y), original_shape[:-dim] or [1])
    if fft_config["nthreads"] > 1:
        n = _optimal_workers(newshape,nthreads)
        newshape = (n,newshape//n,) + original_shape[-dim:]
    else:
        newshape = (newshape,) + original_shape[-dim:]
    return newshape

def _sequential_call(fftfunc,setup, *args,**kwargs):
    [fftfunc(setup,*arg,**kwargs) for arg in zip(*args)] 

def _calculate_fft(fftfunc,*args,**kwds):
    nthreads = fft_config["nthreads"]

    if nthreads > 1:
        pool = Pool(nthreads)
        workers = [pool.apply_async(_sequential_call, args = (fftfunc,) + arg, kwds = kwds) for arg in zip(*args)] 
        results = [w.get() for w in workers]
        pool.close()
    else:
        setup = args[0]
        args = (setup[0],) + args[1:]
        _sequential_call(fftfunc,*args, **kwds)
      
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

class FFTSetupData():
    _split_complex_pointer = None
    
    def __init__(self, shape, dtype):
        
        max_size = 0
        
        self.size = []
        self.shape = shape
        
        for count in shape:
            size = int(np.log2(count))
            self.size.append(size)
            if count != 2**size:
                raise ValueError("Size of the array must be a power of 2.")
            max_size = max(max_size, size)
            
        self.double = not dtype in ("float32", "complex64") 

        #determine output array dtype 
        if dtype == "float32":
            self.dtype = np.dtype("complex64")
        elif dtype == "float64":
            self.dtype = np.dtype("complex128")
        elif dtype in ("complex64", "complex128"):
            self.dtype = dtype
        else:
            self.dtype = np.dtype(complex)

        if self.double:
            self.pointer = create_fftsetup(max_size, double = True)
            self.cast_name = "DSPDoubleComplex*"
        else:
            self.pointer = create_fftsetup(max_size, double = False)
            self.cast_name = "DSPComplex*"
            
    def allocate_memory(self):
        
        if self.double:
            self.array_real = empty(self.shape,"float64")
            self.array_imag = empty(self.shape,"float64")
        else:
            self.array_real = empty(self.shape,"float32")
            self.array_imag = empty(self.shape,"float32")            
                    
        self._split_complex_pointer = _create_split_complex_pointer(self.array_real, self.array_imag, double = self.double)
    
    @property            
    def split_complex_pointer(self):
        if self._split_complex_pointer is not None:
            return self._split_complex_pointer
        else:
            self.allocate_memory()
            return self._split_complex_pointer
                    
def _init_setup_and_data(a, overwrite_x, split_complex, dim):
    a = np.asarray(a)
    try:    
        setup = FFTSETUP_DATA[(a.shape[-dim:],a.dtype)] 
    except KeyError:
        setup = [FFTSetupData(a.shape[-dim:],a.dtype) for i in range(fft_config["nthreads"])]    
        FFTSETUP_DATA[(a.shape[-dim:],a.dtype)] = setup
    
    # convert to a valid dtype
    # all setups have same dtype, so it is ok to take the first element
    #print(setup[0].dtype, a.dtype)
    a = np.asarray(a,setup[0].dtype) 
    if split_complex:
        out = empty((2,)+ a.shape, a.real.dtype)       
    else:
        if overwrite_x == False:
            out = empty(a.shape, a.dtype)
        else:
            out = a
    return setup, a, out

def _init_setup_and_data_split(a, dim):
    real, imag = a
    real, imag = np.asarray(real), np.asarray(imag)
    assert real.dtype == imag.dtype
    assert real.shape == imag.shape
    try:    
        setup = FFTSETUP_DATA[(real.shape[-dim:],real.dtype)] 
    except KeyError:
        setup = [FFTSetupData(real.shape[-dim:],real.dtype) for i in range(fft_config["nthreads"])]    
        FFTSETUP_DATA[(real.shape[-dim:],real.dtype)] = setup
    
    # convert to a valid dtype
    # all setups have same double attribute so it is ok to take the first element
    dtype = "float64" if setup[0].double == True else "float32"
    real, imag = np.asarray(real, dtype), np.asarray(imag, dtype)

    return setup, real, imag


##################
# Worker functions
##################

def _ffti(setup, a, out, direction = +1):

    n = len(a)    

    _a = ffi.from_buffer(a) #make buffer from numpy data
    _pa = ffi.cast(setup.cast_name,_a) #pointer to buffer
   
    _ps = setup.split_complex_pointer #pointer to split complex data
    _out = ffi.from_buffer(out) 
    _pout = ffi.cast(setup.cast_name,_out) #pointer to buffer

    if setup.double:
        lib.vDSP_ctozD(_pa,2,_ps,1,n)
        lib.vDSP_fft_zipD(setup.pointer, _ps, 1, setup.size[0], direction)
        lib.vDSP_ztocD(_ps,1,_pout,2,n)        
    else:
        lib.vDSP_ctoz(_pa,2,_ps,1,n)
        lib.vDSP_fft_zip(setup.pointer, _ps, 1, setup.size[0], direction)
        lib.vDSP_ztoc(_ps,1,_pout,2,n)
 
def _fftis(setup, a, outr, outi, direction = +1):

    n = len(a)    

    _a = ffi.from_buffer(a) #make buffer from numpy data
    _pa = ffi.cast(setup.cast_name,_a) #pointer to buffer

    _ps = _create_split_complex_pointer(outr, outi, double = setup.double)
    
    if setup.double:
        lib.vDSP_ctozD(_pa,2,_ps,1,n)
        lib.vDSP_fft_zipD(setup.pointer, _ps, 1, setup.size[0], direction)
    
    else:
        lib.vDSP_ctoz(_pa,2,_ps,1,n)
        lib.vDSP_fft_zip(setup.pointer, _ps, 1, setup.size[0], direction)
                    
def _sfftis(setup, ar,ai, direction = +1):
    _ps = _create_split_complex_pointer(ar, ai, double = setup.double)
    if setup.double:
        lib.vDSP_fft_zipD(setup.pointer, _ps, 1, setup.size[0], direction)     
    else:
        lib.vDSP_fft_zip(setup.pointer, _ps, 1, setup.size[0], direction)

def _sfftos(setup, ar,ai, outr, outi, direction = +1):
    _pins = _create_split_complex_pointer(ar, ai, double = setup.double)
    _pouts = _create_split_complex_pointer(outr, outi, double = setup.double)
    if setup.double:
        lib.vDSP_fft_zopD(setup.pointer, _pins, 1, _pouts, 1,setup.size[0], direction)     
    else:
        lib.vDSP_fft_zop(setup.pointer, _pins, 1, _pouts, 1, setup.size[0], direction)

def _fft2i(setup, a, out, direction = +1):
    """low level fft worker function. Input and output array must be 2D, contiguous arrays"""
    n = a.shape[0] * a.shape[1]
    
    _a = ffi.from_buffer(a) #make buffer from numpy data
    _out = ffi.from_buffer(out) 
    
    _pa = ffi.cast(setup.cast_name,_a) #pointer to buffer
    _pout = ffi.cast(setup.cast_name,_out) #pointer to buffer
    _ps = setup.split_complex_pointer #pointer to split complex data
    
    if setup.double:
        lib.vDSP_ctozD(_pa,2,_ps,1,n)
        lib.vDSP_fft2d_zipD(setup.pointer, _ps, 1, 0, setup.size[0], setup.size[1],direction)
        lib.vDSP_ztocD(_ps,1,_pout,2,n)
    else:
        lib.vDSP_ctoz(_pa,2,_ps,1,n)
        lib.vDSP_fft2d_zip(setup.pointer, _ps, 1, 0, setup.size[0], setup.size[1],direction)
        lib.vDSP_ztoc(_ps,1,_pout,2,n)

def _fft2is(setup, a, outr, outi, direction = +1):
    """low level fft worker function. Input and output array must be 2D, contiguous arrays"""
    n = a.shape[0] * a.shape[1]
    
    _a = ffi.from_buffer(a) #make buffer from numpy data
    
    _pa = ffi.cast(setup.cast_name,_a) #pointer to buffer
    _ps = _create_split_complex_pointer(outr, outi, double = setup.double)
    
    if setup.double:
        lib.vDSP_ctozD(_pa,2,_ps,1,n)
        lib.vDSP_fft2d_zipD(setup.pointer, _ps, 1, 0, setup.size[0], setup.size[1],direction)
    else:
        lib.vDSP_ctoz(_pa,2,_ps,1,n)
        lib.vDSP_fft2d_zip(setup.pointer, _ps, 1, 0, setup.size[0], setup.size[1],direction)

def _fft(a,overwrite_x = False, split_complex = False, direction = +1):
    setup, a, out = _init_setup_and_data(a, overwrite_x,split_complex,1)
    shape = _optimal_flattened_shape(a.shape,  dim = 1)
    _a = a.reshape(shape)
    if split_complex == False:
        _out =  out.reshape(shape)
        _calculate_fft(_ffti, setup, _a,_out, direction = direction)
    else:
        _out_real, _out_imag =  out[0].reshape(shape), out[1].reshape(shape)
        _calculate_fft(_fftis, setup, _a,_out_real,_out_imag, direction = direction)
    return out
    

def _sfft(a,overwrite_x = False, direction = +1):
    setup, real, imag = _init_setup_and_data_split(a, dim = 1)
    shape = _optimal_flattened_shape(real.shape,  dim = 1)
    _real,_imag = real.reshape(shape), imag.reshape(shape)
    if overwrite_x == True:
        _calculate_fft(_sfftis, setup, _real, _imag, direction = direction)
        return real, imag
    else:
        out_real, out_imag = empty(real.shape, real.dtype), empty(imag.shape, imag.dtype)
        _out_real,_out_imag = out_real.reshape(shape), out_imag.reshape(shape)
        _calculate_fft(_sfftos, setup, _real, _imag, _out_real, _out_imag, direction = direction)
        return out_real, out_imag

def _fft2(a,overwrite_x = False, split_complex = False, direction = +1):
    setup, a, out = _init_setup_and_data(a, overwrite_x,split_complex,2)
    shape = _optimal_flattened_shape(a.shape,  dim = 2)
    _a = a.reshape(shape)
    if split_complex == False:
        _out =  out.reshape(shape)
        _calculate_fft(_fft2i, setup, _a,_out, direction = direction)
    else:
        _out_real, _out_imag =  out[0].reshape(shape), out[1].reshape(shape)
        _calculate_fft(_fft2is, setup, _a,_out_real,_out_imag, direction = direction)
    return out

def fft(a, overwrite_x = False, split_complex = False):
    """Returns dicrete Fourer transform."""
    return _fft(a,overwrite_x = overwrite_x, split_complex = split_complex, direction = +1)

def ifft(a, overwrite_x = False, split_complex = False):
    """Returns dicrete Fourer transform."""
    return _fft(a,overwrite_x = overwrite_x, split_complex = split_complex, direction = +1)
    
def sfft(a, overwrite_x = False):
    return _sfft(a,overwrite_x,+1)

def isfft(a, overwrite_x = False):
    return _sfft(a,overwrite_x,-1)
        
def fft2(a, overwrite_x = False, split_complex = False):
    """Returns dicrete 2D Fourer transform."""
    return _fft2(a,overwrite_x = overwrite_x, split_complex = split_complex, direction = +1)

def ifft2(a, overwrite_x = False, split_complex = False):
    """Returns dicrete 2D Fourer transform."""
    return _fft2(a,overwrite_x = overwrite_x, split_complex = split_complex, direction = -1)
   
    