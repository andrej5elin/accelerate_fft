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

def _optimal_flattened_shape(original_shape, dim = 1):
    """Finds a shape for a flattened array, so that we can then iterate over it"""
    nthreads = fft_config["nthreads"]
    newshape = reduce((lambda x,y: x*y), original_shape[:-dim] or [1])
    if fft_config["nthreads"] > 1:
        n = _optimal_workers(newshape,nthreads)
        newshape = (n,newshape//n,) + original_shape[-dim:]
    else:
        newshape = (newshape,) + original_shape[-dim:]
    return newshape

def _sequential_call(fftfunc,setup, *args,**kwargs):
    # a simple sequential runner 
    [fftfunc(setup,*arg,**kwargs) for arg in zip(*args)] 

def _calculate_fft(fftfunc,*args,**kwds):
    """Runs fft function with given arguments (optionally in parallel 
    using a ThreadPool)"""
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
            
        self.double = dtype in ("float64","complex128") 

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

def _get_in_out_dtype(in_dtype, split_in = False, split_out = False, direction = +1, real_transform = False):
    """Determines input and output arrays dtype from the calculation parameters"""
    
    double = in_dtype in ("float64",  "complex128")
    f,c = (np.dtype("float64"), np.dtype("complex128")) if double else (np.dtype("float32"), np.dtype("complex64"))
    
    if real_transform == True:
        if direction == +1:
            return f , f if split_out else c 
        else:
            return f if split_in else c, f
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
     
def _init_setup_and_arrays(a, overwrite_x, dim = 1, split_in = False, split_out = False, direction = +1, real_transform = False):
    """checks input parameters and creates a valid fft setup and input/output arrays"""
    # make an array and read dtype, and shape
    a = (np.asarray(a[0]),np.asarray(a[1])) if split_in else np.asarray(a)
    dtype = a[0].dtype if split_in else a.dtype
    shape = a[0].shape if split_in else a.shape
    
    # determine the allowed type and shapes of the input and output arrays
    in_dtype, out_dtype = _get_in_out_dtype(dtype, split_in, split_out, direction, real_transform)
    out_shape = _get_out_shape(shape, dim, direction, real_transform)

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
        
    try:    
        setup = FFTSETUP_DATA[(out_shape[-dim:],out_dtype)] 
    except KeyError:
        setup = [FFTSetupData(out_shape[-dim:],out_dtype) for i in range(fft_config["nthreads"])]    
        FFTSETUP_DATA[(out_shape[-dim:],out_dtype)] = setup
        
    return setup, a, out

##################
# Worker functions
##################

def _ffti(setup, a, out, direction = +1):
    """inplace fft transform"""

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
    """inplace fft transform with splitted output"""
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
    """inplace fft transform with splitted input and output"""
    _ps = _create_split_complex_pointer(ar, ai, double = setup.double)
    if setup.double:
        lib.vDSP_fft_zipD(setup.pointer, _ps, 1, setup.size[0], direction)     
    else:
        lib.vDSP_fft_zip(setup.pointer, _ps, 1, setup.size[0], direction)

def _sfftos(setup, ar,ai, outr, outi, direction = +1):
    """out-of-place fft transform with splitted input and output"""
    _pins = _create_split_complex_pointer(ar, ai, double = setup.double)
    _pouts = _create_split_complex_pointer(outr, outi, double = setup.double)
    if setup.double:
        lib.vDSP_fft_zopD(setup.pointer, _pins, 1, _pouts, 1,setup.size[0], direction)     
    else:
        lib.vDSP_fft_zop(setup.pointer, _pins, 1, _pouts, 1, setup.size[0], direction)

def _fft2i(setup, a, out, direction = +1):
    """inplace fft2 transform"""

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
    """inplace fft transform with splitted output"""
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

def _rfft2i(setup, a, out, direction = +1):
    """inplace fft2 transform"""

    n = a.shape[0] * a.shape[1] //2
    
    _a = ffi.from_buffer(a) #make buffer from numpy data
    _out = ffi.from_buffer(out) 
    
    _pa = ffi.cast(setup.cast_name,_a) #pointer to buffer
    _pout = ffi.cast(setup.cast_name,_out) #pointer to buffer
    _ps = setup.split_complex_pointer #pointer to split complex data

    if setup.double:
        lib.vDSP_ctozD(_pa,2,_ps,1,n)
        if direction :
            lib.vDSP_fft2d_zripD(setup.pointer, _ps, 1, 0, setup.size[0], setup.size[1]+1,direction)
        else:
            lib.vDSP_fft2d_zripD(setup.pointer, _ps, 1, 0, setup.size[0], setup.size[1],direction)

        lib.vDSP_ztocD(_ps,1,_pout,2,n)
    else:
        lib.vDSP_ctoz(_pa,2,_ps,1,n)
        if direction:
            lib.vDSP_fft2d_zrip(setup.pointer, _ps, 1, 0, setup.size[0], setup.size[1]+1,direction)
        else:
            lib.vDSP_fft2d_zrip(setup.pointer, _ps, 1, 0, setup.size[0], setup.size[1],direction)

        lib.vDSP_ztoc(_ps,1,_pout,2,n)

def _rfft2is(setup, a, outr, outi, direction = +1):
    """inplace fft transform with splitted output"""
    n = a.shape[0] * a.shape[1]//2
    
    _a = ffi.from_buffer(a) #make buffer from numpy data
    
    _pa = ffi.cast(setup.cast_name,_a) #pointer to buffer
    _ps = _create_split_complex_pointer(outr, outi, double = setup.double)

    if setup.double:
        lib.vDSP_ctozD(_pa,2,_ps,1,n)
        lib.vDSP_fft2d_zripD(setup.pointer, _ps, 1, 0, setup.size[0], setup.size[1]+1,direction)
    else:
        lib.vDSP_ctoz(_pa,2,_ps,1,n)
        lib.vDSP_fft2d_zrip(setup.pointer, _ps, 1, 0, setup.size[0], setup.size[1]+1,direction)



def generalized_fft(a,dim = 1, overwrite_x = False, split_in = False, split_out = False, direction = +1, real_transform = False):
    setup, a, out = _init_setup_and_arrays(a, overwrite_x, dim = dim, split_in = split_in, split_out = split_out, direction = direction, real_transform = real_transform)
    
    if split_in == True:
        a_real, a_imag = a
        shape = _optimal_flattened_shape(a_real.shape, dim = dim)
        _a_real, _a_imag = a_real.reshape(shape), a_imag.reshape(shape)
       
        if split_out == True:
            if overwrite_x == False:
                #out of place transform
                out_real, out_imag = out
                shape = _optimal_flattened_shape(out_real.shape,  dim = dim)
                _out_real, _out_imag =  out_real.reshape(shape), out_imag.reshape(shape)
                _fftfunc = _sfftos if dim == 1 else None
                _calculate_fft(_fftfunc, setup, _a_real,_a_imag,_out_real,_out_imag, direction = direction)
            else:
                #inplace transform
                _fftfunc = _sfftis if dim == 1 else None
                _calculate_fft(_fftfunc, setup, _a_real,_a_imag,direction = direction)

    else:
        shape = _optimal_flattened_shape(a.shape, dim = dim)
        _a = a.reshape(shape)
            
        if split_out == False:
            shape = _optimal_flattened_shape(out.shape, dim = dim)
            _out =  out.reshape(shape)
            if real_transform:
                _fftfunc = None if dim == 1 else _rfft2i
            else:
                _fftfunc = _ffti if dim == 1 else _fft2i
            _calculate_fft(_fftfunc, setup, _a,_out, direction = direction)
            
        else:
            shape = _optimal_flattened_shape(out[0].shape,  dim = dim)
            _out_real, _out_imag =  out[0].reshape(shape), out[1].reshape(shape)
            if real_transform:
                _fftfunc = None if dim == 1 else _rfft2is
            else:
                _fftfunc = _fftis if dim == 1 else _fft2is
            
            
            _calculate_fft(_fftfunc, setup, _a,_out_real,_out_imag, direction = direction)
    
    return out
    

def fft(a, overwrite_x = False, split_complex = False):
    """Returns dicrete Fourer transform."""
    return generalized_fft(a,overwrite_x = overwrite_x, split_out = split_complex, direction = +1)

def ifft(a, overwrite_x = False, split_complex = False):
    """Returns dicrete Fourer transform."""
    return generalized_fft(a,overwrite_x = overwrite_x, split_out = split_complex, direction = +1)
    
def sfft(a, overwrite_x = False):
    return generalized_fft(a,overwrite_x = overwrite_x, split_in = True, split_out = True, direction = +1)

def isfft(a, overwrite_x = False):
    return generalized_fft(a,overwrite_x = overwrite_x, split_in = True, split_out = True, direction = -1)
        
def fft2(a, overwrite_x = False, split_complex = False):
    """Returns dicrete 2D Fourer transform."""
    return generalized_fft(a,overwrite_x = overwrite_x, split_out = split_complex, direction = +1, dim = 2)

def ifft2(a, overwrite_x = False, split_complex = False):
    """Returns dicrete 2D Fourer transform."""
    return generalized_fft(a,overwrite_x = overwrite_x, split_out = split_complex, direction = -1, dim = 2)
  
def rfft2(a, split_complex = False):
    return generalized_fft(a, split_out = split_complex, direction = +1, dim = 2, real_transform = True)

#def irfft2(a, split_complex = False):
#    return generalized_fft(a, split_out = split_complex, direction = -1, dim = 2, real_transform = True)
    