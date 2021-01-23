__version__ = "0.0.1"

from _accelerate_fft_cffi import ffi, lib
import numpy as np

#: a placehold for FFT setups. This gets populated when calling fft()
FFT = {}

def clear_setup(name = None):
    """Clears fft setup"""
    if name is None:
        FFT.clear()
    else:
        FFT.pop(name)

class _FFTSetup():
    def __init__(self, shape):
        radix = 2 #only radix 2 supported
        self.size = int(np.log2(shape[0]))
        if shape[0] != 2**self.size:
            raise ValueError("Size of the array must be a power of 2.")
        if len(shape) != 1:
            raise ValueError("Only 1-D arrays supported")
            
        self.pointer = lib.vDSP_create_fftsetup(self.size, radix)
        self.array_real = np.empty((2**self.size),"float32")
        _array_real = ffi.from_buffer(self.array_real)
        self.array_imag = np.empty((2**self.size),"float32")
        _array_imag = ffi.from_buffer(self.array_imag)
        
        self.split_complex_pointer = ffi.new("DSPSplitComplex*") 
        self.split_complex_pointer.realp = ffi.cast("float*",_array_real)
        self.split_complex_pointer.imagp = ffi.cast("float*",_array_imag)
        FFT[shape] = self
        
    def __del__(self):
        lib.vDSP_destroy_fftsetup(self.pointer)
                    
def fft(a, overwrite_x = False):
    """Returns dicrete Fourer transform of complex data."""
    
    a = np.asarray(a)
    if a.dtype not in ("complex64",):
        raise ValueError("Only complex64 supported")
    if overwrite_x == False:
        out = np.empty_like(a)
    else:
        out = a
    try:    
        setup = FFT[a.shape]
    except KeyError:
        setup = _FFTSetup(a.shape)
    n = len(a)
    
    _a = ffi.from_buffer(a) #make buffer from numpy data
    _out = ffi.from_buffer(out) 
    _pa = ffi.cast("DSPComplex*",_a) #pointer to buffer
    _pout = ffi.cast("DSPComplex*",_out) #pointer to buffer
   
    _ps = setup.split_complex_pointer #pointer to split complex data
    
    lib.vDSP_ctoz(_pa,2,_ps,1,n)
    lib.vDSP_fft_zip(setup.pointer, _ps, 1, setup.size, +1)
    lib.vDSP_ztoc(_ps,1,_pout,2,n)
    
    return out
    
    
        
        
        
        

   
    