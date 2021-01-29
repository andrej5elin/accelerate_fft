# accelerate_fft
A simple fft using apple's accelerate (vDSP) library.

Wrapper of apple's accelerate (vDSP) FFT routines. It implements multiple-signal
version of ffts. 

Note that vDSP FFT only works for input data of length that is a power of 2,
e.g. ... 256, 512, 1024 ... Input data has to be contiguous. Transformation
is performed over the last axis (1d fft) and last two axes (2d fft).

This module has a very primitive multi-threading support. Useful for 
large-size multi-signal 2d FFT. Input data has to be multi-dimensional, and it 
must have a size that is a multiple of nthread  * nfft, where
nthread is number of threads used, and nfft is the size of the fft. 

