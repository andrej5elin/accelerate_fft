accelerate_fft
==============

Wrapper of Apple's Accelerate (vDSP) FFT routines. It implements multiple-signal version of ffts.

Note that vDSP FFT only works for input data of length that is a power of 2, e.g. ... 256, 512, 1024 ... Input data has to be contiguous. Transformation is performed over the last axis (1d fft) and last two axes (2d fft).

This module has a very primitive multi-threading support. Useful for large-size multi-signal 2d FFT. Input data has to be multi-dimensional, and it must have a size that is a multiple of nthread * nfft, where nthread is number of threads used, and nfft is the size of the fft.

Why?
----
For intel-based macs just use `mkl_fft <https://github.com/IntelPython/mkl_fft>`_, which is one of the fastest FFT libraries out there.
The accelerate_fft was developed for M-powered macs, where the accelerate framework is expected to be faster than MKL running over Rosetta.

Requisites
----------

* cffi
* numpy

Although the package should build fine under Rosetta, you really should have an osx-arm64 (conda-forge) python distribution if you use this on Apple Silicon chips.  

Installation
------------

Install with pip::

    $ pip install accelerate_fft
    
or clone the latest development version and run::

    $ python setup.py install

Usage
-----

Use it like numpy fft::

    >>> import numpy as np
    >>> import accelerate_fft as fft
    >>> a = np.random.randn(4,8)
    >>> np.allclose(fft.fft(a), np.fft.fft(a))
    True
    
Note that real transform in accelerate has a wierd format, and is not 
the same as numpy's rfft and rfft2. Use :func:`unpack` or :func:`unpack2` to 
convert to numpy-like format. Note also the scaling factor 2::

    >>> np.allclose(fft.unpack(fft.rfft(a))/2, np.fft.rfft(a))
    True
    >>> np.allclose(fft.unpack2(fft.rfft2(a))/2, np.fft.rfft2(a))
    True
    
The inverse transforms are also different from numpy's. You should scale the results to get the proper inverse like::

    >>> np.allclose(fft.ifft(fft.fft(a))/8, a)
    True
    >>> np.allclose(fft.ifft2(fft.fft2(a))/8/4, a)
    True
    >>> np.allclose(fft.irfft(fft.rfft(a))/8/2, a)
    True
    >>> np.allclose(fft.irfft2(fft.rfft2(a))/8/4/2, a)
    True    
  
    
For multi-dimensional data (ndim >= 3), you can use threading to speed up computation of 2d transforms::

    >>> a = np.random.randn(32,512,512) + 0j 
    >>> fft.set_nthreads(4)
    1
    >>> out = fft.fft2(a)
    
Note that for small arrays, because of the python thread-creation overhead, this may be slower than::

    >>> fft.set_nthreads(1)
    4

License
-------

``dtmm`` is released under MIT license so you can use it freely.


