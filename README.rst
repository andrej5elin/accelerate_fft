accelerate_fft
==============

.. image:: https://img.shields.io/pypi/pyversions/accelerate_fft
    :target: https://pypi.org/project/accelerate_fft/
    :alt: Python version
    
.. image:: https://github.com/andrej5elin/accelerate_fft/workflows/Upload%20Python%20Package/badge.svg  
    :target: https://github.com/andrej5elin/accelerate_fft/

.. image:: https://codecov.io/gh/andrej5elin/accelerate_fft/branch/main/graph/badge.svg
    :target: https://codecov.io/gh/andrej5elin/accelerate_fft


CFFI-based wrapper of Apple's Accelerate (vDSP) FFT routines. It implements multiple-signal version of 1D ffts (real and complex) and 2D ffts (real and complex). In the current version, the transform is performed over the last (1d fft) axis or the last two (2d fft) axes. Therefore, if a different axis or axes are requested, or if input data is not c-contiguous, a full copy of input data is made.

Note that vDSP's FFT only work for input data of length that is a power of 2, e.g. ... 256, 512, 1024 ... 

This module has a very primitive multi-threading support. Useful for large-size multi-signal 2d FFT and multi-signal 1d FFT. For multi-threaded calculation, input data has to have at least three dimesnions for 2d transform and at least two dimensions for 1d transform.

Why?
----
For Intel-based macs use `mkl_fft <https://github.com/IntelPython/mkl_fft>`_, which is one of the fastest FFT libraries out there.
However, the Accelerate implementation can be faster than MKL on Apple Silicon macs. See benchmarks below.



Requisites
----------

* cffi
* numpy  

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
    
Note that real transforms in accelerate have a wierd format, and the outputs are not 
the same as numpy's rfft and rfft2. Use :func:`unpack` or :func:`unpack2` to 
convert to numpy-like format. Note also the scaling factor of 2::

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
    
Because vDSP's FFT work in split-complex data format. You may prepare and retrieve the data in this format directly. This is controlled by the `split_in` and `split_out` arguments, e.g.::

    >>> real, imag = fft.fft2(a, split_out = True)
    >>> real, imag = fft.ifft2((real, imag), split_in = True, split_out = True, overwrite_x = True)
    
Here, the `overwrite_x` argument defines that the transform is done inplace and it overwrites the input data.

Benchmarks
----------

Testing was done using `fft2` on input array of shape `(8,2048,2048)` and of `"complex64"` dtype.

+------------------+------+------+------+------+------+------+------+------+------+------+------+------+
|                  |      MAC Mini (2020) ,M1, 8GB           | MBP (Early 2015), 2.7 GHz Intel i5, 8GB |
+------------------+------+------+------+------+------+------+------+------+------+------+------+------+
|  library         | ``accelerate_fft`` |    ``mkl_fft``     | ``accelerate_fft`` |    ``mkl_fft``     |
+------------------+------+------+------+------+------+------+------+------+------+------+------+------+
|  N threads       |   1  |   2  |   4  |   1  |   2  |   4  |   1  |   2  |   4  |   1  |   2  |   4  |
+==================+======+======+======+======+======+======+======+======+======+======+======+======+  
| t (ms) normal    |  195 |  116 |  85  | 204  |  108 |  66  |  522 |  324 | 298  | 294  |  179 | 180  |
+------------------+------+------+------+------+------+------+------+------+------+------+------+------+
| t (ms) inplace   |  147 |   97 |  59  | 200  |  100 |  59  |  334 |  188 | 171  | 248  |  154 | 153  |
+------------------+------+------+------+------+------+------+------+------+------+------+------+------+

The inplace version (`overwrite_x = True`) was done on split data format (`split_in = True` and `split_out = True`) for the ``accelerate_fft``. Results show that on intel-based MACs, ``mkl_fft`` performs better, however, on Apple Silicon, ``accelerate_fft`` (single-core computation) can be slightly faster. Note that for ``mkl_fft`` you have to call::

>>> [mkl_fft.fft2(x, overwrite_x = True) for x in a]

whereas 

>>> mkl_fft.fft2(a, overwrite_x = True)

was much slower to compute. 

Release notes
-------------

Version 0.2.0
+++++++++++++

Improves multi-threaded fft calculation on smaller arrays and in 1D transforms. 
Implements 'axis' and 'axes' arguments.

License
-------

``accelerate_fft`` is released under MIT license so you can use it freely.


