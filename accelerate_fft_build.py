from cffi import FFI
ffibuilder = FFI()

# cdef() expects a single string declaring the C types, functions and
# globals needed to use the shared object. It must be in valid C syntax.
ffibuilder.cdef("""
typedef unsigned long vDSP_Length;
typedef int FFTRadix;
typedef struct OpaqueFFTSetup *FFTSetup;

struct DSPComplex {
  float               real;
  float               imag;
};
typedef struct DSPComplex               DSPComplex;

struct DSPSplitComplex {
  float *             realp;
  float *             imagp;
};
typedef struct DSPSplitComplex          DSPSplitComplex;

typedef long vDSP_Stride;
typedef int FFTDirection;
                
                
void vDSP_vmul(
    float *__A,       /* input vector 1 */
    long __IA, /* address stride for input vector 1 */
    float *__B,       /* input vector 2 */
    long __IB, /* address stride for input vector 2 */
    float *__C,       /* output vector */
    long __IC, /* address stride for output vector */
    unsigned long __N   /* real output count */
);

FFTSetup vDSP_create_fftsetup(vDSP_Length __Log2n, FFTRadix __Radix);
void vDSP_destroy_fftsetup(FFTSetup __setup);
void vDSP_ctoz(const DSPComplex *__C, vDSP_Stride __IC, const DSPSplitComplex *__Z, vDSP_Stride __IZ, vDSP_Length __N);
void vDSP_ztoc(const DSPSplitComplex *__Z, vDSP_Stride __IZ, DSPComplex *__C, vDSP_Stride __IC, vDSP_Length __N);


void vDSP_fft_zip(FFTSetup __Setup, const DSPSplitComplex *__C, vDSP_Stride __IC, vDSP_Length __Log2N, FFTDirection __Direction);
void vDSP_fft2d_zip(FFTSetup __Setup, const DSPSplitComplex *__C, vDSP_Stride __IC0, vDSP_Stride __IC1, vDSP_Length __Log2N0, vDSP_Length __Log2N1, FFTDirection __Direction);


""")

# set_source() gives the name of the python extension module to
# produce, and some C source code as a string.  This C code needs
# to make the declarated functions, types and globals available,
# so it is often just the "#include".
ffibuilder.set_source("_accelerate_fft_cffi",
"""
     #include <Accelerate/Accelerate.h>   // the C header of the library
"""#,
    # libraries=['accelerate']
     )   # library name, for the linker

if __name__ == "__main__":
    ffibuilder.compile(verbose=True)
    
        