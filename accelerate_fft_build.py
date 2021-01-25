from cffi import FFI
ffibuilder = FFI()

# cdef() expects a single string declaring the C types, functions and
# globals needed to use the shared object. It must be in valid C syntax.

ffibuilder.cdef("""     
                
/*  Define types:

        vDSP_Length for numbers of elements in arrays and for indices of
        elements in arrays.  (It is also used for the base-two logarithm of
        numbers of elements, although a much smaller type is suitable for
        that.)

        vDSP_Stride for differences of indices of elements (which of course
        includes strides).
*/
typedef unsigned long vDSP_Length;

typedef long long     vDSP_Stride;


/*  A DSPComplex or DSPDoubleComplex is a pair of float or double values that
    together represent a complex value.
*/
typedef struct DSPComplex {
    float  real;
    float  imag;
} DSPComplex;
typedef struct DSPDoubleComplex {
    double real;
    double imag;
} DSPDoubleComplex;


/*  A DSPSplitComplex or DSPDoubleSplitComplex is a structure containing
    two pointers, each to an array of float or double.  These represent arrays
    of complex values, with the real components of the values stored in one
    array and the imaginary components of the values stored in a separate
    array.
*/
typedef struct DSPSplitComplex {
    float  * realp;
    float  * imagp;
} DSPSplitComplex;
typedef struct DSPDoubleSplitComplex {
    double * realp;
    double * imagp;
} DSPDoubleSplitComplex;


/*  The following statements declare a few simple types and constants used by
    various vDSP routines.
*/
typedef int FFTDirection;
typedef int FFTRadix;
enum {
    kFFTDirection_Forward         = +1,
    kFFTDirection_Inverse         = -1
};
enum {
    kFFTRadix2                    = 0,
    kFFTRadix3                    = 1,
    kFFTRadix5                    = 2
};
enum {
    vDSP_HALF_WINDOW              = 1,
    vDSP_HANN_DENORM              = 0,
    vDSP_HANN_NORM                = 2
};
    

/*  The following types define 24-bit data.
*/
typedef struct { uint8_t bytes[3]; } vDSP_uint24; // Unsigned 24-bit integer.
typedef struct { uint8_t bytes[3]; } vDSP_int24;  // Signed 24-bit integer.


/*  The following types are pointers to structures that contain data used
    inside vDSP routines to assist FFT and biquad filter operations.  The
    contents of these structures may change from release to release, so
    applications should manipulate the values only via the corresponding vDSP
    setup and destroy routines.
*/
typedef struct OpaqueFFTSetup           *FFTSetup;
typedef struct OpaqueFFTSetupD          *FFTSetupD;
typedef struct vDSP_biquad_SetupStruct  *vDSP_biquad_Setup;
typedef struct vDSP_biquad_SetupStructD *vDSP_biquad_SetupD;

    
/*  vDSP_biquadm_Setup or vDSP_biquadm_SetupD is a pointer to a filter object
    to be used with a multi-channel cascaded biquad IIR.  This object carries
    internal state which may be modified by any routine which uses it.  Upon
    creation, the state is initialized such that all delay elements are zero.
 
    Each filter object should only be used in a single thread at a time.
*/
typedef struct vDSP_biquadm_SetupStruct  *vDSP_biquadm_Setup;
typedef struct vDSP_biquadm_SetupStructD *vDSP_biquadm_SetupD;


/*  vDSP_create_fftsetup and vDSP_create_ffsetupD allocate memory and prepare
    constants used by single- and double-precision FFT routines, respectively.

    vDSP_destroy_fftsetup and vDSP_destroy_fftsetupD free the memory.  They
    may be passed a null pointer, in which case they have no effect.
*/
extern  FFTSetup vDSP_create_fftsetup(
    vDSP_Length __Log2n,
    FFTRadix    __Radix)
		;

extern void vDSP_destroy_fftsetup( FFTSetup __setup)
        ;

FFTSetupD vDSP_create_fftsetupD(vDSP_Length __Log2n, FFTRadix __Radix);
void vDSP_destroy_fftsetupD(FFTSetupD __setup);


// Convert a complex array to a complex-split array.
extern void vDSP_ctoz(
    const DSPComplex      *__C,
    vDSP_Stride            __IC,
    const DSPSplitComplex *__Z,
    vDSP_Stride            __IZ,
    vDSP_Length            __N)
       ;
    /*  Map:

            Pseudocode:     Memory:
            C[n]            C[n*IC/2].real + i * C[n*IC/2].imag
            Z[n]            Z->realp[n*IZ] + i * Z->imagp[n*IZ]

        These compute:

            for (n = 0; n < N; ++n)
                Z[n] = C[n];
    */


//  Convert a complex-split array to a complex array.
extern void vDSP_ztoc(
    const DSPSplitComplex *__Z,
    vDSP_Stride            __IZ,
    DSPComplex            *__C,
    vDSP_Stride            __IC,
    vDSP_Length            __N)
        ;
    /*  Map:

            Pseudocode:     Memory:
            Z[n]            Z->realp[n*IZ] + i * Z->imagp[n*IZ]
            C[n]            C[n*IC/2].real + i * C[n*IC/2].imag

        These compute:

            for (n = 0; n < N; ++n)
                C[n] = Z[n];
    */

void vDSP_ctozD(const DSPDoubleComplex *__C, vDSP_Stride __IC, const DSPDoubleSplitComplex *__Z, vDSP_Stride __IZ, vDSP_Length __N);
void vDSP_ztocD(const DSPDoubleSplitComplex *__Z, vDSP_Stride __IZ, DSPDoubleComplex *__C, vDSP_Stride __IC, vDSP_Length __N);

/*  In-place real-to-complex Discrete Fourier Transform routines, with and
    without temporary memory.  We suggest you use the DFT routines instead of
    these.
*/

void vDSP_fft_zip(FFTSetup __Setup, const DSPSplitComplex *__C, vDSP_Stride __IC, vDSP_Length __Log2N, FFTDirection __Direction);
void vDSP_fft_zipD(FFTSetupD __Setup, const DSPDoubleSplitComplex *__C, vDSP_Stride __IC, vDSP_Length __Log2N, FFTDirection __Direction);
void vDSP_fft2d_zip(FFTSetup __Setup, const DSPSplitComplex *__C, vDSP_Stride __IC0, vDSP_Stride __IC1, vDSP_Length __Log2N0, vDSP_Length __Log2N1, FFTDirection __Direction);
void vDSP_fft2d_zipD(FFTSetupD __Setup, const DSPDoubleSplitComplex *__C, vDSP_Stride __IC0, vDSP_Stride __IC1, vDSP_Length __Log2N0, vDSP_Length __Log2N1, FFTDirection __Direction);
 
extern void vDSP_fft_zop(
    FFTSetup               __Setup,
    const DSPSplitComplex *__A,
    vDSP_Stride            __IA,
    const DSPSplitComplex *__C,
    vDSP_Stride            __IC,
    vDSP_Length            __Log2N,
    FFTDirection           __Direction);

extern void vDSP_fft_zopD(
    FFTSetupD                    __Setup,
    const DSPDoubleSplitComplex *__A,
    vDSP_Stride                  __IA,
    const DSPDoubleSplitComplex *__C,
    vDSP_Stride                  __IC,
    vDSP_Length                  __Log2N,
    FFTDirection                 __Direction);
         
                """)

ffibuilder.set_source("_accelerate_fft_cffi",
"""
     #include <Accelerate/Accelerate.h>   // the C header of the library
"""#,
    # libraries=['accelerate']
     )   # library name, for the linker

if __name__ == "__main__":
    ffibuilder.compile(verbose=True)
    
        