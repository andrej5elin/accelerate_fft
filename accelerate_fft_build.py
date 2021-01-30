from cffi import FFI
ffibuilder = FFI()

# cdef() expects a single string declaring the C types, functions and
# globals needed to use the shared object. It must be in valid C syntax.

ffibuilder.cdef("""     
                

typedef unsigned long vDSP_Length;
typedef long long     vDSP_Stride;


typedef struct DSPComplex {
    float  real;
    float  imag;
} DSPComplex;
typedef struct DSPDoubleComplex {
    double real;
    double imag;
} DSPDoubleComplex;

typedef struct DSPSplitComplex {
    float  * realp;
    float  * imagp;
} DSPSplitComplex;
typedef struct DSPDoubleSplitComplex {
    double * realp;
    double * imagp;
} DSPDoubleSplitComplex;


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
    

typedef struct OpaqueFFTSetup           *FFTSetup;
typedef struct OpaqueFFTSetupD          *FFTSetupD;


extern FFTSetup vDSP_create_fftsetup(
    vDSP_Length __Log2n,
    FFTRadix    __Radix)
		;

void vDSP_destroy_fftsetup( FFTSetup __setup)
        ;

FFTSetupD vDSP_create_fftsetupD(vDSP_Length __Log2n, FFTRadix __Radix);
void vDSP_destroy_fftsetupD(FFTSetupD __setup);

extern void vDSP_ctoz(
    const DSPComplex      *__C,
    vDSP_Stride            __IC,
    const DSPSplitComplex *__Z,
    vDSP_Stride            __IZ,
    vDSP_Length            __N)
       ;


extern void vDSP_ztoc(
    const DSPSplitComplex *__Z,
    vDSP_Stride            __IZ,
    DSPComplex            *__C,
    vDSP_Stride            __IC,
    vDSP_Length            __N)
        ;


void vDSP_ctozD(const DSPDoubleComplex *__C, vDSP_Stride __IC, const DSPDoubleSplitComplex *__Z, vDSP_Stride __IZ, vDSP_Length __N);
void vDSP_ztocD(const DSPDoubleSplitComplex *__Z, vDSP_Stride __IZ, DSPDoubleComplex *__C, vDSP_Stride __IC, vDSP_Length __N);

void vDSP_fft_zipt(FFTSetup __Setup, const DSPSplitComplex *__C, vDSP_Stride __IC, const DSPSplitComplex *__Buffer, vDSP_Length __Log2N, FFTDirection __Direction);
void vDSP_fft_ziptD(FFTSetupD __Setup, const DSPDoubleSplitComplex *__C, vDSP_Stride __IC, const DSPDoubleSplitComplex *__Buffer, vDSP_Length __Log2N, FFTDirection __Direction);

void vDSP_fft_zrip(FFTSetup __Setup, const DSPSplitComplex *__C, vDSP_Stride __IC, vDSP_Length __Log2N, FFTDirection __Direction);
void vDSP_fft_zripD(FFTSetupD __Setup, const DSPDoubleSplitComplex *__C, vDSP_Stride __IC, vDSP_Length __Log2N, FFTDirection __Direction);

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
         
void vDSP_fft2d_zripD(FFTSetupD __Setup, const DSPDoubleSplitComplex *__C, vDSP_Stride __IC0, vDSP_Stride __IC1, vDSP_Length __Log2N0, vDSP_Length __Log2N1, FFTDirection __flag);
void vDSP_fft2d_zrip(FFTSetup __Setup, const DSPSplitComplex *__C, vDSP_Stride __IC0, vDSP_Stride __IC1, vDSP_Length __Log2N0, vDSP_Length __Log2N1, FFTDirection __Direction);
void vDSP_fft2d_zop(FFTSetup __Setup, const DSPSplitComplex *__A, vDSP_Stride __IA0, vDSP_Stride __IA1, const DSPSplitComplex *__C, vDSP_Stride __IC0, vDSP_Stride __IC1, vDSP_Length __Log2N0, vDSP_Length __Log2N1, FFTDirection __Direction);
void vDSP_fft2d_zopD(FFTSetupD __Setup, const DSPDoubleSplitComplex *__A, vDSP_Stride __IA0, vDSP_Stride __IA1, const DSPDoubleSplitComplex *__C, vDSP_Stride __IC0, vDSP_Stride __IC1, vDSP_Length __Log2N0, vDSP_Length __Log2N1, FFTDirection __Direction);
void vDSP_fft2d_zrop(FFTSetup __Setup, const DSPSplitComplex *__A, vDSP_Stride __IA0, vDSP_Stride __IA1, const DSPSplitComplex *__C, vDSP_Stride __IC0, vDSP_Stride __IC1, vDSP_Length __Log2N0, vDSP_Length __Log2N1, FFTDirection __Direction);
void vDSP_fft2d_zropD(FFTSetupD __Setup, const DSPDoubleSplitComplex *__A, vDSP_Stride __IA0, vDSP_Stride __IA1, const DSPDoubleSplitComplex *__C, vDSP_Stride __IC0, vDSP_Stride __IC1, vDSP_Length __Log2N0, vDSP_Length __Log2N1, FFTDirection __Direction);

void vDSP_fftm_zip(FFTSetup __Setup, const DSPSplitComplex *__C, vDSP_Stride __IC, vDSP_Stride __IM, vDSP_Length __Log2N, vDSP_Length __M, FFTDirection __Direction);
void vDSP_fftm_zipD(FFTSetupD __Setup, const DSPDoubleSplitComplex *__C, vDSP_Stride __IC, vDSP_Stride __IM, vDSP_Length __Log2N, vDSP_Length __M, FFTDirection __Direction);
void vDSP_fftm_zop(FFTSetup __Setup, const DSPSplitComplex *__A, vDSP_Stride __IA, vDSP_Stride __IMA, const DSPSplitComplex *__C, vDSP_Stride __IC, vDSP_Stride __IMC, vDSP_Length __Log2N, vDSP_Length __M, FFTDirection __Direction);
void vDSP_fftm_zopD(FFTSetupD __Setup, const DSPDoubleSplitComplex *__A, vDSP_Stride __IA, vDSP_Stride __IMA, const DSPDoubleSplitComplex *__C, vDSP_Stride __IC, vDSP_Stride __IMC, vDSP_Length __Log2N, vDSP_Length __M, FFTDirection __Direction);
void vDSP_fftm_zrip(FFTSetup __Setup, const DSPSplitComplex *__C, vDSP_Stride __IC, vDSP_Stride __IM, vDSP_Length __Log2N, vDSP_Length __M, FFTDirection __Direction);
void vDSP_fftm_zripD(FFTSetupD __Setup, const DSPDoubleSplitComplex *__C, vDSP_Stride __IC, vDSP_Stride __IM, vDSP_Length __Log2N, vDSP_Length __M, FFTDirection __Direction);
void vDSP_fftm_zrop(FFTSetup __Setup, const DSPSplitComplex *__A, vDSP_Stride __IA, vDSP_Stride __IMA, const DSPSplitComplex *__C, vDSP_Stride __IC, vDSP_Stride __IMC, vDSP_Length __Log2N, vDSP_Length __M, FFTDirection __Direction);
void vDSP_fftm_zropD(FFTSetupD __Setup, const DSPDoubleSplitComplex *__A, vDSP_Stride __IA, vDSP_Stride __IMA, const DSPDoubleSplitComplex *__C, vDSP_Stride __IC, vDSP_Stride __IMC, vDSP_Length __Log2N, vDSP_Length __M, FFTDirection __Direction);
                """)

ffibuilder.set_source("_accelerate_fft_cffi",
"""
     #include <Accelerate/Accelerate.h>   // the C header of the library
"""
#,extra_link_args=["-framework Accelerate"]
#,
    #,libraries=['framework Accelerate']
     )   # library name, for the linker

if __name__ == "__main__":
    ffibuilder.compile(verbose=True)
    
        