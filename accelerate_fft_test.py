import unittest
import accelerate_fft as fft
import numpy as np

np.random.seed(0)

f64 =  np.array([[1,2,3,4],[4,5,6,7],[1,2,3,1],[3,4,1,2]], "float64")
f32 =  np.array([[1,2,3,4],[4,5,6,7],[1,2,3,1],[3,4,1,2]], "float32")
c64 =  np.array([[1,2,3,4],[4,5,6,7],[1,2,3,1],[3,4,1,2]], "complex64")
c128 = np.array([[1,2,3,4],[4,5,6,7],[1,2,3,1],[3,4,1,2]], "complex128")

class BaseTest(unittest.TestCase):
    def setUp(self):
        fft.set_nthreads(1)
    def assert_equal(self,a,b, rtol = 1.e-4,atol=1.e-6):
        self.assertTrue(np.allclose(a,b,rtol,atol))
    
class TestShape(BaseTest):
    
    def test_wrong_fft_shape(self):
        with self.assertRaises(ValueError):
            fft.fft([1,2,3])

    def test_wrong_fft2_shape(self):
        with self.assertRaises(ValueError):
            fft.fft2([[1,2,3],[1,2,3]])
            
class TestDtype(BaseTest):
    
    def test_rfft_128(self):
        with self.assertWarns(np.ComplexWarning):
            fft.rfft(c128)
            
    def test_rfft_64(self):
        with self.assertWarns(np.ComplexWarning):
            fft.rfft(c64)

class TestResults(BaseTest):
    
    def test_rfft_float(self):
        a = np.array(np.random.randn(2,4,8),"float32")
        b = a.reshape(2,8,4)
        self.assert_equal(fft.unpack(fft.rfft(a))/2., np.fft.rfft(a))
        self.assert_equal(fft.unpack(fft.rfft(b))/2., np.fft.rfft(b))

    def test_rfft_split_out_float(self):
        a = np.array(np.random.randn(2,4,8),"float32")
        b = a.reshape(2,8,4)
        for x in (a,b):
            r,i = fft.rfft(x, split_out = True)
            c = r + 1j*i
            self.assert_equal(fft.unpack(c)/2., np.fft.rfft(x))

    def test_rfft_double(self):
        a = np.array(np.random.randn(2,4,8),"float32")
        b = a.reshape(2,8,4)
        self.assert_equal(fft.unpack(fft.rfft(a))/2., np.fft.rfft(a))
        self.assert_equal(fft.unpack(fft.rfft(b))/2., np.fft.rfft(b))
        
    def test_rfft_split_out_double(self):
        a = np.array(np.random.randn(2,4,8),"float64")
        b = a.reshape(2,8,4)
        for x in (a,b):
            r,i = fft.rfft(x, split_out = True)
            c = r + 1j*i
            self.assert_equal(fft.unpack(c)/2., np.fft.rfft(x))

    def test_fft_float(self):
        a = np.array(np.random.randn(2,4,8),"complex64")
        b = a.reshape(2,8,4)
        self.assert_equal(fft.fft(a), np.fft.fft(a))
        self.assert_equal(fft.fft(b), np.fft.fft(b))
        
    def test_fft_double(self):
        a = np.array(np.random.randn(2,4,8),"complex128")
        b = a.reshape(2,8,4) 
        self.assert_equal(fft.fft(a), np.fft.fft(a))
        self.assert_equal(fft.fft(b), np.fft.fft(b))
    
    def test_fft2_float(self):
        a = np.array(np.random.randn(2,4,8),"complex64")
        b = a.reshape(2,8,4) 
        self.assert_equal(fft.fft2(a), np.fft.fft2(a))
        self.assert_equal(fft.fft2(b), np.fft.fft2(b))
        
    def test_fft2_double(self):
        a = np.array(np.random.randn(2,4,8),"complex128")
        b = a.reshape(2,8,4)
        self.assert_equal(fft.fft2(a), np.fft.fft2(a))
        self.assert_equal(fft.fft2(b), np.fft.fft2(b))
        
    def test_rfft2_float(self):
        a = np.array(np.random.randn(2,4,8),"float32")
        b = a.reshape(2,8,4)
        self.assert_equal(fft.fft2(a), np.fft.fft2(a))
        self.assert_equal(fft.fft2(b), np.fft.fft2(b))
        
    def test_rfft2_double(self):
        a = np.array(np.random.randn(2,4,8),"float64")
        b = a.reshape(2,8,4)
        self.assert_equal(fft.unpack2(fft.rfft2(a))/2, np.fft.rfft2(a))
        self.assert_equal(fft.unpack2(fft.rfft2(b))/2, np.fft.rfft2(b))
            
    def test_ifft_float(self):
        a = np.array(np.random.randn(2,4,8),"float32")
        b = a.reshape(2,8,4)
        self.assert_equal(fft.ifft(a)/8, np.fft.ifft(a))
        self.assert_equal(fft.ifft(b)/4, np.fft.ifft(b))
        
    def test_ifft_double(self):
        a = np.array(np.random.randn(2,4,8),"float64")
        b = a.reshape(2,8,4)
        self.assert_equal(fft.ifft(a)/8, np.fft.ifft(a))
        self.assert_equal(fft.ifft(b)/4, np.fft.ifft(b))
        
    def test_ifft2_float(self):
        a = np.array(np.random.randn(2,4,8),"float32")
        b = a.reshape(2,8,4)
        self.assert_equal(fft.ifft2(a)/(8*4), np.fft.ifft2(a))
        self.assert_equal(fft.ifft2(b)/(8*4), np.fft.ifft2(b))

    def test_ifft2_double(self):
        a = np.array(np.random.randn(2,4,8),"float64")
        b = a.reshape(2,8,4)
        self.assert_equal(fft.ifft2(a)/(8*4), np.fft.ifft2(a))
        self.assert_equal(fft.ifft2(b)/(8*4), np.fft.ifft2(b))
        
    def test_irfft_float(self):
        a = np.array(np.random.randn(2,4,8),"float32")
        b = a.reshape(2,8,4)
        self.assert_equal(fft.irfft(fft.rfft(a))/8/2, a)
        self.assert_equal(fft.irfft(fft.rfft(b))/4/2, b)
        
    def test_irfft_double(self):
        a = np.array(np.random.randn(2,4,8),"float64")
        b = a.reshape(2,8,4)
        self.assert_equal(fft.irfft(fft.rfft(a))/8/2, a)
        self.assert_equal(fft.irfft(fft.rfft(b))/4/2, b)
        
    def test_irfft2_float(self):
        a = np.array(np.random.randn(2,4,8),"float32")
        b = a.reshape(2,8,4)
        self.assert_equal(fft.irfft2(fft.rfft2(a))/8/4/2, a)
        self.assert_equal(fft.irfft2(fft.rfft2(b))/4/8/2, b)

    def test_irfft2_double(self):
        a = np.array(np.random.randn(2,4,8),"float64")
        b = a.reshape(2,8,4)
        self.assert_equal(fft.irfft2(fft.rfft2(a))/8/4/2, a)
        self.assert_equal(fft.irfft2(fft.rfft2(b))/4/8/2, b)
        
    def test_fft2_double_split(self):
        a = np.array(np.random.randn(2,4,8),"float64")
        a_split = a.real.copy(), a.imag.copy()
        with self.subTest(i = 0):
            self.assert_equal(fft.ifft2(fft.fft2(a, split_out = True), split_in = True)/8/4, a)
        with self.subTest(i = 1):
            out_split = fft.ifft2(fft.fft2(a, split_out = True), split_in = True, split_out = True)
            self.assert_equal(a_split[0], out_split[0]/8/4)
            self.assert_equal(a_split[1], out_split[1]/8/4)
        with self.subTest(i = 2):
            out_split = fft.ifft2(fft.fft2(a), split_out = True)
            self.assert_equal(a_split[0], out_split[0]/8/4)
            self.assert_equal(a_split[1], out_split[1]/8/4)  
        with self.subTest(i = 3):
            self.assert_equal(fft.ifft2(fft.fft2(a_split, split_in = True))/8/4, a)
        with self.subTest(i = 4):
            out_split = fft.ifft2(fft.fft2(a_split, split_in = True),split_out = True)
            self.assert_equal(a_split[0], out_split[0]/8/4)
            self.assert_equal(a_split[1], out_split[1]/8/4)   
        with self.subTest(i = 5):
            out_split = fft.ifft2(fft.fft2(a_split, split_in = True, split_out = True),split_out = True,split_in = True)
            self.assert_equal(a_split[0], out_split[0]/8/4)
            self.assert_equal(a_split[1], out_split[1]/8/4)                                

    def test_fft2_float_split(self):
        a = np.array(np.random.randn(2,4,8),"float32")
        a_split = a.real.copy(), a.imag.copy()
        with self.subTest(i = 0):
            self.assert_equal(fft.ifft2(fft.fft2(a, split_out = True), split_in = True)/8/4, a)
        with self.subTest(i = 1):
            out_split = fft.ifft2(fft.fft2(a, split_out = True), split_in = True, split_out = True)
            self.assert_equal(a_split[0], out_split[0]/8/4)
            self.assert_equal(a_split[1], out_split[1]/8/4)
        with self.subTest(i = 2):
            out_split = fft.ifft2(fft.fft2(a), split_out = True)
            self.assert_equal(a_split[0], out_split[0]/8/4)
            self.assert_equal(a_split[1], out_split[1]/8/4)  
        with self.subTest(i = 3):
            self.assert_equal(fft.ifft2(fft.fft2(a_split, split_in = True))/8/4, a)
        with self.subTest(i = 4):
            out_split = fft.ifft2(fft.fft2(a_split, split_in = True),split_out = True)
            self.assert_equal(a_split[0], out_split[0]/8/4)
            self.assert_equal(a_split[1], out_split[1]/8/4)   
        with self.subTest(i = 5):
            out_split = fft.ifft2(fft.fft2(a_split, split_in = True, split_out = True),split_out = True,split_in = True)
            self.assert_equal(a_split[0], out_split[0]/8/4)
            self.assert_equal(a_split[1], out_split[1]/8/4)    

    def test_fft_double_split(self):
        a = np.array(np.random.randn(8),"complex128")
        a0 = a.copy()
        a_split = a.real.copy(), a.imag.copy()
        with self.subTest(i = 0):
            out = fft.ifft(fft.fft(a, split_out = True), split_in = True)/8
            self.assert_equal(out, a0)
        with self.subTest(i = 1):
            out_split = fft.ifft(fft.fft(a, split_out = True), split_in = True, split_out = True)
            self.assert_equal(a0.real, out_split[0]/8)
            self.assert_equal(a_split[1], out_split[1]/8)
        with self.subTest(i = 2):
            out_split = fft.ifft(fft.fft(a), split_out = True)
            self.assert_equal(a_split[0], out_split[0]/8)
            self.assert_equal(a_split[1], out_split[1]/8)  
        with self.subTest(i = 3):
            self.assert_equal(fft.ifft(fft.fft(a_split, split_in = True))/8, a)
        with self.subTest(i = 4):
            out_split = fft.ifft(fft.fft(a_split, split_in = True),split_out = True)
            self.assert_equal(a_split[0], out_split[0]/8)
            self.assert_equal(a_split[1], out_split[1]/8)   
        with self.subTest(i = 5):
            out_split = fft.ifft(fft.fft(a_split, split_in = True, split_out = True),split_out = True,split_in = True)
            self.assert_equal(a_split[0], out_split[0]/8)
            self.assert_equal(a_split[1], out_split[1]/8)     

    def test_fft_float_split(self):
        a = np.array(np.random.randn(8),"complex64")
        a0 = a.copy()
        a_split = a.real.copy(), a.imag.copy()
        with self.subTest(i = 0):
            out = fft.ifft(fft.fft(a, split_out = True), split_in = True)/8
            self.assert_equal(out, a0)
        with self.subTest(i = 1):
            out_split = fft.ifft(fft.fft(a, split_out = True), split_in = True, split_out = True)
            self.assert_equal(a0.real, out_split[0]/8)
            self.assert_equal(a_split[1], out_split[1]/8)
        with self.subTest(i = 2):
            out_split = fft.ifft(fft.fft(a), split_out = True)
            self.assert_equal(a_split[0], out_split[0]/8)
            self.assert_equal(a_split[1], out_split[1]/8)  
        with self.subTest(i = 3):
            self.assert_equal(fft.ifft(fft.fft(a_split, split_in = True))/8, a)
        with self.subTest(i = 4):
            out_split = fft.ifft(fft.fft(a_split, split_in = True),split_out = True)
            self.assert_equal(a_split[0], out_split[0]/8)
            self.assert_equal(a_split[1], out_split[1]/8)   
        with self.subTest(i = 5):
            out_split = fft.ifft(fft.fft(a_split, split_in = True, split_out = True),split_out = True,split_in = True)
            self.assert_equal(a_split[0], out_split[0]/8)
            self.assert_equal(a_split[1], out_split[1]/8)   
                               
    def test_rfft2_double_split(self):
        a = np.array(np.random.randn(2,4,8),"float64")
        with self.subTest(i = 0):
            self.assert_equal(fft.irfft2(fft.rfft2(a, split_out = True), split_in = True)/8/4/2, a)

    def test_rfft2_float_split(self):
        a = np.array(np.random.randn(2,4,8),"float32")
        with self.subTest(i = 0):
            self.assert_equal(fft.irfft2(fft.rfft2(a, split_out = True), split_in = True)/8/4/2, a)

    def test_rfft_double_split(self):
        a = np.array(np.random.randn(2,4,8),"float64")
        with self.subTest(i = 0):
            self.assert_equal(fft.irfft(fft.rfft(a, split_out = True), split_in = True)/8/2, a)

    def test_rfft_float_split(self):
        a = np.array(np.random.randn(2,4,8),"float32")
        with self.subTest(i = 0):
            self.assert_equal(fft.irfft(fft.rfft(a, split_out = True), split_in = True)/8/2, a)

  
class TestResultsThreaded( TestResults):
    
    def setUp(self):
        fft.set_nthreads(2)
                       
if __name__ == '__main__':
    unittest.main()
    
    
    