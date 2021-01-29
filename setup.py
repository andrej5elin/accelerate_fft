    
from setuptools import setup #, find_packages


long_description = """
Implements fft using apple's accelerate framework (vDSP)
"""

#packages = find_packages()

setup(name = 'accelerate_fft',
      version = "0.1.0",
      description = "FFT for MAC using vDSP",
      long_description=long_description,
      long_description_content_type="text/markdown",
      author = 'Andrej Petelin',
      author_email = 'andrej.petelin@gmail.com',
      url="https://github.com/andrej5elin/accelerate_fft",
      py_modules = ["accelerate_fft"],
      #packages = packages,
      #include_package_data=True
      #package_data={
        # If any package contains *.dat, or *.ini include them:
      #  '': ['*.dat',"*.ini"]},
      
      setup_requires=["cffi>=1.0.0"],
      cffi_modules=["accelerate_fft_build.py:ffibuilder"], # "filename:global"
      install_requires=["cffi>=1.0.0"],
      classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS X",
    ]
      )