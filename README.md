# Project of Line-by-line Absorption Coefficient Solver

### Brief Introduction
This project tackles the issue of computing the line-by-line absorption coefficient, which is one of the complex challenges when simulating the Earth’s atmosphere. The solution is implemented based on the Atmospheric Radiative Transfer Simulator book and the Radis library in Python. The absorption coefficient is computed using data from the HITRAN line database. As a result, this might help to enable direct access to the measured data if we want to simulate some gas spectra. Simple optimizations are added to the implementation to improve computation speed while ensuring accuracy.

To view the **final report**, please open or download the file "SP_Line_by_line_Solver_Report_Yuxin.pdf"

### Important Setup
1. install [RADIS](https://radis.readthedocs.io/en/latest/) by 
   **pip install radis**
2. install [pyfftw](https://pypi.org/project/pyFFTW/) by **pip install pyFFTW**

### Run
Go to the **main.py**, in the fucntion **main()** either read the inputs from get_atm_gas_data() or set your own molecule, isotope, pressure, and temperature as inputs. Then just run the script to get results and the plot of absorption coefficient.

Test examples of the computation is in the tests folder.

For the final presentation of this project, please open "Line-by-line-solver/website/index.html"

### Remarks
To make sure the program is able to reach the data at the correct location, please check:
- the path to the file **"molparam.txt"** which contains the molar mass of molecules in **"basics.py"** is the correct path in your machine
- the path to the file **"afgl_1986-us_standard.nc"** which contains the real atmosphere data in **"atmosphere.py"** is the correct path in your machine
- the data path to the molecular data downloaded from HITRAN (load_hitran() in loader.py) is correct on your machine.