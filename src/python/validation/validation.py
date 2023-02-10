from os import getcwd
from matplotlib import pyplot as plt
import numpy as np
from utils import readVI
from src.fd1d_TL import coaxial_TL


py_voltage, py_current = readVI(getcwd()+"\\python\\logs\\output.txt")
for_voltage, for_current = readVI(getcwd()+"\\fortran\\logs\\output.txt")
    
plt.figure(figsize=(8, 3.5))

plt.subplot(211)
plt.plot(py_voltage, 'k', linewidth=1, label = 'python')
plt.plot(for_voltage, 'r--', linewidth=1, label = 'fortran')
plt.ylabel('$V$', fontsize='14')
plt.ylim(0, 35)
plt.legend()

plt.subplot(212)
plt.plot(py_current, 'k', linewidth=1, label = 'python')
plt.plot(for_current, 'r--', linewidth=1, label = 'fortran')
plt.ylabel('$I$', fontsize='14')
plt.xlabel('FDTD cells')
plt.ylim(0, 0.2)

plt.subplots_adjust(bottom=0.2, hspace=0.45)
plt.show()
    
