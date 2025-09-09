# B12504097 光電實驗week1__算gamma value

import numpy as np
#--------------------------------------------------------------------------------------------------------------#
Y     = np.array([240, 403, 630, 912], float)   # 積分Y from CIECCalculator
codes = np.array([153, 179, 204, 230], float)   # G值
Y_G255 = 1281                                   # 積分Y from CIECCalculator
#--------------------------------------------------------------------------------------------------------------#

Lrel = Y / Y_G255

m = (codes > 0) & (Lrel > 0)
x = np.log(codes[m] / 255.0)
y = np.log(Lrel[m])

gamma = (x @ y) / (x @ x)
print(f"gamma = {gamma:.4f}")

