from derivative import dxdt
import numpy as np
import matplotlib.pyplot as plt

t = np.linspace(0,2*np.pi,50)
x = np.sin(t)*0.01

# 1. Finite differences with central differencing using 3 points.
result1 = dxdt(x, t, kind="finite_difference", k=1)

# 2. Savitzky-Golay using cubic polynomials to fit in a centered window of length 1
result2 = dxdt(x, t, kind="savitzky_golay", left=.5, right=.5, order=3)

# 3. Spectral derivative
result3 = dxdt(x, t, kind="spectral")

# 4. Spline derivative with smoothing set to 0.01
result4 = dxdt(x, t, kind="spline", s=1e-2)

# 5. Total variational derivative with regularization set to 0.01
result5 = dxdt(x, t, kind="trend_filtered", order=0, alpha=1e-2)

# 6. Kalman derivative with smoothing set to 1
result6 = dxdt(x, t, kind="kalman", alpha=1)



plt.plot(t,x)
plt.plot(t,result1)
plt.show()