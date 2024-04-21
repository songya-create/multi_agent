from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise
from numpy.random import randn
import numpy as np
from numpy.random import randn
from filterpy.kalman import UnscentedKalmanFilter
import matplotlib.pyplot as plt
from filterpy.kalman import UnscentedKalmanFilter as UKF
from filterpy.kalman import unscented_transform, MerweScaledSigmaPoints


def f_cv(x, dt):
    """ 匀速直线运动函数"""

    F = np.array([[1, dt, 0, 0],
                  [0, 1, 0, 0],
                  [0, 0, 1, dt],
                  [0, 0, 0, 1]])
    return F @ x


def h_cv(x):
    return x[[0, 2]]


std_x, std_y = .3, .3
dt = 1.0
zs = [np.array([i + randn() * std_x,
                i + randn() * std_y]) for i in range(100)]
np.random.seed(1234)
sigmas = MerweScaledSigmaPoints(4, alpha = .1, beta = 2., kappa = 1.)
ukf = UKF(dim_x = 4, dim_z = 2, fx = f_cv,
          hx = h_cv, dt = dt, points = sigmas)
ukf.x = np.array([0., 0., 0., 0.])
ukf.R = np.diag([0.09, 0.09])
ukf.Q[0:2, 0:2] = Q_discrete_white_noise(2, dt = 1, var = 0.02)
ukf.Q[2:4, 2:4] = Q_discrete_white_noise(2, dt = 1, var = 0.02)

uxs = []
for z in zs:
    ukf.predict()
    ukf.update(z)
    uxs.append(ukf.x.copy())
uxs = np.array(uxs)

plt.plot(uxs[:, 0], uxs[:, 2])
print(ukf)