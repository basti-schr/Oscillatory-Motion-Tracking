import numpy as np
from quaternion import Quaternion


class MahonyAHRS:
    """
    Madgwick's implementation of Mayhony's AHRS algorithm

    For more information see: http://www.x-io.co.uk/node/8#open_source_ahrs_and_imu_algorithms
    or https://github.com/xioTechnologies/Oscillatory-Motion-Tracking-With-x-IMU/blob/master/%40MahonyAHRS/MahonyAHRS.m

    This is ported from the MATLAB class by SOH Madgwick from 28.09.2011

    :author: basti-schr (basti.schr@gmx.de)
    """

    sample_period = 1 / 256
    quaternion = Quaternion(1, 0, 0, 0)  # output quaternion describing the Earth relative to the sensor
    kp = 1  # algorithm proportional gain
    ki = 0  # algorithm integral gain
    _e_int = np.array([.0, .0, .0])  # integral error

    @staticmethod
    def norm(vector):
        size = 0
        for i in vector:
            size += i ** 2
        if size <= 0:
            raise ZeroDivisionError
        size = np.sqrt(size)
        return (np.array(vector) / size).tolist()

    def __init__(self, sample_period=None, quaternion=None, kp=None, ki=None):
        if sample_period is None:
            sample_period = 1 / 256
        if quaternion is None:
            quaternion = Quaternion(1, 0, 0, 0)
        if kp is None:
            kp = 1
        if ki is None:
            ki = 0

        self.sample_period = sample_period
        self.quaternion = quaternion  # output quaternion describing the Earth relative to the sensor
        self.kp = kp  # algorithm proportional gain
        self.ki = ki  # algorithm integral gain
        self._e_int = np.array([.0, .0, .0])  # integral error

    def update_imu(self, gyroscope, accelerometer):
        gyroscope = np.array(gyroscope)
        accelerometer = np.array(accelerometer)
        q = self.quaternion  # short name local variable for readability

        # normalize accelerometer measurement
        self.norm(accelerometer)

        # estimate the direction of gravity
        v = [2 * (q[1] * q[3] - q[0] * q[2]),
             2 * (q[0] * q[1] + q[2] * q[3]),
             q[0] ** 2 - q[1] ** 2 - q[2] ** 2 + q[3] ** 2
             ]

        # Error is sum of cross product between estimated direction and measured direction of field
        e = np.cross(accelerometer, v)
        if self.ki > 0:
            self._e_int += e * self.sample_period
        else:
            self._e_int = np.array([.0, .0, .0])

        # Apply feedback terms
        gyroscope = gyroscope + self.kp * e + self.ki * self._e_int

        # Compute rate of change of quaternion
        q_dot = q * Quaternion(np.hstack((0, gyroscope)))
        q_dot *= 0.5

        # Integrate to yield quaternion
        q += q_dot * self.sample_period
        self.quaternion = q.norm()

    def update(self, gyroscope, accelerometer, magnetometer):
        print("not implemented at this time! Continue as IMU ...")
        self.update_imu(gyroscope, accelerometer)
