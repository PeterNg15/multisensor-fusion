# Multisensor Fusion using a Kalman Filter
# Fusing together data in a homogeneous sensor network using a Kalman filter. 

When we combine sensor readings we increase the quality of data by reducing noises and uncertainty that exists in individual sensors. These noises are unpredictable so we can’t get rid of them through calibration, therefore this data filtering step is necessary. One approach would be to find the weighted average of the sensor outputs (the weight is the inverse of the sensor’s noise variance). A limitation of this approach is that we cannot control the smoothness of the fused output. Another approach is to modify the Kalman filter so that it would update #n times (where #n is the number of sensors in the network) after the prediction step. The Kalman filter approach can propagate the variance as well as control the smoothness of the fused output through the system noise matrix Q. By fusing #n sensors we can reduce the noise by (1 - 1/sqrt(n))%.

# Output
![image](https://user-images.githubusercontent.com/77201829/147696029-14c3554e-f95d-4b83-b7d8-c8be63ece858.png)
