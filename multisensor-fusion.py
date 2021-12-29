import matplotlib.pyplot as plt
import numpy as np

size = 60
# Create the vectors X and Y
t = np.array(range(size))/3
y = -4 * (np.exp(-0.2 * t)) + 4 #ground truth

#mean and standard dev
mean = 1;
sd = 0.2;

#sensor 1
noise_y1 = np.random.normal(mean,sd, size) #mean of 1, sd of 0.2
y1 = y + noise_y1;
#sensor 2
noise_y2 = np.random.normal(mean,sd, size)
y2 = y + noise_y2;

# Create the plot
plt.plot(t, y1, label='accelerometer 1', linestyle='dashed')
plt.plot(t, y2, label='accelerometer 2', linestyle='dashed');
plt.plot(t, y, label='ground truth', color='green');

#fusion

#time step
dt = 1.00/3.00;
#state matrix
X = np.zeros((2,1));
#covariance matrix
P = np.zeros((2,2));
#system noise
#Q = np.array([[1, 2],[1, 1]]);
Q = np.array([[0.04, 0],[0, 1]])
#observation matrix
H = np.array([1, 0]);
#expected X
EX = np.zeros((size, 2));
#Transition matrix
F = np.array([[1, dt], [0, 1]])

#initialize the kalman filter
def init_kalman(X, y):
    X[0][0] = y[0];
    X[1][0] = 0;
    P = np.array([[100, 0],[0,300]])
    return X, P;

#return [X, P]; X is the predicted val (expected val), P is the probability
def prediction(X, P, Q, F):
    X = np.matmul(F, X);
    P = np.matmul(np.matmul(F, P), F.transpose()) + Q;
    return X, P;

#return [X, P]
def update(X, P, y, R, H):
    Inn = y - np.matmul(H, X);
    S = np.matmul(np.matmul(H, P), H.transpose()) + R;
    K = np.array(np.matmul(P, H.transpose()) / S).reshape(2,1);
    X = X + K*Inn;
    P = P - np.matmul(np.matmul(K, H.reshape(1,2)), P);
    return X, P;

calibrated_bias = -1;
for i in range(size):
    if(i == 0): #initialize using first sensor
        X, P = init_kalman(X, y1+calibrated_bias);
    else: 
        X, P = prediction(X, P, Q, F);
        X, P = update(X, P, y1[i]+calibrated_bias, noise_y1[i], H);
        X, P = update(X, P, y2[i]+calibrated_bias, noise_y2[i], H);
    
    EX[i][0] = X[0][0];
    EX[i][1] = X[1][0];


plt.plot(t, EX[:,0], label='fused', color='red');

plt.xlabel('time (s)');
plt.ylabel('acceleration ms^-1');
plt.legend();
plt.title("Multisensor Fusion with bias by UMass Rocket Team")
# Show the plot
plt.show() 