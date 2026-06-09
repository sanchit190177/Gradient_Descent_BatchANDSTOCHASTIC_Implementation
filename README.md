### Batch GD
import numpy  as np
np.random.seed(3)
x = 2*np.random.rand(100,1) # independent feature
y = 4+3*x+np.random.rand(100,1) # dependent variable
x_b = np.c_[np.ones((100,1)),x]

# define learning rate 
eta=0.1
n_iterations=1000 ## define no of iterations. 
theta = np.random.rand(2,1)
m=x_b.shape[0]
epoch_error=[]
for _ in range(n_iterations):
    predicted = x_b@theta
    error = predicted - y
    gradient = 2/m*x_b.T@(error)
    theta = theta - eta*gradient
    epoch_error.append(sum(error**2)/m)
print(theta)
plt.figure(figsize=(7,3))
plt.plot(np.linspace(1,1000,1000),epoch_error)
plt.xlabel('Epochs')
plt.ylabel('Error')
plt.show()

### Stochastic GD
n_epochs=50
eta = 0.01
t1 = max(100,m) 
t0 = t1*eta
theta = np.random.rand(2,1)

def learning_schedule(t):
    return t0/(t+t1)
    
for epoch in range(n_epochs):
    for i in range(m):
        random_index = np.random.randint(0,m)
        xi=x_b[random_index:random_index+1]
        yi=y[random_index:random_index+1]
        gradient = 2*xi.T@(xi@theta-yi)
        eta = learning_schedule(epoch*m+i)
        theta = theta-eta*gradient
print(theta)
