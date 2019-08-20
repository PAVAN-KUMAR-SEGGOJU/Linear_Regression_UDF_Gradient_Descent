# Linear_Regression_UDF_Gradient_Descent
Linear Regression source code development in python using gradient descent technique
Reference : https://machinelearningmastery.com/linear-regression-tutorial-using-gradient-descent-for-machine-learning/

Simple Linear Regression
When we have a single in put attribute (x) and we want to use linear regression, this is called simple linear regression.

With simple linear regression we want to model our data as follows:

y = B0 + B1 * x

This is a line where y is the output variable we want to predict, x is the input variable we know and B0 and B1 are coefficients we need to estimate.

B0 is called the intercept because it determines where the line intercepts the y axis. In machine learning we can call this the bias, because it is added to offset all predictions that we make. The B1 term is called the slope because it defines the slope of the line or how x translates into a y value before we add our bias.

The model is called Simple Linear Regression because there is only one input variable (x). If there were more input variables (e.g. x1, x2, etc.) then this would be called multiple regression.

Stochastic Gradient Descent
Gradient Descent is the process of minimizing a function by following the gradients of the cost function.

This involves knowing the form of the cost as well as the derivative so that from a given point you know the gradient and can move in that direction, e.g. downhill towards the minimum value.

In Machine learning we can use a similar technique called stochastic gradient descent to minimize the error of a model on our training data.

The way this works is that each training instance is shown to the model one at a time. The model makes a prediction for a training instance, the error is calculated and the model is updated in order to reduce the error for the next prediction.

This procedure can be used to find the set of coefficients in a model that result in the smallest error for the model on the training data. Each iteration the coefficients, called weights (w) in machine learning language are updated using the equation:

w = w – alpha * delta

Where w is the coefficient or weight being optimized, alpha is a learning rate that you must configure (e.g. 0.1) and gradient is the error for the model on the training data attributed to the weight.

Simple Linear Regression with Stochastic Gradient Descent
The coefficients used in simple linear regression can be found using stochastic gradient descent.

Linear regression is a linear system and the coefficients can be calculated analytically using linear algebra. Stochastic gradient descent is not used to calculate the coefficients for linear regression in practice (in most cases).

Linear regression does provide a useful exercise for learning stochastic gradient descent which is an important algorithm used for minimizing cost functions by machine learning algorithms.

As stated above, our linear regression model is defined as follows:

y = B0 + B1 * x

Gradient Descent Iteration #1
Let’s start with values of 0.0 for both coefficients.

B0 = 0.0

B1 = 0.0

y = 0.0 + 0.0 * x

We can calculate the error for a prediction as follows:

error = p(i) – y(i)

Where p(i) is the prediction for the i’th instance in our dataset and y(i) is the i’th output variable for the instance in the dataset.

We can now calculate he predicted value for y using our starting point coefficients for the first training instance:

x=1, y=1

p(i) = 0.0 + 0.0 * 1

p(i) = 0

Using the predicted output, we can calculate our error:

error = 0 – 1

error = -1

We can now use this error in our equation for gradient descent to update the weights. We will start with updating the intercept first, because it is easier.

We can say that B0 is accountable for all of the error. This is to say that updating the weight will use just the error as the gradient. We can calculate the update for the B0 coefficient as follows:

B0(t+1) = B0(t) – alpha * error

Where B0(t+1) is the updated version of the coefficient we will use on the next training instance, B0(t) is the current value for B0 alpha is our learning rate and error is the error we calculate for the training instance. Let’s use a small learning rate of 0.01 and plug the values into the equation to work out what the new and slightly optimized value of B0 will be:

B0(t+1) = 0.0 – 0.01 * -1.0

B0(t+1) = 0.01

Now, let’s look at updating the value for B1. We use the same equation with one small change. The error is filtered by the input that caused it. We can update B1 using the equation:

B1(t+1) = B1(t) – alpha * error * x

Where B1(t+1) is the update coefficient, B1(t) is the current version of the coefficient, alpha is the same learning rate described above, error is the same error calculated above and x is the input value.

We can plug in our numbers into the equation and calculate the updated value for B1:

B1(t+1) = 0.0 – 0.01 * -1 * 1

B1(t+1) = 0.01

We have just finished the first iteration of gradient descent and we have updated our weights to be B0=0.01 and B1=0.01. This process must be repeated for the remaining 4 instances from our dataset.

One pass through the training dataset is called an epoch.
