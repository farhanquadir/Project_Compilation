import pandas as pd
import matplotlib.pyplot as plt

def predict(_a, _b, _c, array):
    result = []
    for x in array:
        val = float(_a + _b * x + _c * (x ** 2))
        result.append(val)
    return result

def gradient_descent_runner(_a, _b, _c, _x_array, _y_array):

    a = _a
    b = _b
    c = _c

    for x in range(0, ITERATION):
        a, b, c = gradient_descent_step(a, b, c, _x_array, _y_array)
    return [a, b, c]


def gradient_descent_step(_a, _b, _c, _x_array, _y_array):
    a_gradient = 0
    b_gradient = 0
    c_gradient = 0

    i = 0
    length = len(_x_array)
    # find gradient of the cost function
    for x in _x_array:
        cost = (-float(_a + _b * x + _c * x ** 2) + float(_y_array[i]))
        #print('cost ' + str(cost))
        cost = cost*cost
        #print('cost '+str(cost))
        a_gradient += (2 / length) * cost * (-1)
        b_gradient += (2 / length) * cost * (-x)
        c_gradient += (2 / length) * cost * (-2 * x)

        i +=1
    
    new_a = _a - LEARNING_RATE * a_gradient
    new_b = _b - LEARNING_RATE * b_gradient
    new_c = _c - LEARNING_RATE * c_gradient
    return [new_a, new_b, new_c]

def accuracy(_a, _b, _c, _x_array, _y_array):
    i = 0
    total_accuracy = 0
    for x in _x_array:
        val = (float(_a + _b * x + _c * (x ** 2))) / _y_array[i]
        per =(1 -(val/_y_array[i]))
        
        total_accuracy += per
        i = i + 1
    return abs(100*total_accuracy/len(_x_array))


datas = pd.read_csv('XY.txt', delimiter=',')
data_array = datas
x = datas.iloc[:, 0].values
y = datas.iloc[:, 1].values
#print(x)
#print(y)

LEARNING_RATE = 0.0001
ITERATION = 1000
initial_a = 0
initial_b = 0
initial_c = 0

a, b, c = 0, 0, 0
a, b, c = gradient_descent_runner(initial_a, initial_b, initial_c, x, y)

print(accuracy(a, b, c, x, y))

plt.scatter(x, y, color='green')

plt.plot(x, predict(a, b, c, x), color='blue')

plt.title('Height vs Weight Plot (Assignment 3 part 1)')
plt.ylabel('Height')
plt.xlabel('Weight')

plt.show()
