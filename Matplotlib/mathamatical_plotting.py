import matplotlib.pyplot as plt
import numpy as np


x = np.array([0, 1, 2, 3, 4])
y = 30 * x*x +40 * x+ 20

plt.plot(x, y, 'r',marker='o', markersize=10, label='Linear Equation')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Linear Equation')

plt.legend()
# plt.xlim(0, 5)
# plt.ylim(0, 150)
plt.show()

# y = mx + c # linear equation
# y = ax^2 + bx + c # quadratic equation
# y = ax^3 + bx^2 + cx + d # cubic equation
# y = ax^4 + bx^3 + cx^2 + dx + e # quartic equation
# y = ax^5 + bx^4 + cx^3 + dx^2 + ex + f # quintic equation
# y = ax^6 + bx^5 + cx^4 + dx^3 + ex^2 + fx + g # sextic equation


# polinomial equation
# m ? slope
# c ? y-intercept # cross the graoph kuch to kudh kar le