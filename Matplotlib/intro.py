import matplotlib.pyplot as plt
import numpy as np

# pyplot is a collection of command style functions that make matplotlib work like MATLAB
# create a simple line plot

x = np.array([1, 2, 3, 4])
y = np.array([12, 25, 18, 20])

x1 = np.array([1, 2, 3, 4])
y1 = np.array([9, 35, 28, 25])

plt.figure(figsize=(10, 10))

plt.plot(x, y,color='red',linestyle='--',linewidth=4,marker='o',markersize=12,markerfacecolor='yellow',markeredgecolor='green',markeredgewidth=2,alpha=0.5)

plt.plot(x1, y1,color='blue',linestyle='--',linewidth=4,marker='o',markersize=12,markerfacecolor='yellow',markeredgecolor='green',markeredgewidth=2,alpha=0.5,label='Line 2',zorder=2,clip_on=True,drawstyle='steps-post')

plt.legend(loc='upper left',fontsize=12)

plt.xlabel('X Values')
plt.xlim(0,5)
plt.ylim(0,40)
plt.ylabel('Y Values')
plt.title('Line Plot')
plt.show()

# other types of plots
# market options in marker paramter in plot function or scatter functionn
# plt.plot(x, y, 'o')

"""
# Marker styles:
'.' - point
'o' - circle
's' - square
'^' - triangle up
'v' - triangle down
'<' - triangle left
'>' - triangle right
'p' - pentagon
'*' - star
'h' - hexagon1
'H' - hexagon2
'+' - plus
'x' - x
'D' - diamond
'd' - thin diamond
'|' - vertical line
'_' - horizontal line
"""

"""
# Line styles:
'-' - solid line
'--' - dashed line
'-.' - dash-dot line
':' - dotted line
"""


# scatter plot
# plt.scatter(x, y,color='red',marker='s',s=100)
# plt.xlabel('X Values')
# plt.ylabel('Y Values')
# plt.title('Scatter Plot')
# plt.show()
