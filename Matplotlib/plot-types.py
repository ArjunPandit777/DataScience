# different types of charts in matplotlib

# line plot
# x = [1, 2, 3, 4, 5]
# y = [10, 20, 30, 40, 50]
# plt.plot(x, y)
# plt.show()

# scatter plot
# x = [1, 2, 3, 4, 5]
# y = [10, 20, 30, 40, 50]
# plt.scatter(x, y)
# plt.show()

# bar plot
# x = [1, 2, 3, 4, 5,6,7,8,9,10]
# y = [10, 20, 30, 40, 50,60,70,80,90,100]
# plt.bar(x, y)
# plt.show()

# histogram
import matplotlib.pyplot as plt
import numpy as np

plt.style.use('_mpl-gallery')

# Make data
n = 20
x = np.sin(np.linspace(0, 2*np.pi, n))
y = np.cos(np.linspace(0, 2*np.pi, n))
z = np.linspace(0, 1, n)

# Plot
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
ax.stem(x, y, z)

ax.set(xticklabels=[],
       yticklabels=[],
       zticklabels=[])

plt.show()


# plt.figure()
plt.show()
## Add some margin
# l, r, b, t = plt.axis()
# dx, dy = r-l, t-b
# plt.axis([l-0.1*dx, r+0.1*dx, b-0.1*dy, t+0.1*dy])


# new added things in graph
def plot_quiver_singularities(min_points, max_points, vector_field_x, vector_field_y, file_path):
    """
    Plot the singularities of vector field
    :param file_path : the path to save the data
    :param vector_field_x : the vector field x component to be plot
    :param vector_field_y : the vector field y component to be plot
    :param min_points : a set (x, y) of min points field
    :param max_points : a set (x, y) of max points  field
    """
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_axes([.13, .3, .6, .6])
    
    ## Plot quiver
    x, y = np.mgrid[-1:1:100*1j, -1:1:100*1j]
    m = np.sqrt(np.power(vector_field_x, 2) + np.power(vector_field_y, 2))
    quiver = ax.quiver(x, y, vector_field_x, vector_field_y, m, zorder=1)
    
    ## Plot critical points
    x = np.linspace(-1, 1, x_steps)
    y = np.linspace(-1, 1, y_steps)
    
    # Draw the min points
    x_indices = np.nonzero(min_points)[0]
    y_indices = np.nonzero(min_points)[1]
    ax.scatter(x[x_indices], y[y_indices], marker='$\\circlearrowright$', s=100, zorder=2)
    
    # Draw the max points
    x_indices = np.nonzero(max_points)[0]
    y_indices = np.nonzero(max_points)[1]
    ax.scatter(x[x_indices], y[y_indices], marker='$\\circlearrowleft$', s=100, zorder=2)
    
    ## Put legends
    marker_min = plt.Line2D((0, 0), (0, 0), markeredgecolor=(1.0, 0.4, 0.0), linestyle='',
                            marker='$\\circlearrowright$', markeredgewidth=1, markersize=10)
    marker_max = plt.Line2D((0, 0), (0, 0), markeredgecolor=(0.2, 0.2, 1.0), linestyle='',
                            marker='$\\circlearrowleft$', markeredgewidth=1, markersize=10)
    plt.legend([marker_min, marker_max], ['CW rot. center', 'CCW rot. center'], numpoints=1,
               loc='center left', bbox_to_anchor=(1, 0.5))
    
    quiver_cax = fig.add_axes([.13, .2, .6, .03])
    fig.colorbar(quiver, orientation='horizontal', cax=quiver_cax)
    
    ## Set axis limits
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    
    ## Add some margin
    # l, r, b, t = plt.axis()
    # dx, dy = r-l, t-b
    # plt.axis([l-0.1*dx, r+0.1*dx, b-0.1*dy, t+0.1*dy])
    
    plt.savefig(file_path + '.png', dpi=dpi)
    plt.close()