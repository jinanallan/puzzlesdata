import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Generate example trajectory data
num_points = 100
x = np.linspace(0, 10, num_points)
y = np.sin(x)

# Create a figure and axis
fig, ax = plt.subplots()
ax.set_xlim(0, 10)
ax.set_ylim(-1.5, 1.5)

# Create an empty plot object (initialization of the animation)
line, = ax.plot([], [], lw=2)

# Function to initialize the animation
def init():
    line.set_data([], [])
    return line,

# Function to update the animation
def update(frame):
    line.set_data(x[:frame], y[:frame])
    return line,

# Create the animation
ani = FuncAnimation(fig, update, frames=num_points, init_func=init, blit=True)

# Show the animation
plt.show()
