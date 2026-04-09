import numpy as np
import matplotlib.pyplot as plt
import math

CONSTANT_OF_GRAVITY = 4*((math.pi)**2)

class Object:
    def __init__(self, mass: float, position: np.ndarray, velocity: np.ndarray, radius: float):
        self.mass = mass
        self.position = position
        self.velocity = velocity
        self.radius = radius

    def get_momentum(self):
        return self.mass * self.velocity
    

def main():
    sun = Object(1, np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0]), 0.00465)
    earth = Object(3.003e-6, np.array([0.0, 1, 0.0]), np.array([-2*(math.pi), 0.0, 0.0]), 0.000043)

#     mu = sun.mass*earth.mass/(sun.mass+earth.mass)

#     l_vec = np.cross(sun.position, sun.get_momentum()) + np.cross(earth.position, earth.get_momentum())
#     l = np.linalg.norm(l_vec)

    fig = plt.figure(figsize=(6,3), facecolor='black')
    fig.canvas.toolbar.set_message = lambda x: ""
    o_ax = fig.add_subplot(1, 2, 1, projection='3d')
    o_ax.set_aspect('equal')
    o_ax.set_facecolor('black')
    o_ax.yaxis.set_pane_color((1,1,1,0))
    o_ax.zaxis.set_pane_color((1,1,1,0))
    o_ax.xaxis.set_pane_color((1,1,1,0))
    for spine in o_ax.spines.values():
        spine.set_color('gray')
    o_ax.tick_params(axis='both', colors='gray')

    o_ax.scatter(0, 0, 0, c='yellow', s=10000*sun.radius)

    e_ax = fig.add_subplot(1, 2, 2)
    e_ax.set_facecolor('black')
    for spine in e_ax.spines.values():
        spine.set_color('white')
    e_ax.tick_params(axis='both', colors='white')

    positions = []
    energies = []

    t = 0
    dt = 0.001
    while t < 10:
        r = earth.position - sun.position
        r_mag = np.linalg.norm(r)

        KE = 0.5*earth.mass*np.linalg.norm(earth.velocity)**2
        PE = CONSTANT_OF_GRAVITY*sun.mass*earth.mass/r_mag
        e = [t, KE - PE]
        energies.append(e)

        positions.append(earth.position.copy())

        r_norm = r / r_mag
        acc = -CONSTANT_OF_GRAVITY*sun.mass*r_norm / (r_mag)**2
        earth.velocity += acc*dt
        earth.position += earth.velocity*dt
        t += dt

    positions = np.array(positions)
    o_ax.plot(positions[:,0], positions[:,1], positions[:,2], c='blue')
    o_ax.scatter(positions[-1,0], positions[-1,1], positions[:,2], c='blue', s=10000*earth.radius)

    energies = np.array(energies)
    e_ax.plot(energies[:,0], energies[:,1], c='red')
    plt.show()


if __name__ == "__main__":
    main()