import numpy as np
import matplotlib.pyplot as plt

CONSTANT_OF_GRAVITY = 6.67e-11

class Object:
    def __init__(self, mass: float, position: np.ndarray, velocity: np.ndarray, radius: float):
        self.mass = mass
        self.position = position
        self.velocity = velocity
        self.radius = radius

    def get_momentum(self):
        return self.mass * self.velocity
    

def main():
    sun = Object(1.9891e30, np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0]), 695700.0)
    earth = Object(5.972e24, np.array([0.0, 152e9, 0.0]), np.array([-29290, 0.0, 0.0]), 6378.137)

#     mu = sun.mass*earth.mass/(sun.mass+earth.mass)

#     l_vec = np.cross(sun.position, sun.get_momentum()) + np.cross(earth.position, earth.get_momentum())
#     l = np.linalg.norm(l_vec)

    fig = plt.figure(figsize=(3,3), facecolor='black')
    fig.canvas.toolbar.set_message = lambda x: ""
    ax = fig.add_subplot(projection='3d')
    ax.set_aspect('equal')
    ax.set_facecolor('black')
    ax.xaxis.set_pane_color((1,1,1,0))
    ax.yaxis.set_pane_color((1,1,1,0))
    ax.zaxis.set_pane_color((1,1,1,0))
    for spine in ax.spines.values():
        spine.set_color('gray')
    ax.tick_params(axis='both', colors='gray')

    ax.scatter(0, 0, 0, c='yellow', s=1000)

    positions = []

    t = 0
    dt = 500
    while t < 3.154e7:
        positions.append(earth.position.copy())
        r = earth.position - sun.position

        r_norm = (1 / np.linalg.norm(r)) * r

        acc = -CONSTANT_OF_GRAVITY*sun.mass*r_norm / (np.linalg.norm(r))**2
        earth.velocity += acc*dt
        earth.position += earth.velocity*dt
        t += dt

    peri = int(len(positions) / 2)
    positions = np.array(positions)
    ax.plot(positions[:,0], positions[:,1], positions[:,2], c='blue')
    ax.scatter(positions[-1,0], positions[-1,1], positions[:,2], c='blue', s=100)
    print(f"aphelion: {int(positions[0,1]*-1000)} kms")
    print(f"perihelion: {int(positions[peri,1]*-1000)} kms")
    plt.show()



if __name__ == "__main__":
    main()