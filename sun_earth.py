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
    

def make_state(sun: Object, earth: Object):
    return np.concatenate([
        sun.position,
        sun.velocity,
        earth.position, 
        earth.velocity
    ])

def update_state(state: np.ndarray, sun: Object, earth: Object):
    sun.position = state[0:3].copy()
    sun.velocity = state[3:6].copy()
    earth.position = state[6:9].copy()
    earth.velocity = state[9:12].copy()


def make_derivatives_state(state: np.ndarray, sun: Object, earth: Object):
    sun_pos = state[0:3]
    sun_vel = state[3:6]
    earth_pos = state[6:9]
    earth_vel = state[9:12]

    r = earth_pos - sun_pos
    r_mag = np.linalg.norm(r)

    sun_acc = CONSTANT_OF_GRAVITY*earth.mass*r / (r_mag)**3
    earth_acc = -CONSTANT_OF_GRAVITY*sun.mass*r / (r_mag)**3

    return np.concatenate([
        sun_vel,
        sun_acc,
        earth_vel,
        earth_acc
    ])


def RK4(state: np.ndarray, sun: Object, earth: Object, dt: float):
    k1 = make_derivatives_state(state, sun, earth)
    k2 = make_derivatives_state(state + 0.5*dt*k1, sun, earth)
    k3 = make_derivatives_state(state + 0.5*dt*k2, sun, earth)
    k4 = make_derivatives_state(state + dt*k3, sun, earth)

    return state + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)


def calculate_energy(sun: Object, earth: Object):
    r_mag = np.linalg.norm(earth.position - sun.position)
    KE_sun = 0.5*sun.mass*np.linalg.norm(sun.velocity)**2
    KE_earth = 0.5*earth.mass*np.linalg.norm(earth.velocity)**2
    PE = -CONSTANT_OF_GRAVITY*sun.mass*earth.mass/r_mag
    return KE_sun + KE_earth + PE


def main():
    sun_mass = 1
    earth_mass = 3.003e-6
    M = sun_mass + earth_mass
    distance = 1
    sun_pos = -earth_mass*distance/M
    earth_pos = sun_mass*distance/M
    earth_vel = np.array([0.0, 2*math.pi, 0.0])
    sun_vel = -earth_mass/sun_mass * earth_vel

    sun = Object(sun_mass, np.array([sun_pos, 0.0, 0.0]), sun_vel, 0.00465)
    earth = Object(earth_mass, np.array([earth_pos, 0.0, 0.0]), earth_vel, 0.000043)

#     mu = sun.mass*earth.mass/(sun.mass+earth.mass)

#     l_vec = np.cross(sun.position, sun.get_momentum()) + np.cross(earth.position, earth.get_momentum())
#     l = np.linalg.norm(l_vec)

    fig = plt.figure(figsize=(6,3), facecolor='black')
    fig.canvas.toolbar.set_message = lambda x: ""
    o_ax = fig.add_subplot(1, 3, 1, projection='3d')
    o_ax.set_aspect('equal')
    o_ax.set_facecolor('black')
    o_ax.yaxis.set_pane_color((1,1,1,0))
    o_ax.zaxis.set_pane_color((1,1,1,0))
    o_ax.xaxis.set_pane_color((1,1,1,0))
    for spine in o_ax.spines.values():
        spine.set_color('gray')
    o_ax.tick_params(axis='both', colors='gray')

    e_ax = fig.add_subplot(1, 3, 2)
    e_ax.set_facecolor('black')
    for spine in e_ax.spines.values():
        spine.set_color('white')
    e_ax.tick_params(axis='both', colors='white')
    e_ax.set_ylim([-0.001,0.001])

    s_ax = fig.add_subplot(1, 3, 3)
    s_ax.set_facecolor('black')
    for spine in s_ax.spines.values():
        spine.set_color('white')
    s_ax.tick_params(axis='both', colors='white')

    t = 0
    dt = 0.001
    nsteps = int(10 / dt)
    earth_positions = np.zeros((nsteps+1, 3))
    sun_positions = np.zeros((nsteps+1, 3))
    times = np.zeros(nsteps+1)
    speeds = np.zeros(nsteps+1)
    energies = np.zeros(nsteps+1)

    times[0] = t
    earth_positions[0] = earth.position.copy()
    sun_positions[0] = sun.position.copy()
    speeds[0] = np.linalg.norm(earth.velocity)
    energies[0] = calculate_energy(sun, earth)

    state = make_state(sun, earth)

    for i in range(1, nsteps+1):
        t += dt
        times[i] = t

        state = RK4(state, sun, earth, dt)
        update_state(state, sun, earth)

        earth_positions[i] = earth.position.copy()
        sun_positions[i] = sun.position.copy()
        speeds[i] = np.linalg.norm(earth.velocity)
        energies[i] = calculate_energy(sun, earth)


    o_ax.plot(earth_positions[:,0], earth_positions[:,1], earth_positions[:,2], c='blue')
    o_ax.scatter(earth_positions[-1,0], earth_positions[-1,1], earth_positions[-1,2], c='blue', s=100000*earth.radius)
    o_ax.plot(sun_positions[:,0], sun_positions[:,1], sun_positions[:,2], c='yellow')
    o_ax.scatter(sun_positions[-1,0], sun_positions[-1,1], sun_positions[-1,2], c='yellow', s=100000*sun.radius)

    s_ax.plot(times, speeds, c='green')

    e_ax.plot(times, energies - energies[0], c='red')
    plt.show()


if __name__ == "__main__":
    main()