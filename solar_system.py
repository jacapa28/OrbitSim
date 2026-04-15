import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from math import pi

CONSTANT_OF_GRAVITY = 4*(pi**2)

class Body:
    def __init__(self, mass: float, position: np.ndarray, velocity: np.ndarray, radius: float, color: str):
        self.mass = mass
        self.position = position
        self.velocity = velocity
        self.radius = radius
        self.color = color


def make_state(bodies: list[Body]):
    state = [np.zeros(3) for i in range(len(bodies)*2)]
    for index, body in enumerate(bodies):
        state[index*2] = body.position
        state[index*2+1] = body.velocity
    return np.concatenate(state)

def update_from_state(state: np.ndarray, bodies: list[Body]):
    for index, body in enumerate(bodies):
        location = index * 6
        body.position = state[location:location+3].copy()
        body.velocity = state[location+3:location+6].copy()


def make_derivatives_state(state: np.ndarray, bodies: list[Body]):
    n = len(bodies)

    positions = [state[i*6:i*6+3] for i in range(n)]
    velocities = [state[i*6+3:i*6+6] for i in range(n)]

    derivatives = []

    for i in range(n):
        derivatives.append(velocities[i])

        acc = np.zeros(3)
        for j in range(n):
            if i == j:
                continue

            r = positions[j] - positions[i]
            r_mag = np.linalg.norm(r)
            acc += CONSTANT_OF_GRAVITY*bodies[j].mass*r / (r_mag)**3

        derivatives.append(acc)

    return np.concatenate(derivatives)


def RK4(state: np.ndarray, bodies: list[Body], dt: float):
    k1 = make_derivatives_state(state, bodies)
    k2 = make_derivatives_state(state + 0.5*dt*k1, bodies)
    k3 = make_derivatives_state(state + 0.5*dt*k2, bodies)
    k4 = make_derivatives_state(state + dt*k3, bodies)

    return state + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)


def set_axis_scale(ax: Axes, positions: np.ndarray):
    all_x = positions[:, 0::3].flatten()
    all_y = positions[:, 1::3].flatten()
    all_z = positions[:, 2::3].flatten()

    x_max = all_x.max()
    x_min = all_x.min()
    y_max = all_y.max()
    y_min = all_y.min()
    z_max = all_z.max()
    z_min = all_z.min()
    
    max_range = np.array([
        x_max - x_min,
        y_max - y_min,
        z_max - z_min
    ]).max() / 2

    x_mid = (x_min + x_max) / 2
    y_mid = (y_min + y_max) / 2
    z_mid = (z_min + z_max) / 2

    ax.set_xlim(x_mid - max_range, x_mid + max_range)
    ax.set_ylim(y_mid - max_range, y_mid + max_range)
    ax.set_zlim((z_mid - max_range) / 4, (z_mid + max_range) / 4)


def main():
    sol = Body(
        1, 
        np.array([-0.004984, -0.005223, 0.000167]), 
        np.array([0.002649, -0.000898, -0.000045]), 
        695700, 
        "gold"
    )
    mercury = Body(
        1.66014e-7, 
        np.array([-0.182422, -0.434562, -0.018645]), 
        np.array([7.427865, -3.419124, -0.960439]), 
        2439.7, 
        "darkgoldenrod"
    )
    venus = Body(
        2.447837e-6, 
        np.array([-0.576805, -0.446787, 0.027095]), 
        np.array([4.462737, -5.878881, -0.338143]), 
        6051.8, 
        "orange"
    )
    earth = Body(
        3.003496e-6, 
        np.array([-0.921514, -0.412320, 0.000198]), 
        np.array([2.448452, -5.761284, 0.000461]), 
        6371, 
        "deepskyblue"
    )
    mars = Body(
        3.227157e-7, 
        np.array([-1.512798, 0.701485, 0.051952]), 
        np.array([-1.973634, -4.189456, -0.039359]), 
        3389.5, 
        "orangered"
    )
    jupiter = Body(
        9.545940e-4, 
        np.array([0.271540, 5.104690, -0.027246]), 
        np.array([-2.782220, 0.277527, 0.061104]), 
        69911, 
        "bisque"
    )
    saturn = Body(
        2.858149e-4,
        np.array([9.512388, -1.203112, -0.357819]),
        np.array([0.142533, 2.016153, -0.040854]),
        58232,
        "cornsilk"
    )
    uranus = Body(
        4.365802e-5,
        np.array([10.759127, 16.297305, -0.078859]),
        np.array([-1.208608, 0.724013, 0.018360]),
        25362,
        "powderblue"
    )
    neptune = Body(
        5.150331e-5,
        np.array([29.877219, -0.313894, -0.682087]),
        np.array([0.004452, 1.152644, -0.023697]),
        24622,
        "blue"
    )
    luna = Body(
        3.695203e-8,
        np.array([-0.923746, -0.413860, 0.000047]),
        np.array([2.565240, -5.928905, -0.014254]),
        1737.4,
        "whitesmoke"
    )
    phobos = Body(
        5.381168e-15,
        np.array([-1.512765, 0.701536, 0.051938]),
        np.array([-2.298568, -3.929212, 0.138640]),
        11.1,
        "tan"
    )
    deimos = Body(
        7.422994e-15,
        np.array([-1.512896, 0.701368, 0.051987]),
        np.array([-1.782717, -4.376669, -0.137477]),
        6.2,
        "burlywood"
    )

    bodies = [sol, luna, earth]
    k = len(bodies)

    fig = plt.figure(figsize=(6,3), facecolor='black')
    fig.canvas.toolbar.set_message = lambda x: ""
    o_ax = fig.add_subplot(projection='3d')
    o_ax.set_aspect('equal')
    o_ax.set_facecolor('black')
    o_ax.yaxis.set_pane_color((1,1,1,0))
    o_ax.zaxis.set_pane_color((1,1,1,0))
    o_ax.xaxis.set_pane_color((1,1,1,0))
    for spine in o_ax.spines.values():
        spine.set_color('gray')
    o_ax.tick_params(axis='both', colors='gray')

    t = 0
    dt = 0.0002
    nsteps = int(1 / dt)
    times = np.zeros(nsteps+1)
    positions = np.zeros((nsteps+1, 3*k))

    times[0] = t
    current_positions = []
    for body in bodies:
        current_positions.append(body.position.copy())
    positions[0] = np.concatenate(current_positions)

    state = make_state(bodies)

    for i in range(1, nsteps+1):
        t += dt
        times[i] = t

        state = RK4(state, bodies, dt)
        update_from_state(state, bodies)

        current_positions = []
        for body in bodies:
            current_positions.append(body.position.copy())
        positions[i] = np.concatenate(current_positions)

    for index, body in enumerate(bodies):
        o_ax.plot(
            positions[:,index*3], 
            positions[:,index*3+1], 
            positions[:,index*3+2], 
            c=body.color
        )
        o_ax.scatter(
            positions[-1,index*3], 
            positions[-1,index*3+1], 
            positions[-1,index*3+2], 
            c=body.color, 
            s=body.radius/10000
        )
        print(f"{body.color}:  X = {positions[-1,index*3]*149597870.7} Y = {positions[-1,index*3+1]*149597870.7} Z = {positions[-1,index*3+2]*149597870.7}\n")

    set_axis_scale(o_ax, positions)
    plt.show()


if __name__ == "__main__":
    main()