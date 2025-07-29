import numpy as np
import matplotlib.pyplot as plt
import random
from scipy.spatial.transform import Rotation as R

def straight_line(start, n_steps, step=1.0):
    # Sample a random unit direction vector uniformly on the sphere
    vec = np.random.normal(size=3)
    vec /= np.linalg.norm(vec)

    # Generate trajectory along this direction
    x = start[0] + step * np.arange(n_steps) * vec[0]
    y = start[1] + step * np.arange(n_steps) * vec[1]
    z = start[2] + step * np.arange(n_steps) * vec[2]
    return x, y, z


def helix(start, n_steps, radius=5, pitch=1.0, clockwise=True, step=1.0):
    # 1. Generate standard z-axis helix
    theta = np.linspace(0, 2 * np.pi * n_steps * step / (2 * np.pi * radius), n_steps)
    theta = -theta if clockwise else theta
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    z = pitch * theta / (2 * np.pi)
    helix_coords = np.stack((x, y, z), axis=1)

    # 2. Generate a random unit vector (axis of the helix)
    rand_axis = np.random.normal(size=3)
    rand_axis /= np.linalg.norm(rand_axis)

    # 3. Compute rotation from z-axis to rand_axis
    z_axis = np.array([0, 0, 1])
    if np.allclose(rand_axis, z_axis):
        R_align = np.eye(3)
    elif np.allclose(rand_axis, -z_axis):
        R_align = R.from_rotvec(np.pi * np.array([1, 0, 0])).as_matrix()
    else:
        rot_axis = np.cross(z_axis, rand_axis)
        rot_angle = np.arccos(np.dot(z_axis, rand_axis))
        R_align = R.from_rotvec(rot_angle * rot_axis / np.linalg.norm(rot_axis)).as_matrix()

    # 4. Rotate the helix
    helix_rotated = helix_coords @ R_align.T

    # 5. Translate to start position
    helix_translated = helix_rotated + np.array(start)

    return helix_translated[:,0],helix_translated[:,1],helix_translated[:,2]

def cast_and_surge(start, n_steps, surge_axis=None, A0=5.0, decay=0.05, freq=0.5, step=1.0):
    """
    Generates an oscillating cast-and-surge movement.
    - start: starting point (3,)
    - n_steps: number of steps
    - surge_axis: direction of forward motion (default random)
    - A0: initial amplitude of oscillation
    - decay: exponential decay rate of oscillation amplitude
    - freq: angular frequency of oscillation
    - step: forward displacement per time step
    """
    # Forward direction (surge)
    if surge_axis is None:
        surge_axis = np.random.randn(3)
        surge_axis /= np.linalg.norm(surge_axis)
    else:
        surge_axis = np.array(surge_axis) / np.linalg.norm(surge_axis)

    # Oscillation direction (orthogonal to surge_axis)
    tmp = np.random.randn(3)
    cast_axis = np.cross(surge_axis, tmp)
    cast_axis /= np.linalg.norm(cast_axis)

    # Time array
    t = np.arange(n_steps)

    # Amplitude decay
    amp = A0 * np.exp(-decay * t)

    # Surge and cast components
    surge_disp = step * t[:, None] * surge_axis
    cast_disp = (amp * np.sin(freq * t))[:, None] * cast_axis

    # Combine and add start offset
    traj = start + surge_disp + cast_disp

    return traj[:, 0], traj[:, 1], traj[:, 2]

def persistent_random_walk(start, n_steps, step_size=1.0, turn_std=np.pi / 10):
    """
    Generate a 3D persistent random walk with constant step size.
    - start: initial 3D position
    - n_steps: number of steps
    - step_size: fixed length of each step
    - turn_std: standard deviation (radians) for angular deviation between steps
    """
    traj = np.zeros((n_steps, 3))
    traj[0] = start

    # Initial direction
    direction = np.random.randn(3)
    direction /= np.linalg.norm(direction)

    for t in range(1, n_steps):
        # Small random rotation around a random axis
        axis = np.random.randn(3)
        axis /= np.linalg.norm(axis)
        angle = np.random.normal(loc=0.0, scale=turn_std)
        rot = R.from_rotvec(angle * axis)
        direction = rot.apply(direction)
        direction /= np.linalg.norm(direction)

        traj[t] = traj[t-1] + step_size * direction

    x, y, z = traj[:, 0], traj[:, 1], traj[:, 2]
    return x, y, z
    


def generate_random_trajectory(total_steps, step_size=1.0,
                                  mvt_types=['straight', 'helix'],
                                  mvt_args={}):

    move_functions = {
        'straight': straight_line,
        'cast_and_surge': cast_and_surge,
        'helix': helix,
        'persistent_random_walk': persistent_random_walk
    }

    remaining_steps = total_steps
    segments = []
    move_names = []
    last_point = (0.0, 0.0, 0.0)

    while remaining_steps > 0:
        move = random.choice(mvt_types)        

        # Merge base arguments with override from mvt_args
        kwargs = mvt_args.get(move, {}).copy()
        kwargs['start'] = last_point

        kwargs['n_steps'] = min(mvt_args.get(move, {}).get('n_steps',100),remaining_steps)

        if move in move_functions:
            x, y, z = move_functions[move](**kwargs)
        else:
            raise ValueError(f"Unknown movement type: {move}")

        segments.append(np.vstack([x, y, z]))
        move_names.extend([move] * kwargs['n_steps'])
        last_point = (x[-1], y[-1], z[-1])
        remaining_steps -= kwargs['n_steps']

    full_traj = np.hstack(segments)
    return full_traj.T, move_names