import numpy as np
from scipy.interpolate import splprep, splev
from scipy.spatial.transform import Rotation as R

def generate_daughter_vectors(parent_direction, branch_angle_rad):
    p_norm = np.linalg.norm(parent_direction)
    if p_norm < 1e-6: 
        return np.array([1., 0, 0]), np.array([0, 1., 0])
    parent_direction = parent_direction / p_norm

    if abs(parent_direction[2]) < 0.9:
        perp_ref = np.array([0.0, 0.0, 1.0])
    else:
        perp_ref = np.array([1.0, 0.0, 0.0])

    v = np.cross(parent_direction, perp_ref)
    v /= np.linalg.norm(v)

    random_plane_angle = np.random.uniform(0, 2 * np.pi)
    rot_plane = R.from_rotvec(parent_direction * random_plane_angle)
    v_rotated = rot_plane.apply(v)

    rot_d1 = R.from_rotvec(v_rotated * branch_angle_rad)
    rot_d2 = R.from_rotvec(v_rotated * -branch_angle_rad)

    return rot_d1.apply(parent_direction), rot_d2.apply(parent_direction)


class Vessel:
    def __init__(self, start_point, end_point, radius, tortuosity_amplitude=0.0):
        self.start_point = np.array(start_point, dtype=np.float64)
        self.end_point = np.array(end_point, dtype=np.float64)
        self.radius = float(radius)
        
        self.vector = self.end_point - self.start_point
        self.length = np.linalg.norm(self.vector)
        
        if self.length < 1e-5:
            self.direction = np.array([0,0,1.0])
            self.length = 0.01
            self.end_point = self.start_point + self.direction * 0.01
        else:
            self.direction = self.vector / self.length

        self.parent = None
        self.strahler_order = None
        
        if tortuosity_amplitude > 0 and self.length > self.radius * 1.5:
            self.centerline_points = self._generate_spline_path(tortuosity_amplitude)
        else:
            self.centerline_points = np.array([self.start_point, self.end_point])

    def _generate_spline_path(self, amplitude):
        min_pitch = max(6.0, self.radius * 2.5)
        max_loops = self.length / min_pitch
        
        if max_loops < 0.5:
            loops = 0.2
        else:
            loops = np.random.uniform(max_loops * 0.5, max_loops)

        points_per_loop = 8
        num_control_points = max(6, int(loops * points_per_loop))
        
        t_vals = np.linspace(0, 1, num_control_points)
        base_points = self.start_point + np.outer(t_vals, self.vector)
        
        if abs(self.direction[2]) < 0.9: 
            ref = np.array([0,0,1])
        else: 
            ref = np.array([1,0,0])
        perp1 = np.cross(self.direction, ref)
        perp1 /= np.linalg.norm(perp1)
        perp2 = np.cross(self.direction, perp1)
        
        noisy_points = base_points.copy()
        
        phase_freq = loops * 2 * np.pi
        random_phase = np.random.uniform(0, 2*np.pi)
        safe_amp = min(amplitude, self.radius * 2.0)
        
        for i in range(1, num_control_points - 1):
            t = t_vals[i]
            env = np.sin(t * np.pi)  
            theta = t * phase_freq + random_phase
            
            jitter_1 = np.random.uniform(-0.1, 0.1) * safe_amp
            jitter_2 = np.random.uniform(-0.1, 0.1) * safe_amp
            
            disp_1 = (np.cos(theta) * safe_amp * env) + jitter_1
            disp_2 = (np.sin(theta) * safe_amp * env) + jitter_2
            
            noisy_points[i] += (disp_1 * perp1 + disp_2 * perp2)

        tck, u = splprep(noisy_points.T, s=0, k=3) 
        num_fine_points = max(20, int(self.length * 4.0))
        u_fine = np.linspace(0, 1, num_fine_points)
        return np.array(splev(u_fine, tck)).T

