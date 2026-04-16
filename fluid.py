# fluid.py
import taichi as ti
import numpy as np
import config

# Initialize Taichi
ti.init(arch=ti.cpu) 

@ti.data_oriented
class FluidSolver:
    def __init__(self, res_x, res_y):
        self.res_x = res_x
        self.res_y = res_y
        
        # Physics Fields
        self.velocity = ti.Vector.field(2, dtype=float, shape=(res_x, res_y))
        self.new_velocity = ti.Vector.field(2, dtype=float, shape=(res_x, res_y))
        self.pressure = ti.field(dtype=float, shape=(res_x, res_y))
        self.new_pressure = ti.field(dtype=float, shape=(res_x, res_y))
        self.divergence = ti.field(dtype=float, shape=(res_x, res_y))
        
        # Visualization Fields
        self.density = ti.field(dtype=float, shape=(res_x, res_y))
        self.new_density = ti.field(dtype=float, shape=(res_x, res_y))
        
        # Obstacle Map
        self.obstacle_mask = ti.field(dtype=float, shape=(res_x, res_y))

        # Stats Mailbox (Stores Lift and Drag results)
        self.force_results = ti.Vector.field(2, dtype=float, shape=())

    @ti.kernel
    def advect_fields(self, dt: float):
        for i, j in self.velocity:
            # Backtrace where the fluid came from
            p = ti.Vector([float(i), float(j)]) - self.velocity[i, j] * dt
            self.new_velocity[i, j] = self.sample_bilinear(self.velocity, p)
            # 0.99 multiplier prevents "smoke artifacts" from living forever
            self.new_density[i, j] = self.sample_bilinear(self.density, p) * 0.99

    @ti.kernel
    def copy_back_fields(self):
        for i, j in self.velocity:
            self.velocity[i, j] = self.new_velocity[i, j]
            self.density[i, j] = self.new_density[i, j]

    def advect(self, dt: float):
        self.advect_fields(dt)
        self.copy_back_fields()

    @ti.kernel
    def apply_external_forces(self, dt: float):
        for i, j in self.velocity:
            # THE INLET: Constant wind speed from the left
            if i < 3:
                self.velocity[i, j] = ti.Vector([config.WIND_SPEED, 0.0])
                self.density[i, j] = 1.0
            
            # THE STABILIZER: Clamping to prevent numerical explosions
            if self.velocity[i, j].norm() > 1000.0:
                self.velocity[i, j] = self.velocity[i, j].normalized() * 1000.0

            # Walls: Zero out velocity inside obstacles
            if self.obstacle_mask[i, j] > 0.5:
                self.velocity[i, j] = ti.Vector([0, 0])

    @ti.kernel
    def compute_divergence(self):
        for i, j in self.velocity:
            v_l = self.velocity[i-1, j][0] if i > 0 else 0.0
            v_r = self.velocity[i+1, j][0] if i < self.res_x-1 else 0.0
            v_d = self.velocity[i, j-1][1] if j > 0 else 0.0
            v_u = self.velocity[i, j+1][1] if j < self.res_y-1 else 0.0
            self.divergence[i, j] = (v_r - v_l + v_u - v_d) * 0.5

    @ti.kernel
    def pressure_iteration(self):
        for i, j in self.pressure:
            p_l = self.pressure[i-1, j] if i > 0 else 0.0
            p_r = self.pressure[i+1, j] if i < self.res_x-1 else 0.0
            p_d = self.pressure[i, j-1] if j > 0 else 0.0
            p_u = self.pressure[i, j+1] if j < self.res_y-1 else 0.0
            self.new_pressure[i, j] = (p_l + p_r + p_u + p_d - self.divergence[i, j]) * 0.25
        
        for i, j in self.pressure:
            self.pressure[i, j] = self.new_pressure[i, j]

    def solve_pressure(self):
        self.compute_divergence()
        for _ in range(config.JACOBI_ITERS):
            self.pressure_iteration()

    @ti.kernel
    def project(self):
        for i, j in self.velocity:
            if self.obstacle_mask[i, j] < 0.5:
                p_l = self.pressure[i-1, j] if i > 0 else 0.0
                p_r = self.pressure[i+1, j] if i < self.res_x-1 else 0.0
                p_d = self.pressure[i, j-1] if j > 0 else 0.0
                p_u = self.pressure[i, j+1] if j < self.res_y-1 else 0.0
                self.velocity[i, j] -= ti.Vector([p_r - p_l, p_u - p_d]) * 0.5

    @ti.kernel
    def compute_forces_kernel(self):
        self.force_results[None] = ti.Vector([0.0, 0.0])
        for i, j in self.pressure:
            if self.obstacle_mask[i, j] > 0.5:
                p_l = self.pressure[i-1, j] if i > 0 else self.pressure[i, j]
                p_r = self.pressure[i+1, j] if i < self.res_x-1 else self.pressure[i, j]
                p_d = self.pressure[i, j-1] if j > 0 else self.pressure[i, j]
                p_u = self.pressure[i, j+1] if j < self.res_y-1 else self.pressure[i, j]
                
                # Atomic addition into the mailbox
                self.force_results[None][0] += (p_l - p_r) # Drag
                self.force_results[None][1] += (p_d - p_u) # Lift

    def get_stats(self):
        self.compute_forces_kernel()
        forces = self.force_results[None]
        
        ref_chord = self.res_x * 0.25
        K = 2500.0 / (config.WIND_SPEED**2) 

        # FLIP THE SIGN: -forces[1] ensures that pushing UP is POSITIVE lift
        cl = (-forces[1] * K) / ref_chord
        cd = (forces[0] * K) / ref_chord
        
        if abs(cl) < 0.001: cl = 0.0
        if abs(cd) < 0.01: cd = 0.01

        l_d = cl / cd # Removed abs() so you can see negative lift if you dive
        
        return cl, cd, l_d

    @ti.func
    def sample_bilinear(self, field, p):
        p[0] = ti.max(0.5, ti.min(float(self.res_x) - 1.5, p[0]))
        p[1] = ti.max(0.5, ti.min(float(self.res_y) - 1.5, p[1]))
        i, j = int(p[0]), int(p[1])
        f, g = p[0] - float(i), p[1] - float(j)
        return (1-f)*(1-g)*field[i, j] + f*(1-g)*field[i+1, j] + \
               (1-f)*g*field[i, j+1] + f*g*field[i+1, j+1]

    def step(self):
        dt = config.DT
        self.advect(dt)
        self.apply_external_forces(dt)
        self.solve_pressure()
        self.project()

    def update_obstacles(self, mask_np):
        self.obstacle_mask.from_numpy(mask_np)

    def get_render_data(self):
        return self.density.to_numpy()

    @ti.kernel
    def reset_fields(self):
        for i, j in self.velocity:
            self.velocity[i, j] = ti.Vector([0, 0])
            self.pressure[i, j] = 0.0
            self.density[i, j] = 0.0
            self.obstacle_mask[i, j] = 0.0

    def reset_simulation(self):
        self.reset_fields()