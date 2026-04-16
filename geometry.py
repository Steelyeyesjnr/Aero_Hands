import numpy as np
import cv2

class GeometryManager:
    def __init__(self):
        # We store everything as float32 to prevent 'geometry wrecking' during rotation
        self.objects = [] 
        self.current_object = []
        self.is_drawing = False

    def start_drawing(self):
        self.is_drawing = True
        self.current_object = []
    
    def scale_object(self, factor):
        """Scales the object relative to its bounding box center."""
        if not self.objects: return
        
        obj = self.objects[-1]
        
        # 1. Find the center
        min_xy = np.min(obj, axis=0)
        max_xy = np.max(obj, axis=0)
        center = (min_xy + max_xy) / 2.0
        
        # Scale relative to center
        self.objects[-1] = (obj - center) * factor + center
    
    def translate_object(self, dx, dy):
        """Moves the object by a relative offset (dx, dy)."""
        if not self.objects: return
        
        # Shift every point in the object by the hand's displacement
        self.objects[-1] += np.array([dx, dy], dtype=np.float32)

    def stop_drawing(self):
        self.is_drawing = False
        if len(self.current_object) > 2:
            # Save the final shape as a high-precision numpy array
            self.objects.append(np.array(self.current_object, dtype=np.float32))
        self.current_object = []

    def clear_all(self):
        self.objects = []
        self.current_object = []

    def add_point(self, x, y):
        if not self.is_drawing: return
        
        new_pt = [float(x), float(y)]
        
        # Distance Filter: Prevents jagged lines from hand tremors
        if len(self.current_object) > 0:
            last_pt = self.current_object[-1]
            dist = np.sqrt((new_pt[0] - last_pt[0])**2 + (new_pt[1] - last_pt[1])**2)
            if dist < 8: return # Only add point if moved at least 8 pixels
            
        self.current_object.append(new_pt)

    def rotate_object(self, angle_degrees):
        """Rotates the object while maintaining high floating-point precision."""
        if not self.objects: return
            
        obj = self.objects[-1]
        
        # STABLE PIVOT: Calculate center based on the 'box' around the object
        # keeps the wing from 'walking' or drifting away
        min_xy = np.min(obj, axis=0)
        max_xy = np.max(obj, axis=0)
        center = (min_xy + max_xy) / 2.0
        
        theta = np.radians(angle_degrees)
        c, s = np.cos(theta), np.sin(theta)
        R = np.array(((c, -s), (s, c)))
        
        # The Math: Move to origin (0,0), rotate, then move back
        self.objects[-1] = (obj - center) @ R.T + center

    def load_airfoil_preset(self, frame_w, frame_h, type_idx, aoa=0):
        self.clear_all()
        x = np.linspace(0, 1, 61)
        
        # NACA 4-Digit Parameters: [Camber, Camber Position, Thickness]
        presets = {
            1: [0.0, 0.0, 0.12],  # NACA 0012: Symmetric
            2: [0.04, 0.4, 0.12], # NACA 4412: High Camber
            3: [0.02, 0.4, 0.06], # NACA 2406: Thin
            4: [0.0, 0.0, 0.02],  # Flat Plate
            5: [0.06, 0.3, 0.15]  # Heavy Lift
        }
        
        m, p, t = presets.get(type_idx, presets[1])
        
        # 1. Thickness and Camber Math
        yt = 5 * t * (0.2969*np.sqrt(x) - 0.1260*x - 0.3516*x**2 + 0.2843*x**3 - 0.1015*x**4)
        yc = np.zeros_like(x)
        if p > 0:
            mask = x <= p
            yc[mask] = (m / p**2) * (2*p*x[mask] - x[mask]**2)
            yc[~mask] = (m / (1-p)**2) * ((1-2*p) + 2*p*x[~mask] - x[~mask]**2)

        # 2. Combine and FLIP Y-AXIS
        # We negate the Y values here to account for OpenCV's inverted Y-axis
        upper = np.stack([x, -(yc + yt)], axis=1) 
        lower = np.stack([x[::-1], -(yc - yt)[::-1]], axis=1)
        airfoil = np.vstack([upper, lower])
        
        # 3. Scale and Position
        scale = frame_w * 0.25
        offset_x, offset_y = frame_w * 0.3, frame_h * 0.5
        
        # Rotation
        theta = np.radians(aoa)
        R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        
        final_pts = (airfoil @ R.T * scale) + [offset_x, offset_y]
        self.objects.append(final_pts.astype(np.float32))

    def get_obstacle_mask(self, res_x, res_y, frame_w, frame_h):
        """Converts high-precision drawings into a binary mask for the fluid solver."""
        mask = np.zeros((res_y, res_x), dtype=np.float32)
        scale_x = res_x / frame_w
        scale_y = res_y / frame_h
        
        for obj in self.objects:
            # Scale coordinates to the grid resolution
            scaled_obj = (obj * [scale_x, scale_y]).astype(np.int32)
            cv2.fillPoly(mask, [scaled_obj], 1.0)
            
        return mask.T # Transpose for Taichi grid [x, y]

    def draw(self, frame):
        """Draws the objects onto the webcam frame using integer casting only for display."""
        for obj in self.objects:
            # CAST TO INT ONLY HERE: This keeps the original 'self.objects' data pristine
            pts = obj.astype(np.int32)
            cv2.polylines(frame, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
            
        if self.is_drawing and len(self.current_object) > 1:
            pts = np.array(self.current_object).astype(np.int32)
            cv2.polylines(frame, [pts], isClosed=False, color=(0, 255, 0), thickness=2)