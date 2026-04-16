# main.py
import os
import csv
import cv2
import config
from vision import HandTracker
from geometry import GeometryManager
from fluid import FluidSolver
from datetime import datetime
import numpy as np

def main():
    tracker = HandTracker()
    geom = GeometryManager()
    fluid = FluidSolver(config.GRID_RES_X, config.GRID_RES_Y)

    # --- STATE VARIABLES ---
    sim_running = False  
    current_aoa = 5.0 
    prev_left_hand_pos = None
    prev_hand_size = None  
    view_mode = 0  # 0: Smoke, 1: Pressure, 2: Velocity
    modes = ["SMOKE", "PRESSURE", "VELOCITY"]
    
    # Generate a unique log for this session
    log_filename = f"aero_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    log_timer = 0 

    print("System Online.")
    print("Controls: [S] Start/Pause | [L] Log Data | [C] Clear | [1-5] Airfoils | [Q] Quit")

    while True:
        frame, hands_data = tracker.get_hand_positions()
        if frame is None: continue
        h, w, _ = frame.shape

        # 1. KEYBOARD CONTROLS (Captured ONCE for responsiveness)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): break

        # LOGGING
        if key == ord('l'):
            cl, cd, l_d = fluid.get_stats()
            data_row = {
                "Timestamp": datetime.now().strftime("%H:%M:%S"),
                "AoA": round(current_aoa, 2),
                "Cl": round(cl, 4),
                "Cd": round(cd, 4),
                "L_D_Ratio": round(l_d, 4)
            }
            file_exists = os.path.isfile(log_filename)
            with open(log_filename, mode='a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=data_row.keys())
                if not file_exists: writer.writeheader()
                writer.writerow(data_row)
            log_timer = 30 

        if key == ord('v'):
            view_mode = (view_mode + 1) % 3

        # CONTROLS
        if key == ord('c'):
            geom.clear_all()
            fluid.reset_simulation()
            current_aoa = 0.0
            
        if key == ord('p'):
            current_aoa = 5.0
            geom.load_airfoil_preset(w, h, 1, current_aoa)
            fluid.reset_simulation()

        if key == ord('.'): # Fixed back to 'S'
            sim_running = not sim_running

        if ord('1') <= key <= ord('5'):
            idx = int(chr(key))
            current_aoa = 0.0
            geom.load_airfoil_preset(w, h, idx, current_aoa)

        # ROTATION
        if key == 82 or key == ord('w'): # UP
            geom.rotate_object(1.0)
            current_aoa += 1.0
        if key == 84 or key == ord('s'): # DOWN
            geom.rotate_object(-1.0)
            current_aoa -= 1.0

        # 2. PROCESS HANDS
        for hand in hands_data:
            label, landmarks = hand['label'], hand['landmarks']
            
            if label == "Right": # DRAWING
                dist = tracker.get_pinch_distance(landmarks)
                px = (int(landmarks[8][0] * w), int(landmarks[8][1] * h))
                if dist < 0.4: 
                    if not geom.is_drawing: geom.start_drawing()
                    geom.add_point(px[0], px[1])
                else:
                    if geom.is_drawing: geom.stop_drawing()

            if label == "Left": # MOVE & SCALE
                curr_hand_size = np.sqrt((landmarks[0][0]-landmarks[9][0])**2 + (landmarks[0][1]-landmarks[9][1])**2)
                cx, cy = int(landmarks[9][0] * w), int(landmarks[9][1] * h)
                open_dist = np.sqrt((landmarks[4][0]-landmarks[20][0])**2 + (landmarks[4][1]-landmarks[20][1])**2)
                
                if open_dist > 0.4:
                    cv2.circle(frame, (cx, cy), 15, (0, 255, 255), 2)
                    if prev_left_hand_pos is not None:
                        dx, dy = cx - prev_left_hand_pos[0], cy - prev_left_hand_pos[1]
                        scale_f = 1.0 + (curr_hand_size / prev_hand_size - 1.0) * 1.5
                        geom.translate_object(dx, dy)
                        geom.scale_object(scale_f)
                    prev_left_hand_pos, prev_hand_size = (cx, cy), curr_hand_size
                else:
                    prev_left_hand_pos = prev_hand_size = None

        # 3. PHYSICS & RENDER
        geom.draw(frame)
        fluid.update_obstacles(geom.get_obstacle_mask(config.GRID_RES_X, config.GRID_RES_Y, w, h))
        if sim_running: fluid.step()
        
        # --- 4. RENDER ENGINE & DASHBOARD ---
        cl, cd, l_d = fluid.get_stats()
        
        # Decide which data to pull from the solver based on view_mode
        if view_mode == 0:    # SMOKE (Density)
            raw_data = fluid.get_render_data()
            cmap = cv2.COLORMAP_OCEAN
        elif view_mode == 1:  # PRESSURE
            raw_data = fluid.get_heatmap_data(mode="pressure")
            cmap = cv2.COLORMAP_JET
        else:                # VELOCITY
            raw_data = fluid.get_heatmap_data(mode="velocity")
            cmap = cv2.COLORMAP_MAGMA

        # Process the raw data into a displayable heatmap
        # Note: We transpose (.T) because Taichi and OpenCV axes are swapped
        heat_view = (cv2.resize(raw_data.T, (w, h)) * 255).astype(np.uint8)
        flow_color = cv2.applyColorMap(heat_view, cmap)

        # Draw Stats Box
        cv2.rectangle(frame, (10, 10), (240, 155), (40, 40, 40), -1) 
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        f_scale = 0.55  # Reduced from 0.7
        thick = 1       # Reduced from 2 for a cleaner look
        v_space = 25    # Tightened vertical spacing

        # Text elements with adjusted positioning
        cv2.putText(frame, f"Lift (Cl): {cl:2.2f}", (20, 35), font, f_scale, (100, 255, 100), thick)
        cv2.putText(frame, f"Drag (Cd): {cd:2.2f}", (20, 35 + v_space), font, f_scale, (100, 100, 255), thick)
        cv2.putText(frame, f"L/D Ratio: {l_d:2.2f}", (20, 35 + v_space*2), font, f_scale, (255, 255, 100), thick)
        cv2.putText(frame, f"Angle (AoA): {current_aoa:2.1f}", (20, 35 + v_space*3), font, f_scale, (255, 255, 255), thick)
        
        mode_text = ["SMOKE", "PRESSURE", "VELOCITY"][view_mode]
        cv2.putText(frame, f"MODE: {mode_text}", (20, 35 + v_space*4), font, f_scale, (0, 255, 255), thick)

        if log_timer > 0:
            cv2.putText(frame, "DATA LOGGED!", (w//2-100, h-50), font, 1.0, (0, 255, 255), 3)
            log_timer -= 1
            
        # Composite the hand-tracking frame with the physics heatmap
        final_display = cv2.addWeighted(frame, 0.6, flow_color, 0.4, 0)
        cv2.imshow("Aero Hands - Virtual Wind Tunnel", final_display)
    tracker.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()