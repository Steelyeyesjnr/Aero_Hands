# display
WINDOW_WIDTH = 1024
WINDOW_HEIGHT = 512
FPS = 60

# physics
GRID_RES_X = 256
GRID_RES_Y = 128

SCALE_FACTOR = WINDOW_WIDTH / GRID_RES_X

# simulation
DT = 0.02 # time step
JACOBI_ITERS = 60 # number of iters for pressure solver
WIND_SPEED = 200 # base speed of fluid

# interaction
PINCH_THRESHOLD = 0.05
DRAW_COLOR = (0, 255, 0)
UI_COLOR = (255, 255, 255)