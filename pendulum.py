import pygame
import sys
import numpy as np
from numpy.linalg import inv
from math import sin, cos, pi
import pygame_gui

pygame.init()

# Set up screen
w, h = 1366, 768
screen = pygame.display.set_mode((w, h))
pygame.display.set_caption('Double Pendulum Simulation')

# Colors
WHITE = (255, 255, 255)
CYAN = (0, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)

# GUI setup
manager = pygame_gui.UIManager((w, h))
panel_width = 350

inputs = {
    'm1': pygame_gui.elements.UITextEntryLine(relative_rect=pygame.Rect((w - panel_width + 20, 50), (100, 30)), manager=manager),
    'm2': pygame_gui.elements.UITextEntryLine(relative_rect=pygame.Rect((w - panel_width + 20, 100), (100, 30)), manager=manager),
    'l1': pygame_gui.elements.UITextEntryLine(relative_rect=pygame.Rect((w - panel_width + 20, 150), (100, 30)), manager=manager),
    'l2': pygame_gui.elements.UITextEntryLine(relative_rect=pygame.Rect((w - panel_width + 20, 200), (100, 30)), manager=manager),
    'a1': pygame_gui.elements.UITextEntryLine(relative_rect=pygame.Rect((w - panel_width + 20, 250), (100, 30)), manager=manager),
    'a2': pygame_gui.elements.UITextEntryLine(relative_rect=pygame.Rect((w - panel_width + 20, 300), (100, 30)), manager=manager),
}

# Add labels for each input
labels = {
    'm1': pygame_gui.elements.UILabel(relative_rect=pygame.Rect((w - panel_width + 130, 50), (100, 30)), text='Mass 1', manager=manager),
    'm2': pygame_gui.elements.UILabel(relative_rect=pygame.Rect((w - panel_width + 130, 100), (100, 30)), text='Mass 2', manager=manager),
    'l1': pygame_gui.elements.UILabel(relative_rect=pygame.Rect((w - panel_width + 130, 150), (100, 30)), text='Length 1', manager=manager),
    'l2': pygame_gui.elements.UILabel(relative_rect=pygame.Rect((w - panel_width + 130, 200), (100, 30)), text='Length 2', manager=manager),
    'a1': pygame_gui.elements.UILabel(relative_rect=pygame.Rect((w - panel_width + 130, 250), (100, 30)), text='Angle 1', manager=manager),
    'a2': pygame_gui.elements.UILabel(relative_rect=pygame.Rect((w - panel_width + 130, 300), (100, 30)), text='Angle 2', manager=manager),
}

# Font for time display
font = pygame.font.SysFont(None, 36)

start_button = pygame_gui.elements.UIButton(
    relative_rect=pygame.Rect((w - panel_width + 20, 360), (150, 40)),
    text='Start Simulation',
    manager=manager
)

# Constants and initial values
g = 9.81
scale = 100
offset1 = (w // 3, h // 3)
# offset2 = (3 * w // 4 - panel_width // 2, h // 4)
offset2 = offset1

def G(y, t):
    a1d, a2d = y[0], y[1]
    a1, a2 = y[2], y[3]
    m11, m12 = (m1 + m2) * l1, m2 * l2 * cos(a1 - a2)
    m21, m22 = l1 * cos(a1 - a2), l2
    m = np.array([[m11, m12], [m21, m22]])
    f1 = -m2 * l2 * a2d * a2d * sin(a1 - a2) - (m1 + m2) * g * sin(a1)
    f2 = l1 * a1d * a1d * sin(a1 - a2) - g * sin(a2)
    accel = inv(m).dot([f1, f2])
    return np.array([accel[0], accel[1], a1d, a2d])

def RK4_step(y, t, dt):
    k1 = G(y, t)
    k2 = G(y + 0.5 * k1 * dt, t + 0.5 * dt)
    k3 = G(y + 0.5 * k2 * dt, t + 0.5 * dt)
    k4 = G(y + k3 * dt, t + dt)
    return dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6

def update(a1, a2, offset):
    x1 = l1 * scale * sin(a1) + offset[0]
    y1 = l1 * scale * cos(a1) + offset[1]
    x2 = x1 + l2 * scale * sin(a2)
    y2 = y1 + l2 * scale * cos(a2)
    return (x1, y1), (x2, y2)

clock = pygame.time.Clock()
trace_surface = pygame.Surface((w, h))
trace_surface.fill(BLACK)
trace_surface.set_colorkey(BLACK)

running = True
simulation_started = False
y1 = y2 = np.zeros(4)
t = 0.0
delta_t = 0.02

m1 = m2 = l1 = l2 = 1.0
prev_point1 = None
prev_point2 = None

while running:
    time_delta = clock.tick(60) / 1000.0

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        if event.type == pygame_gui.UI_BUTTON_PRESSED and event.ui_element == start_button:
            try:
                m1 = float(inputs['m1'].get_text() or 1.0)
                m2 = float(inputs['m2'].get_text() or 1.0)
                l1 = float(inputs['l1'].get_text() or 1.0)
                l2 = float(inputs['l2'].get_text() or 1.0)
                a1 = float(inputs['a1'].get_text() or 30)
                a2 = float(inputs['a2'].get_text() or -30)
                y1 = np.array([0.0, 0.0, np.deg2rad(a1), np.deg2rad(a1)])
                y2 = np.array([0.0, 0.0, np.deg2rad(a2), np.deg2rad(-a2)])
                t = 0.0
                simulation_started = True
                trace_surface.fill(BLACK)
                prev_point1 = None
                prev_point2 = None
            except ValueError:
                pass

        manager.process_events(event)

    manager.update(time_delta)
    screen.fill(BLACK)

    if simulation_started:
        point1, point2 = update(y1[2], y1[3], offset1)
        point3, point4 = update(y2[2], y2[3], offset2)

        # Draw traces
        if prev_point1:
            pygame.draw.line(trace_surface, GREEN, prev_point1, point2, 1)
        if prev_point2:
            pygame.draw.line(trace_surface, RED, prev_point2, point4, 1)
        prev_point1 = point2
        prev_point2 = point4

        screen.blit(trace_surface, (0, 0))

        # Draw pendulums
        pygame.draw.line(screen, WHITE, offset1, point1, 4)
        pygame.draw.line(screen, WHITE, point1, point2, 4)
        pygame.draw.circle(screen, RED, [int(x) for x in point1], 10)
        pygame.draw.circle(screen, BLUE, [int(x) for x in point2], 10)

        pygame.draw.line(screen, CYAN, offset2, point3, 4)
        pygame.draw.line(screen, CYAN, point3, point4, 4)
        pygame.draw.circle(screen, RED, [int(x) for x in point3], 10)
        pygame.draw.circle(screen, BLUE, [int(x) for x in point4], 10)

        y1 = y1 + RK4_step(y1, t, delta_t)
        y2 = y2 + RK4_step(y2, t, delta_t)
        t += delta_t

        time_surf = font.render(f"Time: {t:.2f} s", True, WHITE)
        screen.blit(time_surf, (50, 20))

    manager.draw_ui(screen)
    pygame.display.update()

pygame.quit()
sys.exit()
