import pygame
import sys
from pygame.locals import *
from math import sin, cos, pi
import numpy as np
from numpy.linalg import inv


def G(y, t):
    a1d, a2d = y[0], y[1]
    a1, a2 = y[2], y[3]

    m11, m12 = (m1+m2)*l1, m2*l2*cos(a1-a2)
    m21, m22 = l1*cos(a1-a2), l2
    m = np.array([[m11, m12], [m21, m22]])

    f1 = -m2*l2*a2d*a2d*sin(a1-a2) - (m1+m2)*g*sin(a1)
    f2 = l1*a1d*a1d*sin(a1-a2) - g*sin(a2)
    f = np.array([f1, f2])

    accel = inv(m).dot(f)

    return np.array([accel[0], accel[1], a1d, a2d])


def RK4_step(y, t, dt):
    k1 = G(y, t)
    k2 = G(y+0.5*k1*dt, t+0.5*dt)
    k3 = G(y+0.5*k2*dt, t+0.5*dt)
    k4 = G(y+k3*dt, t+dt)

    return dt * (k1 + 2*k2 + 2*k3 + k4) / 6


def update(a1, a2,offset):
    scale = 100
    x1 = l1*scale * sin(a1) + offset[0]
    y1 = l1*scale * cos(a1) + offset[1]
    x2 = x1 + l2*scale * sin(a2)
    y2 = y1 + l2*scale * cos(a2)

    return (x1, y1), (x2, y2)


def render(point1, point2, point3, point4):
    scale = 10
    x1, y1 = int(point1[0]), int(point1[1])
    x2, y2 = int(point2[0]), int(point2[1])
    x3, y3 = int(point3[0]), int(point3[1])
    x4, y4 = int(point4[0]), int(point4[1])

    if prev_point:
        x1p, y1p, x2p, y2p = prev_point[0], prev_point[1], prev_point[2], prev_point[3]
        pygame.draw.line(trace, pygame.Color("green"), (x1p, y1p),
                         (x2, y2), 1)   # Tracing the pendulum
        pygame.draw.line(trace, pygame.Color("red"), (x2p, y2p),
                         (x4, y4), 1)   # Tracing the pendulum

    screen.fill(BLACK)
    screen.blit(trace, (0, 0))

    pygame.draw.line(screen, WHITE, offset1, (x1, y1), 5)
    pygame.draw.line(screen, WHITE, (x1, y1), (x2, y2), 5)
    pygame.draw.circle(screen, WHITE, offset1, 8)
    pygame.draw.circle(screen, WHITE, offset2, 8)
    pygame.draw.circle(screen, RED, (x1, y1), int(0.5*int(m1*scale)))
    pygame.draw.circle(screen, BLUE, (x2, y2), int(0.5*int(m2*scale)))
    pygame.draw.line(screen, WHITE, offset2, (x3,y3),5)
    pygame.draw.line(screen,WHITE,(x3,y3),(x4,y4),5)
    pygame.draw.circle(screen,RED,(x3,y3),int(0.5*int(m1*scale)))
    pygame.draw.circle(screen,BLUE, (x4,y4), int(0.5*int(m2*scale)))

    return (x2, y2, x4, y4)

w, h = 1366 , 768
WHITE = (255,255,255)
BLACK = (0,0,0)
RED = (255,0,0)
BLUE = (0,0,255)
LT_BLUE = (230,230,255)
offset1 = (100, 100)
offset2 = (200, 100)

screen = pygame.display.set_mode((w,h))
screen.fill(BLACK)
trace = screen.copy()
pygame.display.update()
clock = pygame.time.Clock()

# parameters
m1, m2 = 1.0, 1.0
l1, l2 = 1.0, 1.0
# a1, a2 = pi/4, -1.0
g = 9.81

prev_point = None
t = 0.0
delta_t = 0.02
y1 = np.array([0.0, 0.0, np.deg2rad(6.37),np.deg2rad(6.37) ])
y2 = np.array([0.0, 0.0, np.deg2rad(6.37),np.deg2rad(-6.37) ])
pygame.font.init()
myfont = pygame.font.SysFont('Comic Sans MS', 24)

while True:
	for event in pygame.event.get():
		if event.type == pygame.QUIT:
			sys.exit()

	point1, point2  = update(y1[2], y1[3],offset1)
	point3, point4  = update(y2[2], y2[3],offset2)
	prev_point = render(point1, point2,point3, point4)

	time_string = 'Time: {} seconds'.format(round(t,1))
	text = myfont.render(time_string, False, WHITE)
	screen.blit(text, (10,10))

	t += delta_t
	y1 = y1 + RK4_step(y1, t, delta_t)
	y2 = y2 + RK4_step(y2, t, delta_t)

	clock.tick(60)
	pygame.display.update()
