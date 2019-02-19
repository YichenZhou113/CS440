# geometry.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Jongdeog Lee (jlee700@illinois.edu) on 09/12/2018

"""
This file contains geometry functions that relate with Part1 in MP2.
"""

import math
import numpy as np
from const import *


def computeCoordinate(start, length, angle):
    """Compute the end cooridinate based on the given start position, length and angle.

        Args:
            start (tuple): base of the arm link. (x-coordinate, y-coordinate)
            length (int): length of the arm link
            angle (int): degree of the arm link from x-axis to couter-clockwise

        Return:
            End position of the arm link, (x-coordinate, y-coordinate)
    """
    angle = (angle*2*math.pi)/360
    return (start[0]+length*math.cos(angle),start[1]-length*math.sin(angle))

def doesArmTouchObstacles(armPos, obstacles):
    """Determine whether the given arm links touch obstacles

        Args:
            armPos (list): start and end position of all arm links [(start, end)]
            obstacles (list): x-, y- coordinate and radius of obstacles [(x, y, r)]

        Return:
            True if touched. False it not.
    """
    for i in range(len(armPos)):
        cur_arm = armPos[i]
        arm_x = [cur_arm[0][0],cur_arm[1][0]]
        arm_y = [cur_arm[0][1],cur_arm[1][1]]
        if (arm_x[0] != arm_x[1]):
            arm_a = (arm_y[1]-arm_y[0])/(arm_x[1]-arm_x[0])
            arm_b = arm_y[1]-arm_a*arm_x[1]
            for i in range(len(obstacles)):
                cur_obs = obstacles[i]
                x_range = np.linspace(arm_x[0],arm_x[1],1000)
                y_range = arm_a * x_range + arm_b
                for j in range(1000):
                    cur_x = x_range[j]
                    cur_y = y_range[j]
                    if(((cur_y-cur_obs[1])**2 +(cur_x-cur_obs[0])**2) <= cur_obs[2]**2):
                        return True
        if (arm_x[0] == arm_x[1]):
            for i in range(len(obstacles)):
                cur_obs = obstacles[i]
                y_range = np.linspace(arm_y[0],arm_y[1],1000)
                cur_x = arm_x[0]
                for j in range(1000):
                    cur_y = y_range[j]
                    if(((cur_y-cur_obs[1])**2 +(cur_x-cur_obs[0])**2) <= cur_obs[2]**2):
                        return True


    #print(obstacles)

    return False

def doesArmTouchGoals(armEnd, goals):
    """Determine whether the given arm links touch goals

        Args:
            armEnd (tuple): the arm tick position, (x-coordinate, y-coordinate)
            goals (list): x-, y- coordinate and radius of goals [(x, y, r)]

        Return:
            True if touched. False it not.
    """
    for i in range(len(goals)):
        single_goal = goals[i]
        if((single_goal[1]-armEnd[1])**2+(single_goal[0]-armEnd[0])**2<single_goal[2]**2):
            return True
    return False


def isArmWithinWindow(armPos, window):
    """Determine whether the given arm stays in the window

        Args:
            armPos (list): start and end position of all arm links [(start, end)]
            window (tuple): (width, height) of the window

        Return:
            True if all parts are in the window. False it not.
    """
    for pos in armPos:
        if(pos[0][0]<0 or pos[0][0]>window[0] or pos[1][0]<0 or pos[1][0]>window[0]):
            return False
        if(pos[0][1]<0 or pos[0][1]>window[1] or pos[1][1]<0 or pos[1][1]>window[1]):
            return False


    return True
