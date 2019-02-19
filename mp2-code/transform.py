
# transform.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Jongdeog Lee (jlee700@illinois.edu) on 09/12/2018

"""
This file contains the transform function that converts the robot arm map
to the maze.
"""
import copy
from arm import Arm
from maze import Maze
from search import *
from geometry import *
from const import *
from util import *

def transformToMaze(arm, goals, obstacles, window, granularity):
    """This function transforms the given 2D map to the maze in MP1.

        Args:
            arm (Arm): arm instance
            goals (list): [(x, y, r)] of goals
            obstacles (list): [(x, y, r)] of obstacles
            window (tuple): (width, height) of the window
            granularity (int): unit of increasing/decreasing degree for angles

        Return:
            Maze: the maze instance generated based on input arguments.

    """
    alpha_min = arm.getArmLimit()[0][0]
    alpha_max = arm.getArmLimit()[0][1]
    num_rows = int(((alpha_max-alpha_min)/granularity))+1
    beta_min = arm.getArmLimit()[1][0]
    beta_max = arm.getArmLimit()[1][1]
    num_cols = int(((beta_max-beta_min)/granularity))+1
    maze = []
    for i in range(int(num_rows)):
        maze.insert(len(maze), [])
    for i in range(int(num_rows)):
        for j in range(int(num_cols)):
            maze[i].insert(len(maze[i]), ' ')

    #print(maze)
    print(alpha_min,beta_min)
    """print('y not',len(maze))
    print('goals are:',goals)
    print('obstacles are:',obstacles)
    print('window',window)
    print('rows and cols',(num_rows,num_cols))"""

    start_point = arm.getArmAngle()
    print(start_point)
    print('fuck me',int((start_point[0]-alpha_min)/granularity),int((start_point[1]-beta_min)/granularity))


    maze[int((start_point[0]-alpha_min)/granularity)][int((start_point[1]-beta_min)/granularity)] = 'P'

    alpha = alpha_min
    alpha_index = 0
    while(alpha <= alpha_max):
        beta = beta_min
        beta_index = 0
        while(beta <= beta_max):
            arm.setArmAngle((alpha, beta))
            cur_armEnd = arm.getEnd()
            cur_armPos = arm.getArmPos()
            if(doesArmTouchObstacles(cur_armPos, obstacles)):
                maze[alpha_index][beta_index] = '%'
            if(doesArmTouchGoals(cur_armEnd, goals)):
                maze[alpha_index][beta_index] = '.'
            if not (isArmWithinWindow(cur_armPos, window)):
                maze[alpha_index][beta_index] = '%'
            #print(cur_coord)
            beta += granularity
            beta_index += 1
        alpha += granularity
        alpha_index += 1

    f = open('mazeyichen.txt','w')
    for i in range(num_rows):
        for j in range(num_cols):
            f.write(maze[i][j])
        f.write('\n')
    f.close()

    maze_ret = Maze(maze,[alpha_min,beta_min],granularity)
    #maze_ret.saveToFile('bb.txt')
    return maze_ret

    pass
