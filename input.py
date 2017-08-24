# """""""""""""""""""""""""""""""""""""""""""
# "                                         "
# "       Input.py -- Charles Herrmann      "
# "               6/22/16                   "
# "                                         "
# """""""""""""""""""""""""""""""""""""""""""

# For anyone looking at this, other than me, Im so sorry.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import sys
import numpy
import random
import tensorflow as tf
from Tkinter import *

class Input(object):

  def __init__(self, pits, goals):
    self.player_pos = 2
    self.size = 5
    self.originals = [pits, goals]
    self.pits = pits
    self.goals = goals
    self.e_layer = numpy.zeros((self.size, self.size))
    self.total_score = 0

  def grid(self):
    grid = numpy.zeros((self.size, self.size, 3))
    for i in range(self.size):
      for j in range(self.size):
        if self.e_layer[i][j] == -1:
          grid[i][j][1] = 1
        elif self.e_layer[i][j] == 1:
          grid[i][j][0] = 1
    grid[self.size-1][self.player_pos][2] = 1
    return grid

  def restart(self):
    self.pits = self.originals[0]
    self.goals = self.originals[1]
    self.e_layer = numpy.zeros((self.size, self.size))
    self.player_pos = 2
    self.total_score = 0

  def insert_elements(self):
    layer = numpy.zeros(self.size)
    pp = gp = -1
    if random.randint(self.pits, self.pits*10) > 5 and self.pits > 0:
      pp = random.randint(0, self.size - 1)
      self.pits -= 1
    if random.randint(self.goals, self.goals*10) > 5 and self.goals > 0:
      gp = random.randint(0, self.size - 1)
      while gp == pp:
        gp = random.randint(0, self.size - 1)
      self.goals -= 1
    if pp != -1:
      layer[pp] = -1
    if gp != -1:
      layer[gp] = 1
    return layer

  def move(self, action):
    if action == 0:
      if self.player_pos > 0:
        self.player_pos -= 1
    elif action == 2:
      if self.player_pos < self.size - 1:
        self.player_pos += 1
    tmp = self.e_layer[self.size - 1]
    for t in tmp:
      if t == -1:
        self.pits += 1
      elif t == 1:
        self.goals += 1
    for i in range(self.size - 1):
      self.e_layer[self.size - i - 1] = self.e_layer[self.size - i - 2]
    self.e_layer[0] = self.insert_elements()
    self.total_score += self.reward()

  def reward(self):
    if self.e_layer[self.size - 1][self.player_pos] == -1:
      return -100
    elif self.e_layer[self.size - 1][self.player_pos] == 1:
      return 50
    else:
      return -0.04

  def total_out(self, type):
    count = 0
    for i in range(self.size):
      for j in range(self.size):
        if self.e_layer[i][j] == type:
          count += 1
    return count

  def display(self):
    grid = numpy.zeros((self.size,self.size), dtype='<U2')
    for i in range(self.size):
      for j in range(self.size):
        if self.e_layer[i][j] == -1:
          grid[i][j] = '-'
        elif self.e_layer[i][j] == 1:
          grid[i][j] = '+'
        elif i == (self.size - 1) and j == self.player_pos:
          grid[i][j] = 'P'
        else:
          grid[i][j] = ' '
    print(grid)
    print('')

