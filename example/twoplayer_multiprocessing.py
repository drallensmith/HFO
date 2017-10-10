#!/usr/bin/env python
"""
This is a script to explore the use of two players controlled by the same
script but two (coordinated) processes. It is also for looking at high vs low
feature spaces and other aspects of state processing (e.g., amount of
error tolerance needed).
"""
from __future__ import print_function, division
# encoding: utf-8

import argparse
#import functools
import itertools
import math
import multiprocessing
import multiprocessing.sharedctypes
import os
import random
#import struct
import subprocess
import sys
import time
import warnings

#import numpy as np

# The below will be needed if doing data gathering
# (see commented-out section at bottom of file),
# as will the commented-out do_theilsen function below,
# if wishing to use the code as-is.
from scipy import stats

try:
  import hfo
except ImportError:
  print('Failed to import hfo. To install hfo, in the HFO directory'\
    ' run: \"pip install .\"')
  exit()

# TARGET_ONLY: just goes between target locations, as opposed to (random) other actions;
# can result in running low on stamina if steps per episode are too high,
# due to MOVE_TO always going on maximum Dash power.
TARGET_ONLY = True
FULLSTATE = False

HALF_FIELD_WIDTH = 34 # y coordinate
HALF_FIELD_FULL_WIDTH = HALF_FIELD_WIDTH * 2.2
HALF_FIELD_LENGTH = 52.5 # x coordinate
HALF_FIELD_FULL_LENGTH = HALF_FIELD_LENGTH * 1.2
GOAL_WIDTH = 14.02
MAX_DIST = math.sqrt((HALF_FIELD_LENGTH*HALF_FIELD_LENGTH) +
                     (4.0*HALF_FIELD_WIDTH*HALF_FIELD_WIDTH))
print("MAX_DIST is {0:n}".format(MAX_DIST))
ERROR_TOLERANCE = math.pow(sys.float_info.epsilon,0.25)
# Below is due to quantization - even in fullstate!
REAL_POS_ERROR_TOLERANCE = math.exp(math.log(MAX_DIST)+0.1) - MAX_DIST
print("REAL_POS_ERROR_TOLERANCE is {0!r}".format(REAL_POS_ERROR_TOLERANCE))
ANGLE_DEGREE_ERROR_TOLERANCE = math.degrees(0.1) # quantization of angles

MAX_REAL_X_VALID = 1.1*HALF_FIELD_LENGTH
MIN_REAL_X_VALID = -0.1*HALF_FIELD_LENGTH
MAX_REAL_Y_VALID = 1.1*HALF_FIELD_WIDTH
MIN_REAL_Y_VALID = -1.1*HALF_FIELD_WIDTH

def normalize_full_range(pos, min_val, max_val):
  assert max_val > min_val
  top = (pos - min_val)/(max_val - min_val)
  bot = 1.0 - -1.0
  num = (top*bot) + -1.0
  return num

POS_ERROR_TOLERANCE = 2.0*(REAL_POS_ERROR_TOLERANCE/HALF_FIELD_FULL_LENGTH)
print("POS_ERROR_TOLERANCE is {0!r}".format(POS_ERROR_TOLERANCE))

# Below - requires scipy

def do_theilsen(x_data, y_data):
    slope, ignored_intercept, low, high = stats.theilslopes(y_data, x_data)
    if math.isnan(slope):
        slope, ignored_intercept, ignored_rvalue, ignored_pvalue, stderr = stats.linregress(x_data, y_data)
        low = slope - (1.96*stderr)
        high = slope + (1.96*stderr)
    intercept_list = []
    intercept_low_list = []
    intercept_high_list = []
    for x_use, y_use in zip(x_data, y_data):
        intercept_list.append(y_use - (slope*x_use))
        intercept_low_list.append(y_use - (low*x_use))
        intercept_high_list.append(y_use - (high*x_use))
    intercept = stats.trim_mean(intercept_list, proportiontocut=0.25)
    intercept_low = stats.trim_mean(intercept_low_list, proportiontocut=0.25)
    intercept_high = stats.trim_mean(intercept_high_list, proportiontocut=0.25)
    return slope, intercept, low, high, intercept_low, intercept_high

def unnormalize(pos, min_val, max_val, silent=False, allow_outside=False):
  assert max_val > min_val
  if not allow_outside:
    if (not silent) and (abs(pos) > 1.0+ERROR_TOLERANCE):
      print("Pos {0:n} fed to unnormalize (min_val {1:n}, max_val {2:n})".format(
        pos, min_val, max_val), file=sys.stderr)
      sys.stderr.flush()
    pos = min(1.0,max(-1.0,pos))
  top = (pos+1.0)/2.0 # top = (pos - -1.0)/(1.0 - -1.0)
  bot = max_val - min_val
  return (top*bot) + min_val

def get_y_unnormalized(y_pos, silent=False, check_reverse=True, allow_outside=False):
  y_pos_real = unnormalize(y_pos, MIN_REAL_Y_VALID, MAX_REAL_Y_VALID, silent=silent,
                           allow_outside=allow_outside)
  if check_reverse:
    if allow_outside:
      est_y_pos = get_y_normalized_full_range(y_pos_real)
    else:
      est_y_pos = get_y_normalized(y_pos_real, silent=silent)
    if abs(y_pos - est_y_pos) > POS_ERROR_TOLERANCE:
      raise RuntimeError(
        "Bad denormalization/normalization of {0:n} to {1:n}; reverse {2:n}".format(
          y_pos, y_pos_real, est_y_pos))
  return y_pos_real

def get_x_unnormalized(x_pos, silent=False, check_reverse=True, allow_outside=False):
  x_pos_real = unnormalize(x_pos, MIN_REAL_X_VALID, MAX_REAL_X_VALID, silent=silent,
                           allow_outside=allow_outside)
  if check_reverse:
    if allow_outside:
      est_x_pos = get_x_normalized_full_range(x_pos_real)
    else:
      est_x_pos = get_x_normalized(x_pos_real, silent=silent)
    if abs(x_pos - est_x_pos) > POS_ERROR_TOLERANCE:
      raise RuntimeError(
        "Bad denormalization/normalization of {0:n} to {1:n}; reverse {2:n}".format(
          x_pos, x_pos_real, est_x_pos))
  return x_pos_real

def normalize(pos, min_val, max_val, silent=False):
  assert max_val > min_val
  top = (pos - min_val)/(max_val - min_val)
  bot = 1.0 - -1.0
  num = (top*bot) + -1.0
  if (not silent) and (abs(num) > 1.0+POS_ERROR_TOLERANCE):
    print(
      "Pos {0:n} gives normalized num {1:n} (min_val {2:n}, max_val {3:n})".format(
        pos, num, min_val, max_val), file=sys.stderr)
    sys.stderr.flush()
  return min(1.0,max(-1.0,num))



def get_y_normalized(y_pos, silent=False):
  return normalize(y_pos, MIN_REAL_Y_VALID, MAX_REAL_Y_VALID, silent=silent)

def get_y_normalized_full_range(y_pos):
  return normalize_full_range(y_pos, MIN_REAL_Y_VALID, MAX_REAL_Y_VALID)

def get_x_normalized(x_pos, silent=False):
  return normalize(x_pos, MIN_REAL_X_VALID, MAX_REAL_X_VALID, silent=silent)

def get_x_normalized_full_range(x_pos):
  return normalize_full_range(x_pos, MIN_REAL_X_VALID, MAX_REAL_X_VALID)


GOAL_POS_X = get_x_normalized(HALF_FIELD_LENGTH)
GOAL_TOP_POS_Y = get_y_normalized(-1.0*(GOAL_WIDTH/2.0))
GOAL_BOTTOM_POS_Y = get_y_normalized(GOAL_WIDTH/2.0)

# Avoiding ball going out of bounds by accident
MAX_POS_Y_BALL_SAFE = get_y_normalized(HALF_FIELD_WIDTH) - POS_ERROR_TOLERANCE
MIN_POS_Y_BALL_SAFE = get_y_normalized(-1*HALF_FIELD_WIDTH) + POS_ERROR_TOLERANCE
MAX_POS_X_BALL_SAFE = get_x_normalized(HALF_FIELD_LENGTH) - POS_ERROR_TOLERANCE
MIN_POS_X_BALL_SAFE = get_x_normalized(0) + POS_ERROR_TOLERANCE

MAX_POS_X_OK = 1.0 - ERROR_TOLERANCE
MIN_POS_X_OK = -1.0 + ERROR_TOLERANCE
MAX_POS_Y_OK = MAX_POS_X_OK
MIN_POS_Y_OK = MIN_POS_X_OK

def get_dist_real(ref_x, ref_y, src_x, src_y, silent=False):
  ref_x_real = get_x_unnormalized(ref_x, silent=silent, allow_outside=silent)
  ref_y_real = get_y_unnormalized(ref_y, silent=silent, allow_outside=silent)
  src_x_real = get_x_unnormalized(src_x, silent=silent, allow_outside=silent)
  src_y_real = get_y_unnormalized(src_y, silent=silent, allow_outside=silent)

  return math.sqrt(math.pow((ref_x_real - src_x_real),2) +
                   math.pow((ref_y_real - src_y_real),2))

def get_dist_from_real(ref_x_real, ref_y_real, src_x_real, src_y_real):
    return math.sqrt(math.pow((ref_x_real - src_x_real),2) +
                     math.pow((ref_y_real - src_y_real),2))

def get_dist_normalized(ref_x, ref_y, src_x, src_y):
  return math.sqrt(math.pow((ref_x - src_x),2) +
                   math.pow(((HALF_FIELD_WIDTH/HALF_FIELD_LENGTH)*(ref_y - src_y)),2))

#Instead of adding six as a dependency, this code was copied from the six implementation.
#six is Copyright (c) 2010-2015 Benjamin Peterson

if sys.version_info[0] >= 3:
    def iterkeys(d, **kw):
        return iter(d.keys(**kw))

    def iteritems(d, **kw):
        return iter(d.items(**kw))

    def itervalues(d, **kw):
        return iter(d.values(**kw))
else:
    def iterkeys(d, **kw):
        return iter(d.iterkeys(**kw))

    def iteritems(d, **kw):
        return iter(d.iteritems(**kw))

    def itervalues(d, **kw):
        return iter(d.itervalues(**kw))

##BIT_LIST_LEN = 8
##STRUCT_PACK = "BH"

def get_dist_from_proximity(proximity, max_dist=MAX_DIST):
  proximity_real = unnormalize(proximity, 0.0, 1.0)
  dist = (1 - proximity_real)*max_dist
  if (dist > (max_dist+ERROR_TOLERANCE)) or (dist <= (-1.0*ERROR_TOLERANCE)):
    warnings.warn("Proximity {0!r} gives dist {1!r} (max_dist {2:n})".format(
      proximity,dist,max_dist))
    sys.stderr.flush()
    return min(max_dist,max(0,dist))
  return max(0,dist)

def get_proximity_from_dist(dist, max_dist=MAX_DIST):
  if dist > (max_dist+ERROR_TOLERANCE):
    print("Dist {0:n} is above max_dist {1:n}".format(dist, max_dist), file=sys.stderr)
    sys.stderr.flush()
  proximity_real = min(1.0, max(0.0, (1.0 - (dist/max_dist))))
  return normalize(proximity_real, 0.0, 1.0)

def get_angle(sin_angle,cos_angle):
  if max(abs(sin_angle),abs(cos_angle)) <= sys.float_info.epsilon:
    warnings.warn(
      "Unable to accurately determine angle from sin_angle {0:n} cos_angle {1:n}".format(
        sin_angle, cos_angle))
    return None
  angle = math.degrees(math.atan2(sin_angle,cos_angle))
  if angle < 0:
    angle += 360
  return angle

def get_abs_angle(body_angle,rel_angle):
  if (body_angle is None) or (rel_angle is None):
    return None
  angle = body_angle + rel_angle
  while angle >= 360:
    angle -= 360
  if angle < 0:
    if abs(angle) <= ANGLE_DEGREE_ERROR_TOLERANCE:
      return 0.0
    warnings.warn("Bad body_angle {0!r} and/or rel_angle {1!r}".format(
      body_angle,rel_angle))
    return None
  return angle

def get_angle_diff(angle1, angle2):
  return min((angle1 - angle2),abs(angle1 - angle2 - 360),abs(angle2 - angle1 - 360))

def reverse_angle(angle):
  angle += 180
  while angle >= 360:
    angle -= 360
  return angle

def get_abs_x_y_pos(abs_angle, dist, self_x_pos, self_y_pos, warn=True, of_what="",
                    allow_outside=False):
  """
  Use angle, distance, and own position to estimate the absolute position
  of a landmark feature (such as another player or the ball),
  or use in reverse with a known-position landmark to estimate own position.
  """
  poss_xy_pos_real = {}
  max_deviation_xy_pos_real = {}
  total_deviation_xy_pos_real = {}
  dist_xy_pos_real = {}
  self_x_pos_real = get_x_unnormalized(self_x_pos, allow_outside=True)
  self_y_pos_real = get_y_unnormalized(self_y_pos, allow_outside=True)

  start_string = ""
  if of_what:
    start_string = of_what + ': '

  if allow_outside:
    angles_check = [abs_angle]
  else:
    angles_check = [(abs_angle-ANGLE_DEGREE_ERROR_TOLERANCE),
                    (abs_angle+ANGLE_DEGREE_ERROR_TOLERANCE),
                    (abs_angle-(ANGLE_DEGREE_ERROR_TOLERANCE/2.0)),
                    (abs_angle+(ANGLE_DEGREE_ERROR_TOLERANCE/2.0)),
                    abs_angle]

  for angle in angles_check:
    angle_radians = math.radians(angle)
    sin_angle = math.sin(angle_radians)
    cos_angle = math.cos(angle_radians)


    est_x_pos_real = (cos_angle*dist) + self_x_pos_real
    est_y_pos_real = (sin_angle*dist) + self_y_pos_real
    if allow_outside:
      poss_xy_pos_real[angle] = (est_x_pos_real, est_y_pos_real)
    elif (((MIN_REAL_X_VALID-REAL_POS_ERROR_TOLERANCE)
          <= est_x_pos_real <=
          (MAX_REAL_X_VALID+REAL_POS_ERROR_TOLERANCE)) and
          ((MIN_REAL_Y_VALID-REAL_POS_ERROR_TOLERANCE)
           <= est_y_pos_real <=
           (MAX_REAL_Y_VALID+REAL_POS_ERROR_TOLERANCE))):
      poss_xy_pos_real[angle] = (est_x_pos_real, est_y_pos_real)

      if est_x_pos_real < MIN_REAL_X_VALID:
        x_deviation = (MIN_REAL_X_VALID-est_x_pos_real)/(MAX_REAL_X_VALID-MIN_REAL_X_VALID)
      elif est_x_pos_real > MAX_REAL_X_VALID:
        x_deviation = (est_x_pos_real-MAX_REAL_X_VALID)/(MAX_REAL_X_VALID-MIN_REAL_X_VALID)
      else:
        x_deviation = 0.0

      if est_y_pos_real < MIN_REAL_Y_VALID:
        y_deviation = (MIN_REAL_Y_VALID-est_y_pos_real)/(MAX_REAL_Y_VALID-MIN_REAL_Y_VALID)
      elif est_y_pos_real > MAX_REAL_Y_VALID:
        y_deviation = (est_y_pos_real-MAX_REAL_Y_VALID)/(MAX_REAL_Y_VALID-MIN_REAL_Y_VALID)
      else:
        y_deviation = 0.0

      max_deviation_xy_pos_real[angle] = max(x_deviation,y_deviation)
      total_deviation_xy_pos_real[angle] = x_deviation+y_deviation

      dist_xy_pos_real[angle] = get_dist_from_real(est_x_pos_real,
                                                   est_y_pos_real,
                                                   min(MAX_REAL_X_VALID,
                                                       max(MIN_REAL_X_VALID,
                                                           est_x_pos_real)),
                                                   min(MAX_REAL_Y_VALID,
                                                       max(MIN_REAL_Y_VALID,
                                                           est_y_pos_real)))

    elif (angle == abs_angle) and (not poss_xy_pos_real):
      error_strings = []
      if not ((MIN_REAL_X_VALID-REAL_POS_ERROR_TOLERANCE)
              <= est_x_pos_real <=
              (MAX_REAL_X_VALID+REAL_POS_ERROR_TOLERANCE)):
        error_strings.append(
          "{0!s}Bad est_x_pos_real {1:n} from self_x_pos_real {2:n} (self_x_pos {3:n}), angle {4:n} ({5:n} {6:n}), dist {7:n}".format(
            start_string,est_x_pos_real, self_x_pos_real, self_x_pos, abs_angle, sin_angle, cos_angle, dist))
      if not ((MIN_REAL_Y_VALID-REAL_POS_ERROR_TOLERANCE)
              <= est_y_pos_real <=
              (MAX_REAL_Y_VALID+REAL_POS_ERROR_TOLERANCE)):
        error_strings.append(
          "{0!s}Bad est_y_pos_real {1:n} from self_y_pos_real {2:n} (self_y_pos {3:n}), angle {4:n} ({5:n} {6:n}), dist {7:n}".format(
            start_string, est_y_pos_real, self_y_pos_real, self_y_pos, abs_angle, sin_angle, cos_angle, dist))
      if FULLSTATE and (dist < 10) and (not of_what.startswith("Ball")):
        raise RuntimeError("\n".join(error_strings))
      elif warn or (dist < 10) or (FULLSTATE and (dist < 50)):
        print("\n".join(error_strings), file=sys.stderr)
        sys.stderr.flush()
      return (None, None)

  poss_angles = list(poss_xy_pos_real.keys())
  if len(poss_angles) > 1:
    poss_angles.sort(key=lambda angle: abs(abs_angle-angle))
    poss_angles.sort(key=lambda angle: total_deviation_xy_pos_real[angle])
    poss_angles.sort(key=lambda angle: max_deviation_xy_pos_real[angle])
    poss_angles.sort(key=lambda angle: dist_xy_pos_real[angle])
  elif not poss_angles:
    return (None, None)
  est_x_pos_real, est_y_pos_real = poss_xy_pos_real[poss_angles[0]]
  if allow_outside:
    est_x_pos = get_x_normalized_full_range(est_x_pos_real)
    est_y_pos = get_y_normalized_full_range(est_y_pos_real)
  elif warn:
    est_x_pos = get_x_normalized(est_x_pos_real)
    est_y_pos = get_y_normalized(est_y_pos_real)
  else:
    est_x_pos = get_x_normalized(est_x_pos_real, silent=True)
    est_y_pos = get_y_normalized(est_y_pos_real, silent=True)


  if warn or ((not allow_outside) and (dist < 50)):
    est_dist = get_dist_real(self_x_pos, self_y_pos,
                             est_x_pos, est_y_pos)

    if abs(dist - est_dist) > ERROR_TOLERANCE:
      est2_dist = get_dist_from_real(self_x_pos_real, self_y_pos_real,
                                     est_x_pos_real, est_y_pos_real)
      print(
        "{0!s}Difference in input ({1:n}), output ({2:n}) distances (est2 {3:n}) for get_abs_x_y_pos".format(
          start_string, dist, est_dist, est2_dist),
        file=sys.stderr)
      print(
        "Input angle {0:n}, used angle {1:n}, self_x_pos {2:n}, self_y_pos {3:n}".format(
          abs_angle, poss_angles[0], self_x_pos, self_y_pos),
        file=sys.stderr)
      print(
        "Self_x_pos_real {0:n}, self_y_pos_real {1:n}, est_x_pos_real {2:n}, est_y_pos_real {3:n}".format(
          self_x_pos_real, self_y_pos_real, est_x_pos_real, est_y_pos_real), file=sys.stderr)
      print("Est_x_pos {0:n}, est_y_pos {1:n}".format(est_x_pos, est_y_pos), file=sys.stderr)
      sys.stderr.flush()

  return (est_x_pos, est_y_pos)

landmark_start_to_location = {
  13: (GOAL_POS_X, get_y_normalized(0.0)), # center of goal
  22: (get_x_normalized(HALF_FIELD_LENGTH-16.5), get_y_normalized(0.0)), # Penalty line center
  31: (get_x_normalized(0.0), get_y_normalized(0.0)), # center of entire field
  34: (get_x_normalized(0.0), get_y_normalized(-HALF_FIELD_WIDTH)), # Top Left
  37: (get_x_normalized(HALF_FIELD_LENGTH), get_y_normalized(-HALF_FIELD_WIDTH)), # Top Right
  40: (get_x_normalized(HALF_FIELD_LENGTH), get_y_normalized(HALF_FIELD_WIDTH)), # Bottom Right
  43: (get_x_normalized(0,0), get_y_normalized(HALF_FIELD_WIDTH))} # Bottom Left

TARGET_POS = list(landmark_start_to_location.values())

PROX_AT_THREE = get_proximity_from_dist(3.0)

def get_weight_from_proximity(prox, using_angle=True):
  dist = get_dist_from_proximity(prox)

  if dist <= 3.0:
    weight = min(1.0,((prox+1.0)/2.0))
  elif (not FULLSTATE) and (dist >= 40.0): # from server parameters
    weight = (1.0+((PROX_AT_THREE+1.0)/2.0))/(1.0+((dist/3.0)**2.0))
  else:
    weight = (5.0+((PROX_AT_THREE+1.0)/2.0))/(5.0+((dist/3.0)**2.0))

  if using_angle and (prox > 0.95): # angle problems at short distances
    max_weight = (1.0-prox)/0.05
    max_weight = max(max_weight, ERROR_TOLERANCE)
    weight = min(max_weight,min)

  return weight

def filter_low_level_state(state, namespace):
  """Extract information from low-level state."""
  bit_list = []
  for pos in (0, 1, 9, 11, 12, 50, 54, 10): # TODO: Replace with constants!
    if state[pos] > 0:
      bit_list.append(True)
    else:
      bit_list.append(False)
##  if (state[0] > 0) and (max(abs(state[58]),abs(state[59]))
##                         >= sys.float_info.epsilon): # player2 angle valid
##    bit_list.append(True)
##  else:
##    bit_list.append(False)
##  if len(bit_list) != BIT_LIST_LEN:
##    raise RuntimeError(
##      "Length of bit_list {0:n} not same as BIT_LIST_LEN {1:n}".format(
##        len(bit_list), BIT_LIST_LEN))

  self_dict = {}
  if state[0] > 0: # self position valid
    if max(abs(state[5]),abs(state[6])) <= sys.float_info.epsilon:
      warnings.warn("Validity {0:d}{1:d} but invalid body angle {2:n}, {3:n}".format(
        int(bit_list[0]),int(bit_list[1]),state[5],state[6]))
      self_dict['body_angle'] = None
    else:
      self_dict['body_angle'] = get_angle(state[5],state[6])

    x_pos_from = {}
    x_pos_weight = {}
    x_pos_from_prox = {}
    y_pos_from = {}
    y_pos_weight = {}
    y_pos_from_prox = {}

    if min(state[46],state[47]) < 1.0:
      x_pos1_real = get_dist_from_proximity(state[46],(HALF_FIELD_LENGTH*1.1))
      x_pos2_real = HALF_FIELD_LENGTH - get_dist_from_proximity(state[47],(HALF_FIELD_LENGTH*1.1))
      if state[46] >= 1.0:
        x_pos_from['OOB'] = get_x_normalized(x_pos2_real)
        x_pos_weight['OOB'] = min(0.25,max(ERROR_TOLERANCE,(1.0-abs(state[47])))/4.0)
        x_pos_from_prox['OOB'] = state[47]
      elif state[47] >= 1.0:
        x_pos_from['OOB'] = get_x_normalized(x_pos1_real)
        x_pos_weight['OOB'] = min(0.25,max(ERROR_TOLERANCE,(1.0-abs(state[46])))/4.0)
        x_pos_from_prox['OOB'] = state[46]
      elif abs(x_pos1_real-x_pos2_real) > REAL_POS_ERROR_TOLERANCE:
        print(
          "state[46] {0:n}, state[47] {1:n}, x_pos1_real {2:n}, x_pos2_real {3:n}".format(
            state[46],state[47],x_pos1_real,x_pos2_real),file=sys.stderr)
        sys.stderr.flush()
        self_dict['x_pos_from_OOB'] = None
      else:
        x_pos_from['OOB'] = get_x_normalized((x_pos1_real+x_pos2_real)/2.0)
        x_pos_weight['OOB'] = 0.5
        x_pos_from_prox['OOB'] = max(state[46],state[47])
    else:
      self_dict['x_pos_from_OOB'] = None

    if min(state[48],state[49]) < 1.0:
      y_pos1_real = get_dist_from_proximity(state[48],
                                            max_dist=(HALF_FIELD_WIDTH*2.0*1.05)) - HALF_FIELD_WIDTH
      y_pos2_real = HALF_FIELD_WIDTH - get_dist_from_proximity(state[49],
                                                               max_dist=(HALF_FIELD_WIDTH*2.0*1.05))
      if state[48] >= 1.0:
        if ((MIN_REAL_Y_VALID-REAL_POS_ERROR_TOLERANCE) <= y_pos2_real <=
            (MAX_REAL_Y_VALID+REAL_POS_ERROR_TOLERANCE)):
          y_pos_from['OOB'] = get_y_normalized(y_pos2_real)
          y_pos_weight['OOB'] = min(0.25,max(ERROR_TOLERANCE,(1.0-abs(state[49])))/4.0)
          y_pos_from_prox['OOB'] = state[49]
        else:
          raise RuntimeError(
            "state[48] is {0:n}, state[49] is {1:n}, y_pos2_real appears to be {2:n}".format(
              state[48],state[49],y_pos2_real))
      elif state[49] >= 1.0:
        if ((MIN_REAL_Y_VALID-REAL_POS_ERROR_TOLERANCE) <= y_pos1_real <=
            (MAX_REAL_Y_VALID+REAL_POS_ERROR_TOLERANCE)):
          y_pos_from['OOB'] = get_y_normalized(y_pos1_real)
          y_pos_weight['OOB'] = min(0.25,max(ERROR_TOLERANCE,(1.0-abs(state[48])))/4.0)
          y_pos_from_prox['OOB'] = state[48]
        else:
          raise RuntimeError(
            "state[48] is {0:n}, state[49] is {1:n}, y_pos1_real appears to be {2:n}".format(
              state[48],state[49],y_pos1_real))
      elif abs(y_pos1_real-y_pos2_real) > REAL_POS_ERROR_TOLERANCE:
        print(
          "state[48] {0:n}, state[49] {1:n}, y_pos1_real {2:n}, y_pos2_real {3:n}".format(
            state[48],state[49],y_pos1_real,y_pos2_real),file=sys.stderr)
        sys.stderr.flush()
        self_dict['y_pos_from_OOB'] = None
      else:
        y_pos_from['OOB'] = get_y_normalized((y_pos1_real+y_pos2_real)/2.0)
        y_pos_weight['OOB'] = 0.5
        y_pos_from_prox['OOB'] = max(state[48],state[49])
    else:
      self_dict['y_pos_from_OOB'] = None

    if min(namespace.other_dict['self_x_pos'].value,
           namespace.other_dict['self_y_pos'].value,
           namespace.other_dict['x_pos'].value,
           namespace.other_dict['y_pos'].value) >= -1.0:
      x_pos_from['OTHER'] = namespace.other_dict['self_x_pos'].value
      y_pos_from['OTHER'] = namespace.other_dict['self_y_pos'].value
      other_dist = get_dist_real(x_pos_from['OTHER'],
                                 y_pos_from['OTHER'],
                                 namespace.other_dict['x_pos'].value,
                                 namespace.other_dict['y_pos'].value)
      other_proximity = get_proximity_from_dist(other_dist)
      x_pos_weight['OTHER'] = y_pos_weight['OTHER'] = get_weight_from_proximity(other_proximity,
                                                                                using_angle=False)
      if abs(x_pos_from['OTHER']) > (1.0-ERROR_TOLERANCE):
        x_pos_weight['OTHER'] = min(ERROR_TOLERANCE,x_pos_weight['OTHER'])
      if abs(y_pos_from['OTHER']) > (1.0-ERROR_TOLERANCE):
        y_pos_weight['OTHER'] = min(ERROR_TOLERANCE,y_pos_weight['OTHER'])
      x_pos_from_prox['OTHER'] = y_pos_from_prox['OTHER'] = other_proximity
      if (max(abs(state[58]),abs(state[59])) > sys.float_info.epsilon) and (bit_list[0] > 0.0):
        other_dist2 = get_dist_from_proximity(state[60])
        if abs(other_dist - other_dist2) > REAL_POS_ERROR_TOLERANCE:
          print(
            "Other dists do not match: other_dist {0!r} ({1!r}), other_dist2 {2!r} ({3!r})".format(
              other_dist, other_proximity, other_dist2, state[60]) +
            "\nOther says self position is {0!r} ({1!r}), {2!r} ({3!r}) and other position is {4!r}, {5!r}".format(
              x_pos_from['OTHER'], get_x_unnormalized(x_pos_from['OTHER'], allow_outside=True),
              y_pos_from['OTHER'], get_y_unnormalized(y_pos_from['OTHER'], allow_outside=True),

              namespace.other_dict['x_pos'].value,
              namespace.other_dict['y_pos'].value),
            file=sys.stderr)
          sys.stderr.flush()
          del(x_pos_from['OTHER'],y_pos_from['OTHER'],
              x_pos_weight['OTHER'],y_pos_weight['OTHER'],
              x_pos_from_prox['OTHER'],y_pos_from_prox['OTHER'])
          self_dict['x_pos_from_OTHER'] = self_dict['y_pos_from_OTHER'] = None
    else:
      self_dict['x_pos_from_OTHER'] = self_dict['y_pos_from_OTHER'] = None

    if self_dict['body_angle'] is not None:
      for landmark_start, xy_location in iteritems(landmark_start_to_location):
        rel_angle = get_angle(state[landmark_start],state[landmark_start+1])
        abs_angle = get_abs_angle(self_dict['body_angle'],rel_angle)
        if abs_angle is not None:
          rev_angle = reverse_angle(abs_angle)
          dist = get_dist_from_proximity(state[landmark_start+2])
          x_pos, y_pos = get_abs_x_y_pos(abs_angle=rev_angle,
                                         dist=dist,
                                         self_x_pos=xy_location[0],
                                         self_y_pos=xy_location[1],
                                         warn=False,
                                         allow_outside=bool(
                                           max(state[46],state[47],state[48],state[49]) >= 1.0),
                                         of_what="Rev " + str(landmark_start))
          if ((x_pos is not None) and
              (max(x_pos,y_pos) <= (1.0+POS_ERROR_TOLERANCE)) and
              (min(x_pos,y_pos) >= (-1.0-POS_ERROR_TOLERANCE))):
            x_pos_from[landmark_start] = x_pos
            y_pos_from[landmark_start] = y_pos
            x_pos_weight[landmark_start] = get_weight_from_proximity(state[landmark_start+2])
            y_pos_weight[landmark_start] = x_pos_weight[landmark_start]
            x_pos_from_prox[landmark_start] = state[landmark_start+2]
            y_pos_from_prox[landmark_start] = x_pos_from_prox[landmark_start]

    x_pos_total_strict = 0.0
    x_pos_weight_strict = 0.0
    if 'OOB' in x_pos_from:
      x_pos_total_strict += x_pos_from['OOB']*x_pos_weight['OOB']
      x_pos_weight_strict += x_pos_weight['OOB']
      if 'OTHER' in x_pos_from:
        x_pos_total_strict += x_pos_from['OTHER']*x_pos_weight['OTHER']
        x_pos_weight_strict += x_pos_weight['OTHER']
      x_pos_strict = x_pos_total_strict/x_pos_weight_strict
    elif 'OTHER' in x_pos_from:
      x_pos_strict = x_pos_from['OTHER']
    else:
      x_pos_strict = None
    self_dict['x_pos_strict'] = x_pos_strict

    for landmark in iterkeys(landmark_start_to_location):
      if landmark not in x_pos_from:
        self_dict['x_pos_from_' + str(landmark)] = None
        self_dict['x_pos_from_dist_' + str(landmark)] = MAX_DIST
      if landmark not in y_pos_from:
        self_dict['y_pos_from_' + str(landmark)] = None
        self_dict['y_pos_from_dist_' + str(landmark)] = MAX_DIST

    y_pos_total = 0.0
    y_pos_weight_total = 0.0
    for from_which, pos in iteritems(y_pos_from):
      y_pos_total += pos*y_pos_weight[from_which]
      y_pos_weight_total += y_pos_weight[from_which]
      self_dict['y_pos_from_' + str(from_which)] = pos
      self_dict['y_pos_from_dist_' + str(from_which)] = get_dist_from_proximity(
        y_pos_from_prox[from_which])
    self_dict['y_pos'] = y_pos_total/y_pos_weight_total
    for from_which, pos in iteritems(y_pos_from):
      if (((y_pos_weight[from_which]/y_pos_weight_total)*abs(pos-self_dict['y_pos']))
          > POS_ERROR_TOLERANCE):
        print(
          "Source {0!r} (weight {1:n} prox {2:n}) y_pos {3:n} vs overall {4:n}".format(
            from_which,
            y_pos_weight[from_which]/
            y_pos_weight_total,
            y_pos_from_prox[from_which],
            pos,
            self_dict['y_pos']),
          file=sys.stderr)
        sys.stderr.flush()

    x_pos_total = 0.0
    x_pos_weight_total = 0.0
    for from_which, pos in iteritems(x_pos_from):
      x_pos_total += pos*x_pos_weight[from_which]
      x_pos_weight_total += x_pos_weight[from_which]
      self_dict['x_pos_from_' + str(from_which)] = pos
      self_dict['x_pos_from_dist_' + str(from_which)] = get_dist_from_proximity(
        x_pos_from_prox[from_which])
    self_dict['x_pos'] = x_pos_total/x_pos_weight_total
    noted_other = False
    for from_which, pos in iteritems(x_pos_from):
      if ((x_pos_strict is not None)
          and (from_which != 'OOB')
          and (x_pos_weight[from_which] > ERROR_TOLERANCE)):
        namespace.data_dict[from_which]['off_orig'].append(pos-x_pos_strict)
        namespace.data_dict[from_which]['abs_off_orig'].append(abs(pos-x_pos_strict))
        namespace.data_dict[from_which]['off'].append(get_x_unnormalized(pos,
                                                                         allow_outside=True)
                                                      - get_x_unnormalized(x_pos_strict,
                                                                           allow_outside=True))
        namespace.data_dict[from_which]['dist'].append(
          get_dist_from_proximity(x_pos_from_prox[from_which]))
        namespace.data_dict[from_which]['x_pos'].append(x_pos_strict)
        namespace.data_dict[from_which]['y_pos'].append(self_dict['y_pos'])
        if from_which == 'OTHER':
          namespace.data_dict[from_which]['dist_strict'].append(
            get_dist_real(x_pos_strict,self_dict['y_pos'],
                          namespace.other_dict['x_pos'].value,
                          namespace.other_dict['y_pos'].value))
        else:
          namespace.data_dict[from_which]['dist_strict'].append(
            get_dist_real(x_pos_strict,self_dict['y_pos'],
                          landmark_start_to_location[from_which][0],
                          landmark_start_to_location[from_which][1]))
      weight_prop = x_pos_weight[from_which]/x_pos_weight_total
      if (((weight_prop*abs(pos-self_dict['x_pos'])) > POS_ERROR_TOLERANCE)
          or ((weight_prop >= 0.1) and (abs(pos-self_dict['x_pos']) > POS_ERROR_TOLERANCE))
          or ((weight_prop > 0.01)
              and ((((x_pos_from_prox[from_which]+1.0)/2.0)*abs(pos-self_dict['x_pos']))
                   > POS_ERROR_TOLERANCE))):
        print(
          "Source {0!r} (weight {1:n} prox {2:n}/{3:n}) x_pos {4:n} ({5:n}) vs overall {6:n} ({7:n})".format(
            from_which,
            weight_prop,
            x_pos_from_prox[from_which],
            get_dist_from_proximity(x_pos_from_prox[from_which]),
            pos,
            get_x_unnormalized(pos, allow_outside=True),
            self_dict['x_pos'],
            get_x_unnormalized(self_dict['x_pos'],
                               allow_outside=True)),
          file=sys.stderr)
        if from_which == 'OTHER':
          noted_other = True
        elif (not noted_other) and ('OTHER' in x_pos_from):
          print(
            "Other x-pos {0:n} ({1:n}; weight {2:n}, prox {3:n}/{4:n})".format(
              x_pos_from['OTHER'],
              get_x_unnormalized(x_pos_from['OTHER']),
              (x_pos_weight['OTHER']/x_pos_weight_total),
              x_pos_from_prox['OTHER'],
              get_dist_from_proximity(x_pos_from_prox['OTHER'])),
            file=sys.stderr)
          noted_other = True
        sys.stderr.flush()


    if max(abs(state[58]),abs(state[59])) <= sys.float_info.epsilon:
      warnings.warn("Validity {0:d}{1:d} but invalid other angle {2:n}, {3:n}".format(
        int(bit_list[0]),int(bit_list[1]),state[58],state[59]))
      self_dict['other_angle'] = None
    else:
      self_dict['other_angle'] = get_angle(state[58],state[59])
  else:
    self_dict['x_pos'] = None
    self_dict['y_pos'] = None
    if max(abs(state[5]),abs(state[6])) > sys.float_info.epsilon:
      warnings.warn("Validity {0:d}{1:d} but valid body angle {2:n}, {3:n}".format(
        int(bit_list[0]),int(bit_list[1]),state[5],state[6]))
      self_dict['body_angle'] = get_angle(state[5],state[6])
    else:
      self_dict['body_angle'] = None
    if max(abs(state[58]),abs(state[59])) > sys.float_info.epsilon:
      warnings.warn("Validity {0:d}{1:d} but valid other angle {2:n}, {3:n}".format(
        int(bit_list[0]),int(bit_list[1]),state[58],state[59]))
      self_dict['other_angle'] = get_angle(state[58],state[59])
    else:
      self_dict['other_angle'] = None

  if state[1] > 0: # self velocity valid
    self_dict['vel_rel_angle'] = get_angle(state[2],state[3])
    self_dict['vel_mag'] = state[4]
  else:
    self_dict['vel_rel_angle'] = None
    self_dict['vel_mag'] = -1

  # NOTE: This is, due to prior uses of this code, the location of the closer goalPOST,
  # not the center of the goal.
  goal_dict = {}
  # below check: self position valid, goal pos possible
  if (state[0] > 0) and (max(state[18],state[21]) > -1):
    if state[18] > state[21]: # top post closer
      which_goal = "Top"
      goal_dict['dist'] = get_dist_from_proximity(state[18])
      goal_dict['rel_angle'] = get_angle(state[16],state[17])
    else:
      which_goal = "Bottom"
      goal_dict['dist'] = get_dist_from_proximity(state[21])
      goal_dict['rel_angle'] = get_angle(state[19],state[20])
    if goal_dict['dist'] > (MAX_DIST-REAL_POS_ERROR_TOLERANCE):
      raise RuntimeError(
        "Should not be possible to have dist {0:n} for goal (state[18] {1:n}, state[21] {2:n})".format(
          goal_dict['dist'], state[18], state[21]))
    goal_dict['abs_angle'] = get_abs_angle(self_dict['body_angle'],goal_dict['rel_angle'])
    if goal_dict['abs_angle'] is not None:

      est_x_pos, est_y_pos = get_abs_x_y_pos(abs_angle=goal_dict['abs_angle'],
                                             dist=goal_dict['dist'],
                                             self_x_pos=self_dict['x_pos'],
                                             self_y_pos=self_dict['y_pos'],
                                             warn=bool(goal_dict['dist'] < 10),
                                             of_what=which_goal+" goal post")

      if est_x_pos is not None:
        if (abs(est_x_pos - GOAL_POS_X) > POS_ERROR_TOLERANCE) and (goal_dict['dist'] < 10):
          print(
            "Est x_pos of goal is {0:n} (should be {1:n}; from est abs_angle {2:n}, dist {3:n}, self_x_pos {4:n})".format(
              est_x_pos,GOAL_POS_X,goal_dict['abs_angle'],goal_dict['dist'],self_dict['x_pos']),
            file=sys.stderr)
          sys.stderr.flush()
      goal_dict['x_pos'] = GOAL_POS_X
      if (state[18] > state[21]):
        goal_dict['y_pos'] = GOAL_TOP_POS_Y
        if est_y_pos is not None:
          if (abs(est_y_pos-GOAL_TOP_POS_Y) > POS_ERROR_TOLERANCE) and (goal_dict['dist'] < 10):
            print(
              "Est y_pos of top goalpost is {0:n} (should be {1:n}; from est abs_angle {2:n}, dist {3:n}, self_y_pos {4:n})".format(
                est_y_pos,GOAL_TOP_POS_Y,goal_dict['abs_angle'],goal_dict['dist'],self_dict['y_pos']),
              file=sys.stderr)
            sys.stderr.flush()
      else:
        goal_dict['y_pos'] = GOAL_BOTTOM_POS_Y
        if est_y_pos is not None:
          if (abs(est_y_pos-GOAL_BOTTOM_POS_Y) > POS_ERROR_TOLERANCE) and (goal_dict['dist'] < 10):
            print(
              "Est y_pos of bottom goalpost is {0:n} (should be {1:n}; from est abs_angle {2:n}, dist {3:n}, self_y_pos {4:n})".format(
                est_y_pos,GOAL_BOTTOM_POS_Y,goal_dict['abs_angle'],goal_dict['dist'],self_dict['y_pos']),
              file=sys.stderr)
            sys.stderr.flush()
      if (state[11] < 0) and (get_angle_diff(0.0, goal_dict['rel_angle'])
                              <= 1.0) and (get_dist_real(self_dict['x_pos'],
                                                         self_dict['y_pos'],
                                                         goal_dict['x_pos'],
                                                         goal_dict['y_pos']) < (0.15+0.03)):
        print(
          "Should be in collision with goal ({0:n}, {1:n} vs {2:n}, {3:n}; angle {4:n})".format(
            self_dict['x_pos'],self_dict['y_pos'],
            goal_dict['x_pos'],goal_dict['y_pos'],
            goal_dict['rel_angle']),
          file=sys.stderr)
        sys.stderr.flush()
    else:
      goal_dict['x_pos'] = GOAL_POS_X
      if state[18] > state[21]:
        goal_dict['y_pos'] = GOAL_TOP_POS_Y
      else:
        goal_dict['y_pos'] = GOAL_BOTTOM_POS_Y

  else:
    goal_dict['dist'] = get_dist_from_proximity(-1)
    goal_dict['rel_angle'] = None
    goal_dict['abs_angle'] = None
    goal_dict['x_pos'] = GOAL_POS_X
    if random.random() < 0.5:
      goal_dict['y_pos'] = GOAL_BOTTOM_POS_Y
    else:
      goal_dict['y_pos'] = GOAL_TOP_POS_Y

  ball_dict = {}
  if state[50] > 0: # ball position valid
    ball_dict['dist'] = get_dist_from_proximity(state[53])
    ball_dict['rel_angle'] = get_angle(state[51],state[52])
    if ((state[9] < 0)
        and (ball_dict['rel_angle']
             is not None)
        and (ball_dict['dist']
             < (0.15+0.0425))
        and (get_angle_diff(0.0, ball_dict['rel_angle'])
             <= 1.0)):
      raise RuntimeError(
        "Should be in collision with ball - distance is {0:n}".format(ball_dict['dist']))
    ball_dict['abs_angle'] = get_abs_angle(self_dict['body_angle'],ball_dict['rel_angle'])
    if ball_dict['abs_angle'] is not None:
      ball_dict['x_pos'], ball_dict['y_pos'] = get_abs_x_y_pos(
        abs_angle=ball_dict['abs_angle'],
        dist=ball_dict['dist'],
        self_x_pos=self_dict['x_pos'],
        self_y_pos=self_dict['y_pos'],
        warn=bool(ball_dict['dist'] < 10),
        of_what='Ball')
      if (ball_dict['x_pos'] is None) or (ball_dict['y_pos'] is None):
        if (ball_dict['dist'] < 10):
          print("Unknown ball position despite state[50] > 0", file=sys.stderr)
          sys.stderr.flush()
      else:
        if ((state[9] < 0)
            and (get_angle_diff(0.0, ball_dict['rel_angle'])
                 <= 1.0)
            and (get_dist_real(self_dict['x_pos'],self_dict['y_pos'],
                               ball_dict['x_pos'],ball_dict['y_pos']) < (0.15+0.0425))):
          raise RuntimeError(
            "Should be in collision with ball ({0:n}, {1:n} vs {2:n}, {3:n}; rel_angle {4:n})".format(
              self_dict['x_pos'],self_dict['y_pos'],
              ball_dict['x_pos'],ball_dict['y_pos'],
              ball_dict['rel_angle']))
  else:
    ball_dict['dist'] = get_dist_from_proximity(-1.0)
    ball_dict['rel_angle'] = None
    ball_dict['abs_angle'] = None
    ball_dict['x_pos'] = None
    ball_dict['y_pos'] = None

  if (ball_dict['x_pos'] is None) and (abs(namespace.other_dict['ball_x_pos'].value) < 1.0):
    ball_dict['x_pos'] = namespace.other_dict['ball_x_pos'].value
  if (ball_dict['y_pos'] is None) and (abs(namespace.other_dict['ball_y_pos'].value) < 1.0):
    ball_dict['y_pos'] = namespace.other_dict['ball_y_pos'].value

  if state[54] > 0: # ball velocity valid
    ball_dict['vel_rel_angle'] = get_angle(state[55],state[56])
    ball_dict['vel_mag'] = state[57]
  else:
    ball_dict['vel_rel_angle'] = None
    ball_dict['vel_mag'] = -1.0

  return (bit_list, self_dict, goal_dict, ball_dict)


def get_min_dist(self_dict, x_pos, y_pos):
  """Figure up most likely distance from target landmark."""
  landmark_dists = {}

  for landmark, pos in iteritems(landmark_start_to_location):
    if ((self_dict['x_pos_from_' + str(landmark)] is not None) and
        (self_dict['y_pos_from_' + str(landmark)] is not None)):
      landmark_dists[landmark] = get_dist_real(pos[0],pos[1],
                                               x_pos, y_pos)

  if landmark_dists:
    min_landmark_dist = min(itervalues(landmark_dists))
    if min_landmark_dist < 40.0:
      min_dist_landmarks = [x for x in iterkeys(landmark_dists)
                            if landmark_dists[x] <= min_landmark_dist]
    else:
      min_dist_landmarks = [x for x in iterkeys(landmark_dists)
                            if landmark_dists[x] <= min_landmark_dist] + ['OTHER','OOB']
  else:
    min_dist_landmarks = ['OTHER','OOB']

  min_dist = MAX_DIST

  for landmark in min_dist_landmarks:
    if ((self_dict['x_pos_from_' + str(landmark)] is not None) and
        (self_dict['y_pos_from_' + str(landmark)] is not None)):
      min_dist = min(min_dist,
                     get_dist_real(self_dict['x_pos_from_' + str(landmark)],
                                   self_dict['y_pos_from_' + str(landmark)],
                                   x_pos, y_pos, silent=True))
  return min_dist

def do_target_only(hfo_env,
                   state,
                   bit_list,
                   self_dict,
                   namespace):
  """Figure out and do next action if doing TARGET_ONLY."""

  if not bit_list[0]:
    hfo_env.act(hfo.REORIENT)
    return

  if ((state[7] < min(random.uniform(0.2,0.6),namespace.prestate_self_dict['stamina'])) or
      (max(state[7],namespace.prestate_self_dict['stamina']) <= random.uniform(-1.0,0.2))):
    if state[7] < 0.2:
      print("Attempting to rest; stamina {0!r} (previous {1!r})".format(
        state[7], namespace.prestate_self_dict['stamina']))
    if not bit_list[5]:
      hfo_env.act(hfo.REORIENT)
    else:
      hfo_env.act(hfo.TURN,0.0)
    namespace.prestate_self_dict['stamina'] = state[7]
    return

  namespace.prestate_self_dict['stamina'] = state[7]

  if ((namespace.location_dict['target_x_pos'] is None)
      or any([bit_list[2],bit_list[3],bit_list[7]])): # latter - collision
    target_x_pos, target_y_pos = random.choice(TARGET_POS)
    namespace.location_dict['target_x_pos'] = target_x_pos
    namespace.location_dict['target_y_pos'] = target_y_pos
    namespace.location_dict['prev_dist'] = get_min_dist(self_dict,
                                                        target_x_pos,
                                                        target_y_pos)
    hfo_env.act(hfo.MOVE_TO,target_x_pos,target_y_pos)
    namespace.location_dict['maybe_at'] = False
    return

  current_dist = get_min_dist(self_dict,
                              namespace.location_dict['target_x_pos'],
                              namespace.location_dict['target_y_pos'])
  if (namespace.location_dict['maybe_at']
      and ((current_dist <= ERROR_TOLERANCE)
           or (current_dist >= namespace.location_dict['prev_dist']))):
    if current_dist <= REAL_POS_ERROR_TOLERANCE:
      for landmark in list(iterkeys(landmark_start_to_location)) + ['OTHER','OOB']:
        x_pos_from = 'x_pos_from_' + str(landmark)
        x_pos_from_abs = 'x_pos_from_abs_' + str(landmark)
        x_pos_from_dist = 'x_pos_from_dist_' + str(landmark)
        if self_dict[x_pos_from] is not None:
          namespace.data2_dict[x_pos_from].append(self_dict[x_pos_from])
          namespace.data2_dict[x_pos_from_abs].append(
            abs(namespace.location_dict['target_x_pos'] - self_dict[x_pos_from]))
          namespace.data2_dict[x_pos_from_dist].append(self_dict[x_pos_from_dist])
        else:
          namespace.data2_dict[x_pos_from].append(self_dict['x_pos'])
          namespace.data2_dict[x_pos_from_abs].append(
            abs(namespace.location_dict['target_x_pos'] - self_dict['x_pos']))
          if x_pos_from_dist in self_dict:
            namespace.data2_dict[x_pos_from_dist].append(self_dict[x_pos_from_dist])
          else:
            namespace.data2_dict[x_pos_from_dist].append(MAX_DIST)
        y_pos_from = 'y_pos_from_' + str(landmark)
        y_pos_from_abs = 'y_pos_from_abs_' + str(landmark)
        y_pos_from_dist = 'y_pos_from_dist_' + str(landmark)
        if self_dict[y_pos_from] is not None:
          namespace.data2_dict[y_pos_from].append(self_dict[y_pos_from])
          namespace.data2_dict[y_pos_from_abs].append(
            abs(namespace.location_dict['target_y_pos'] - self_dict[y_pos_from]))
          namespace.data2_dict[y_pos_from_dist].append(self_dict[y_pos_from_dist])
        else:
          namespace.data2_dict[y_pos_from].append(self_dict['y_pos'])
          namespace.data2_dict[y_pos_from_abs].append(
            abs(namespace.location_dict['target_y_pos'] - self_dict['y_pos']))
          if y_pos_from_dist in self_dict:
            namespace.data2_dict[y_pos_from_dist].append(self_dict[y_pos_from_dist])
          else:
            namespace.data2_dict[y_pos_from_dist].append(MAX_DIST)
      namespace.data2_dict['target_x_pos'].append(
        namespace.location_dict['target_x_pos'])
      namespace.data2_dict['target_y_pos'].append(
        namespace.location_dict['target_y_pos'])
      namespace.data2_dict['final_dist'].append(current_dist)
    else:
      print(
        "Could not get closer than {0!r} to target landmark at {1:n}, {2:n}".format(
          current_dist, namespace.location_dict['target_x_pos'],
          namespace.location_dict['target_y_pos']) +
        " (real position {0:n}, {1:n})".format(
          get_x_unnormalized(namespace.location_dict['target_x_pos'],
                             allow_outside=True),
          get_y_unnormalized(namespace.location_dict['target_y_pos'],
                             allow_outside=True)),
        file=sys.stderr)
      sys.stderr.flush()
    target_x_pos = namespace.location_dict['target_x_pos']
    target_y_pos = namespace.location_dict['target_y_pos']
    while ((target_x_pos ==
            namespace.location_dict['target_x_pos']) and
           (target_x_pos ==
            namespace.location_dict['target_y_pos'])):
      target_x_pos, target_y_pos = random.choice(TARGET_POS)
    namespace.location_dict['target_x_pos'] = target_x_pos
    namespace.location_dict['target_y_pos'] = target_y_pos
    if current_dist <= REAL_POS_ERROR_TOLERANCE:
      namespace.location_dict['prev_dist'] = get_dist_real(namespace.data2_dict['target_x_pos'][-1],
                                                           namespace.data2_dict['target_x_pos'][-1],
                                                           target_x_pos,
                                                           target_y_pos)
    else:
      namespace.location_dict['prev_dist'] = get_min_dist(self_dict,
                                                          target_x_pos,
                                                          target_y_pos)
    namespace.location_dict['maybe_at'] = False
  else:
    namespace.location_dict['prev_dist'] = current_dist
    namespace.location_dict['maybe_at'] = True

  hfo_env.act(hfo.MOVE_TO,
              namespace.location_dict['target_x_pos'],
              namespace.location_dict['target_y_pos'])
  return

def do_next_action(hfo_env,
                   state,
                   namespace):
  """Figure out what action to take next, and do it."""
  bit_list, self_dict, ignored_goal_dict, ignored_ball_dict = filter_low_level_state(state, namespace)

  if not bit_list[1]: # self velocity
    hfo_env.act(hfo.REORIENT)
    return

  if TARGET_ONLY:
    return do_target_only(hfo_env,
                          state,
                          bit_list,
                          self_dict,
                          namespace)

  poss_actions_set = set([hfo.DASH,
                          hfo.MOVE_TO, hfo.INTERCEPT, hfo.DRIBBLE_TO,
                          hfo.MOVE, hfo.GO_TO_BALL])

  if not bit_list[0]: # self location
    poss_actions_set.discard(hfo.MOVE_TO)
    poss_actions_set.discard(hfo.MOVE)
    poss_actions_set.discard(hfo.GO_TO_BALL)
  if bit_list[4] or (not bit_list[5]):
    poss_actions_set.discard(hfo.INTERCEPT)
    poss_actions_set.discard(hfo.MOVE)
    poss_actions_set.discard(hfo.GO_TO_BALL)
  if not (bit_list[0] and bit_list[5]):
    poss_actions_set.discard(hfo.DRIBBLE_TO)

  if not poss_actions_set: # should only happen if DASH is removed from the above
    hfo_env.act(hfo.REORIENT)
    return

  action = random.choice(list(poss_actions_set))

  if action in (hfo.INTERCEPT, hfo.MOVE, hfo.GO_TO_BALL):
    hfo_env.act(action)
  elif action == hfo.DASH:
    hfo_env.act(hfo.DASH,random.uniform(80,100),random.uniform(-90,90))
  elif action == hfo.DRIBBLE_TO:
    r = random.random()
    if r < 0.5:
      hfo_env.act(action,
                  random.uniform(MIN_POS_X_BALL_SAFE,0.0),
                  random.uniform(MIN_POS_Y_BALL_SAFE,MAX_POS_Y_BALL_SAFE))
    elif r < 0.75:
      hfo_env.act(action,
                  random.uniform(0.0,MAX_POS_X_BALL_SAFE),
                  random.uniform(MIN_POS_Y_BALL_SAFE,-0.5))
    else:
      hfo_env.act(action,
                  random.uniform(0.0,MAX_POS_X_BALL_SAFE),
                  random.uniform(0.5,MAX_POS_Y_BALL_SAFE))
  elif action == hfo.MOVE_TO:
    hfo_env.act(action,
                random.uniform(MIN_POS_X_OK,MAX_POS_X_OK),
                random.uniform(MIN_POS_Y_OK,MAX_POS_Y_OK))
  else:
    raise RuntimeError(
      "Unknown action # {0!r}".format(action))


def run_player2(conf_dir, args, namespace):
  """Handle actions of second player (with high-level feature set)."""
  hfo_env2 = hfo.HFOEnvironment()

  if args.seed:
    random.seed(args.seed+1)

  connect2_args = [hfo.HIGH_LEVEL_FEATURE_SET, conf_dir, args.port, "localhost",
                   "base_left", False]

  if args.record:
    connect2_args.append(record_dir=args.rdir)

  time.sleep(5)

  print("Player2 connecting")
  sys.stdout.flush()

  hfo_env2.connectToServer(*connect2_args)

  print("Player2 unum is {0:d}".format(hfo_env2.getUnum()))
  sys.stdout.flush()
  namespace.other_dict['unum'].value = hfo_env2.getUnum()

  status=hfo.IN_GAME
  while status != hfo.SERVER_DOWN:
    did_reorient = False
    while status == hfo.IN_GAME:
      state2 = hfo_env2.getState()
      namespace.other_dict['x_pos'].value = state2[0]
      namespace.other_dict['y_pos'].value = state2[1]
      namespace.other_dict['ball_x_pos'].value = state2[3]
      namespace.other_dict['ball_y_pos'].value = state2[4]
      namespace.other_dict['self_x_pos'].value = state2[13]
      namespace.other_dict['self_y_pos'].value = state2[14]

      if (abs(state2[0])
          < 1) and (abs(state2[1])
                     < 1):
        if get_dist_normalized(0.0,
                               0.0,
                               state2[0],
                               state2[1]) > random.uniform(ERROR_TOLERANCE,
                                                           POS_ERROR_TOLERANCE):
          hfo_env2.act(hfo.MOVE_TO,0.0,0.0)
          did_reorient = False
        else:
          if random.random() < 0.5:
            hfo_env2.act(hfo.REORIENT)
            did_reorient = True
          else:
            hfo_env2.act(hfo.TURN,random.uniform(-180,180))
            did_reorient = False
      elif did_reorient and (abs(state2[0]) <= 1) and (abs(state2[1] <= 1)):
        hfo_env2.act(hfo.MOVE_TO,0.0,0.0)
        did_reorient = False
      else:
        hfo_env2.act(hfo.REORIENT)
        did_reorient = True

      status = hfo_env2.step()

    hfo_env2.act(hfo.NOOP)
    status = hfo_env2.step()

  hfo_env2.act(hfo.QUIT)
  sys.exit(0)


def main_explore_states_and_actions():
  """Gather information on high-level vs low-level states and actions."""
  if TARGET_ONLY: # problems with stamina if trials are too long
    default_trials = 20
    default_frames = 2000
  else:
    default_trials = 2
    default_frames = 20000
  parser = argparse.ArgumentParser()
  parser.add_argument('--port', type=int, default=6000,
                      help="Server port")
  parser.add_argument('--seed', type=int, default=None,
                      help="Python randomization seed; uses python default if 0 or not given")
  parser.add_argument('--server-seed', type=int, default=None,
                      help="Server randomization seed; uses default if 0 or not given")
  parser.add_argument('--record', default=False, action='store_true',
                      help="Doing HFO --record")
  parser.add_argument('--rdir', type=str, default='log/',
                      help="Set directory to use if doing HFO --record")
  parser.add_argument('--trials', type=int, default=default_trials,
                      help="Number of trials")
  parser.add_argument('--frames', type=int, default=default_frames,
                      help="Number of frames per trial")
  args=parser.parse_args()

  namespace = argparse.Namespace()

  namespace.other_dict = {'unum': multiprocessing.sharedctypes.Value('B', 0)}
  for name in ['x_pos', 'y_pos', 'ball_x_pos', 'ball_y_pos', 'self_x_pos', 'self_y_pos']:
    namespace.other_dict[name] = multiprocessing.sharedctypes.Value('d', -2.0)

  namespace.data_dict = {} # used if not doing TARGET_ONLY
  for landmark in list(iterkeys(landmark_start_to_location)) + ['OTHER']:
    namespace.data_dict[landmark] = {}
    namespace.data_dict[landmark]['dist'] = []
    namespace.data_dict[landmark]['x_pos'] = []
    namespace.data_dict[landmark]['y_pos'] = []
    namespace.data_dict[landmark]['off'] = []
    namespace.data_dict[landmark]['off_orig'] = []
    namespace.data_dict[landmark]['abs_off_orig'] = []
    namespace.data_dict[landmark]['dist_strict'] = []

  namespace.data2_dict = {} # used with TARGET_ONLY
  for landmark in list(iterkeys(landmark_start_to_location)) + ['OTHER','OOB']:
    namespace.data2_dict['x_pos_from_' + str(landmark)] = []
    namespace.data2_dict['x_pos_from_abs_' + str(landmark)] = []
    namespace.data2_dict['x_pos_from_dist_' + str(landmark)] = []
    namespace.data2_dict['y_pos_from_' + str(landmark)] = []
    namespace.data2_dict['y_pos_from_abs_' + str(landmark)] = []
    namespace.data2_dict['y_pos_from_dist_' + str(landmark)] = []
  namespace.data2_dict['target_x_pos'] = []
  namespace.data2_dict['target_y_pos'] = []
  namespace.data2_dict['final_dist'] = []

  script_dir   = os.path.dirname(os.path.abspath(os.path.realpath(__file__)))
  binary_dir = os.path.normpath(script_dir + "/../bin")
  conf_dir = os.path.join(binary_dir, 'teams/base/config/formations-dt')
  bin_HFO = os.path.join(binary_dir, "HFO")

  player2_process = multiprocessing.Process(target=run_player2,
                                            kwargs={'conf_dir': conf_dir,
                                                    'args': args,
                                                    'namespace': namespace}
                                            )

  popen_list = [sys.executable, "-x", bin_HFO,
                "--frames-per-trial={0:d}".format(args.frames),
                "--untouched-time={0:d}".format(args.frames),
                "--port={0:d}".format(args.port),
                "--offense-agents=2", "--defense-npcs=0",
                "--offense-npcs=0", "--trials={0:d}".format(args.trials),
                "--headless"]

  if FULLSTATE:
    popen_list.append("--fullstate")

  if args.record:
    popen_list.append("--record")
  if args.server_seed:
    popen_list.append("--seed={0:d}".format(args.server_seed))

  HFO_subprocess = subprocess.Popen(popen_list)


  player2_process.daemon = True
  player2_process.start()

  time.sleep(0.2)

  assert (HFO_subprocess.poll() is
          None), "Failed to start HFO with command '{}'".format(" ".join(popen_list))
  assert player2_process.is_alive(), "Failed to start player2 process (exit code {!r})".format(
    player2_process.exitcode)

  if args.seed:
    random.seed(args.seed)

  hfo_env = hfo.HFOEnvironment()

  connect_args = [hfo.LOW_LEVEL_FEATURE_SET, conf_dir, args.port, "localhost",
                  "base_left", False]


  if args.record:
    connect_args.append(record_dir=args.rdir)

  time.sleep(9.8)

  try:
    assert player2_process.is_alive(), "Player2 process exited (exit code {!r})".format(
      player2_process.exitcode)

    print("Main player connecting")
    sys.stdout.flush()

    hfo_env.connectToServer(*connect_args)

    print("Main player unum {0:d}".format(hfo_env.getUnum()))
    sys.stdout.flush()

    for ignored_episode in itertools.count():
      status = hfo.IN_GAME
      namespace.prestate_bit_list = [0,0,0,0,0,0,0]
      namespace.prestate_self_dict = {'x_pos': None,
                                      'y_pos': None,
                                      'x_pos_strict': None,
                                      'body_angle': None,
                                      'other_angle': None,
                                      'stamina': 1.0,
                                      'vel_rel_angle': None,
                                      'vel_mag': -1.0}
      namespace.prestate_goal_dict = {'dist': get_dist_from_proximity(-1),
                                      'rel_angle': None,
                                      'abs_angle': None,
                                      'x_pos': GOAL_POS_X,
                                      'y_pos': 0.0}
      namespace.prestate_ball_dict = {'dist': get_dist_from_proximity(-1),
                                      'rel_angle': None,
                                      'abs_angle': None,
                                      'x_pos': None,
                                      'y_pos': None,
                                      'vel_rel_angle': None,
                                      'vel_mag': -1.0}
      namespace.location_dict = {'prev_dist': get_dist_from_proximity(-1),
                                 'target_x_pos': None,
                                 'target_y_pos': None,
                                 'maybe_at': False}

      while status == hfo.IN_GAME:
        state = hfo_env.getState()

        do_next_action(hfo_env, state, namespace)
        status = hfo_env.step()

      if status == hfo.SERVER_DOWN:
        hfo_env.act(hfo.QUIT)
        break
  finally:
    if HFO_subprocess.poll() is None:
      HFO_subprocess.terminate()
    os.system("killall -9 rcssserver") # remove if ever start doing multi-server testing!

  # below requires scipy + removal of some commented-out lines above
  # first section is for TARGET_ONLY;
  # second section is for not (may not work right now - most recent testing used TARGET_ONLY)

  if TARGET_ONLY:
    print("Have {0:n} data points for targeting".format(len(namespace.data2_dict['target_x_pos'])))
    for pos_from in sorted(list(iterkeys(landmark_start_to_location)) + ['OOB']):
      if pos_from in landmark_start_to_location:
        print("Landmark: {0:n}; position: {1:n}, {2:n}".format(pos_from,
                                                               landmark_start_to_location[pos_from][0],
                                                               landmark_start_to_location[pos_from][1]))
      for x_vs_y in ['x_pos_from_','x_pos_from_abs_','y_pos_from_','y_pos_from_abs_']:
        y_type = x_vs_y + str(pos_from)
        data_description = stats.describe(namespace.data2_dict[y_type])
        print(" Data from: {0} ({1!r})".format(y_type,data_description))
        if 'x_pos_from_' in y_type:
          if 'abs' in y_type:
            x_type_list = ['x_pos_from_dist_' + str(pos_from)]
          else:
            x_type_list = ['target_x_pos']
        else:
          if 'abs' in y_type:
            x_type_list = ['y_pos_from_dist_' + str(pos_from)]
          else:
            x_type_list = ['target_y_pos']
        for x_type in x_type_list:
          if x_type == y_type:
            continue
          results = stats.linregress(namespace.data2_dict[x_type],
                                     namespace.data2_dict[y_type])
          if math.pow(results[2],2.0) > 0.5:
            print("  ***Linregress of {0} vs {1}: {2!r}".format(x_type,y_type,results))
          elif (math.pow(results[2],2.0) > (2/5)) and (results[3] < 0.5):
            print("  !!!Linregress of {0} vs {1}: {2!r}".format(x_type,y_type,results))
          else:
            print("  Linregress of {0} vs {1}: {2!r}".format(x_type,y_type,results))
          if (ERROR_TOLERANCE < abs(results[0]-1.0) < 0.05):
            slope1, intercept1, low, high, ilow, ihigh = do_theilsen(namespace.data2_dict[x_type],
                                                                     namespace.data2_dict[y_type])
            print("  Theilsen of {0} vs {1}: slope {2!r} ({3:n}/{4:n}), intercept {5!r} ({6:n}/{7:n})".format(
              x_type,y_type,slope1,low,high,intercept1,ilow,ihigh))

##  for landmark in sorted(iterkeys(namespace.data_dict)):
##    if landmark == 'OTHER':
##      print("Landmark: OTHER; position? 0.0,0.0; data points {0:n}".format(len(namespace.data_dict[landmark]['off'])))
##    else:
##      print("Landmark: {0:n}; position: {1:n}, {2:n}; data points {3:n}".format(landmark,landmark_start_to_location[landmark][0],
##                                                                                landmark_start_to_location[landmark][1],
##                                                                                len(namespace.data_dict[landmark]['off'])))
##    for x_type in ['dist','dist_strict','x_pos','y_pos']:
##      for y_type in ['off','off_orig','abs_off_orig','dist_strict']:
##        if x_type == y_type:
##          continue
##        results = stats.linregress(namespace.data_dict[landmark][x_type],
##                                   namespace.data_dict[landmark][y_type])
##        print(" Linregress of {0} vs {1}: {2!r}".format(x_type,y_type,results))
##        if (0.05 < math.pow(results[2],2.0) < (1/3)):
##          slope1, intercept1, low, high = do_theilsen(namespace.data_dict[landmark][x_type],
##                                                      namespace.data_dict[landmark][y_type])
##          print(" Theilsen of {0} vs {1}: slope {2!r} ({3:n}/{4:n}), intercept {5!r}".format(
##            x_type,y_type,slope1,low,high,intercept1))

if __name__ == '__main__':
  main_explore_states_and_actions()
