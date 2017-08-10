from ctypes import *
import math
import os
import sys
import warnings

import numpy as np
from numpy.ctypeslib import as_ctypes

hfo_lib = cdll.LoadLibrary(os.path.join(os.path.dirname(__file__),
                                        'libhfo_c.so'))

"""Possible feature sets"""
NUM_FEATURE_SETS = 2
LOW_LEVEL_FEATURE_SET, HIGH_LEVEL_FEATURE_SET = list(range(NUM_FEATURE_SETS))

"""
An enum of the possible HFO actions, including:
  [Low-Level] Dash(power, relative_direction)
  [Low-Level] Turn(direction)
  [Low-Level] Tackle(direction)
  [Low-Level] Kick(power, direction)
  [Mid-Level] Kick_To(target_x, target_y, speed)
  [Mid-Level] Move(target_x, target_y)
  [Mid-Level] Dribble(target_x, target_y)
  [Mid-Level] Intercept(): Intercept the ball
  [High-Level] Move(): Reposition player according to strategy
  [High-Level] Shoot(): Shoot the ball
  [High-Level] Pass(teammate_unum): Pass to teammate
  [High-Level] Dribble(): Offensive dribble
  [High-Level] Catch(): Catch the ball (Goalie Only)
  NOOP(): Do Nothing
  QUIT(): Quit the game
"""
NUM_HFO_ACTIONS = 20
DASH,TURN,TACKLE,KICK,KICK_TO,MOVE_TO,DRIBBLE_TO,INTERCEPT,MOVE,SHOOT,PASS,DRIBBLE,CATCH,NOOP,QUIT,REDUCE_ANGLE_TO_GOAL,MARK_PLAYER,DEFEND_GOAL,GO_TO_BALL,REORIENT = list(range(NUM_HFO_ACTIONS))
ACTION_STRINGS = {DASH: "Dash",
                  TURN: "Turn",
                  TACKLE: "Tackle",
                  KICK: "Kick",
                  KICK_TO: "KickTo",
                  MOVE_TO: "MoveTo",
                  DRIBBLE_TO: "DribbleTo",
                  INTERCEPT: "Intercept",
                  MOVE: "Move",
                  SHOOT: "Shoot",
                  PASS: "Pass",
                  DRIBBLE: "Dribble",
                  CATCH: "Catch",
                  NOOP: "No-op",
                  QUIT: "Quit",
                  REDUCE_ANGLE_TO_GOAL: "Reduce_Angle_To_Goal",
                  MARK_PLAYER: "Mark_Player",
                  DEFEND_GOAL: "Defend_Goal",
                  GO_TO_BALL: "Go_To_Ball",
                  REORIENT: "Reorient"}

"""
Possible game statuses:
  [IN_GAME] Game is currently active
  [GOAL] A goal has been scored by the offense
  [CAPTURED_BY_DEFENSE] The defense has captured the ball
  [OUT_OF_BOUNDS] Ball has gone out of bounds
  [OUT_OF_TIME] Trial has ended due to time limit
  [SERVER_DOWN] Server is not alive
"""
NUM_GAME_STATUS_STATES = 6
IN_GAME, GOAL, CAPTURED_BY_DEFENSE, OUT_OF_BOUNDS, OUT_OF_TIME, SERVER_DOWN = list(range(NUM_GAME_STATUS_STATES))
STATUS_STRINGS = {IN_GAME: "InGame",
                  GOAL: "Goal",
                  CAPTURED_BY_DEFENSE: "CapturedByDefense",
                  OUT_OF_BOUNDS: "OutOfBounds",
                  OUT_OF_TIME: "OutOfTime",
                  SERVER_DOWN: "ServerDown"}

"""Possible action result statuses."""
ACTION_STATUS_UNKNOWN, ACTION_STATUS_BAD, ACTION_STATUS_MAYBE = list(range(-1,2))
ACTION_STATUS_MAYBE_OK = ACTION_STATUS_MAYBE # typos
ACTION_STATUS_STRINGS = {ACTION_STATUS_UNKNOWN: "Unknown",
                         ACTION_STATUS_BAD: "Bad",
                         ACTION_STATUS_MAYBE: "MaybeOK"}

"""Possible sides."""
RIGHT, NEUTRAL, LEFT = list(range(-1,2))

class Player(Structure):pass
Player._fields_ = [
    ('side', c_int),
    ('unum', c_int),
]

hfo_lib.HFO_new.argtypes = None
hfo_lib.HFO_new.restype = c_void_p
hfo_lib.HFO_del.argtypes = [c_void_p]
hfo_lib.HFO_del.restype = None
hfo_lib.connectToServer.argtypes = [c_void_p, c_int, c_char_p, c_int,
                                    c_char_p, c_char_p, c_bool, c_char_p]
hfo_lib.connectToServer.restype = None
hfo_lib.getStateSize.argtypes = [c_void_p]
hfo_lib.getStateSize.restype = c_int
hfo_lib.getState.argtypes = [c_void_p, c_void_p]
hfo_lib.getState.restype = None
hfo_lib.act.argtypes = [c_void_p, c_int, c_void_p]
hfo_lib.act.restype = None
hfo_lib.say.argtypes = [c_void_p, c_char_p]
hfo_lib.say.restype = None
hfo_lib.hear.argtypes = [c_void_p]
hfo_lib.hear.restype = c_char_p
hfo_lib.playerOnBall.argtypes = [c_void_p]
hfo_lib.playerOnBall.restype = Player
hfo_lib.step.argtypes = [c_void_p]
hfo_lib.step.restype = c_int
hfo_lib.numParams.argtypes = [c_int]
hfo_lib.numParams.restype = c_int
hfo_lib.getUnum.argtypes = [c_void_p]
hfo_lib.getUnum.restype = c_int
hfo_lib.getNumTeammates.argtypes = [c_void_p]
hfo_lib.getNumTeammates.restype = c_int
hfo_lib.getNumOpponents.argtypes = [c_void_p]
hfo_lib.getNumOpponents.restype = c_int
hfo_lib.getLastActionStatus.argtypes = [c_void_p, c_int]
hfo_lib.getLastActionStatus.restype = c_int

class HFOEnvironment(object):
  def __init__(self):
    self.obj = hfo_lib.HFO_new()
    self.feature_set = None

  def __del__(self):
    hfo_lib.HFO_del(self.obj)

  def connectToServer(self,
                      feature_set=LOW_LEVEL_FEATURE_SET,
                      config_dir='bin/teams/base/config/formations-dt',
                      server_port=6000,
                      server_addr='localhost',
                      team_name='base_left',
                      play_goalie=False,
                      record_dir=''):
    """
      Connects to the server on the specified port. The
      following information is provided by the ./bin/HFO

      feature_set: High or low level state features
      config_dir: Config directory. Typically HFO/bin/teams/base/config/
      server_port: port to connect to server on
      server_addr: address of server
      team_name: Name of team to join.
      play_goalie: is this player the goalie
      record_dir: record agent's states/actions/rewards to this directory
    """
    hfo_lib.connectToServer(self.obj,
                            feature_set,
                            config_dir.encode('utf-8'),
                            server_port,server_addr.encode('utf-8'),
                            team_name.encode('utf-8'),
                            play_goalie,
                            record_dir.encode('utf-8'))
    self.feature_set = feature_set

  def getStateSize(self):
    """ Returns the number of state features """
    return hfo_lib.getStateSize(self.obj)

  def getState(self, state_data=None):
    """ Returns the current state features """
    if state_data is None:
      state_data = np.zeros(self.getStateSize(), dtype=np.float32)
    hfo_lib.getState(self.obj, as_ctypes(state_data))
    return state_data

  def act(self, action_type, *args):
    """ Performs an action in the environment """
    n_params = hfo_lib.numParams(action_type)
    assert n_params == len(args), 'Incorrect number of params to act: '\
      'Required %d, provided %d'%(n_params, len(args))
    params = np.asarray(args, dtype=np.float32)
    hfo_lib.act(self.obj, action_type, params.ctypes.data_as(POINTER(c_float)))

  def say(self, message):
    """ Transmits a message """
    hfo_lib.say(self.obj, message.encode('utf-8'))

  def hear(self):
    """ Returns the message heard from another player """
    return hfo_lib.hear(self.obj).decode('utf-8')

  def playerOnBall(self):
    """ Returns a player object who last touched the ball """
    return hfo_lib.playerOnBall(self.obj)

  def step(self):
    """ Advances the state of the environment """
    return hfo_lib.step(self.obj)

  @staticmethod
  def actionToString(action):
    """ Returns a string representation of an action """
    return ACTION_STRINGS[action]

  @staticmethod
  def statusToString(status):
    """ Returns a string representation of a game status """
    return STATUS_STRINGS[status]

  def getUnum(self):
    """ Returns the uniform number of the agent """
    return hfo_lib.getUnum(self.obj)

  def getNumTeammates(self):
    """ Returns the number of teammates of the agent """
    return hfo_lib.getNumTeammates(self.obj)

  def getNumOpponents(self):
    """ Returns the number of opponents of the agent """
    return hfo_lib.getNumOpponents(self.obj)

  def getLastActionStatus(self, last_action):
    """
    If last_action is the last action with a recorded status,
    returns ACTION_STATUS_MAYBE for possible success,
    ACTION_STATUS_BAD for no possibility of success,
    or ACTION_STATUS_UNKNOWN if unknown. If it is not the
    last action with a recorded status, returns ACTION_STATUS_UNKNOWN.
    """
    return hfo_lib.getLastActionStatus(self.obj, last_action)

  @staticmethod
  def actionStatusToString(status):
    """Returns a string representation of an action status."""
    return ACTION_STATUS_STRINGS[status]

  @staticmethod
  def _get_angle(sin_angle,cos_angle):
    if max(abs(sin_angle),abs(cos_angle)) <= sys.float_info.epsilon:
      return None
    return math.degrees(math.atan2(sin_angle,cos_angle))

  def _analyze_angle(self,sin_angle,cos_angle,name):
    angle_dict = {name + '_sin_angle': sin_angle,
                  name + '_cos_angle': cos_angle}
    angle_dict[name + '_angle_degrees'] = self._get_angle(sin_angle,cos_angle)
    angle_dict[name + '_angle_valid'] = bool(angle_dict[name + '_angle_degrees'] is not None)
    return angle_dict

  @staticmethod
  def _analyze_unum(raw_unum,should_be_valid):
    if raw_unum > 0:
      unum = raw_unum*100
      if unum < 1:
        unum = math.ceil(unum)
      elif unum > 11:
        unum = math.floor(unum)
      else:
        unum = round(unum,0)
      if 1 <= unum <= 11:
        unum = int(unum)
        if not should_be_valid:
          warnings.warn(
            "Unum {0:n} (from {1:n}) unexpectedly appears valid".format(unum,raw_unum))
        return unum
      elif should_be_valid:
        raise RuntimeError(
          "Unum {0:n} (from {1:n}) should be valid but is not".format(unum,raw_unum))

      return None

    if should_be_valid:
      raise RuntimeError(
        "Unum {0:n} should be valid but is not".format(raw_unum))
    return None

  @staticmethod
  def _check_state_len(state, length_needed):
    if len(state) < length_needed:
      raise RuntimeError(
        "State length {0:n} is too short; should be at least {1:n}".format(
          len(state),length_needed))
    elif len(state) > length_needed:
      warnings.warn("State is length {0:n}, but only know how to interpret {1:n} parts".format(
        len(state),length_needed))

  def _parse_low_level_state(self, state, num_teammates, num_opponents):
    length_needed = 58 + (9*num_teammates) + (9*num_opponents)
    self._check_state_len(state, length_needed)
    state_dict = {}
    state_dict['self_pos_valid'] = bool(state[0] > 0)
    state_dict['self_vel_valid'] = bool(state[1] > 0)
    state_dict.update(self._analyze_angle(state[2],state[3],'self_vel'))
    if state_dict['self_vel_valid'] != state_dict['self_vel_angle_valid']:
      raise RuntimeError(
        "Self velocity validity is {0:n} but sin/cos for angle are {1:n} and {2:n}".format(
          state[1],state[2],state[3]))
    if state_dict['self_vel_valid']:
      state_dict['self_vel_magnitude'] = state[4]
    else: # most likely 0 magnitude...
      state_dict['self_vel_magnitude'] = -1.0
    state_dict.update(self._analyze_angle(state[5],state[6],'self_body'))
    state_dict['stamina'] = state[7]
    state_dict['frozen'] = bool(state[8] > 0)
    state_dict['collides_ball'] = bool(state[9] > 0)
    state_dict['collides_player'] = bool(state[10] > 0)
    state_dict['collides_post'] = bool(state[11] > 0)
    state_dict['kickable'] = bool(state[12] > 0)
    state_dict.update(self._analyze_angle(state[13],state[14],'goal_center'))
    if state_dict['self_pos_valid'] != state_dict['goal_center_angle_valid']:
       raise RuntimeError(
         "Self position validity is {0:n} but sin/cos for goal_center_angle are {1:n} and {2:n}".format(
           state[0],state[13],state[14]))
    if state_dict['self_pos_valid']:
      state_dict['goal_center_proximity'] = state[15]
    elif state_dict['collides_post']:
      state_dict['goal_center_proximity'] = 1.0
    else:
      state_dict['goal_center_proximity'] = None
    state_dict.update(self._analyze_angle(state[16],state[17],'goal_top'))
    if state_dict['self_pos_valid'] != state_dict['goal_top_angle_valid']:
       raise RuntimeError(
         "Self position validity is {0:n} but sin/cos for goal_top_angle are {1:n} and {2:n}".format(
           state[0],state[16],state[17]))
    if state_dict['self_pos_valid']:
      state_dict['goal_top_proximity'] = state[18]
    elif state_dict['collides_post']:
      state_dict['goal_top_proximity'] = 1.0
    else:
      state_dict['goal_top_proximity'] = None
    state_dict.update(self._analyze_angle(state[19],state[20],'goal_bottom'))
    if state_dict['self_pos_valid'] != state_dict['goal_bottom_angle_valid']:
       raise RuntimeError(
         "Self position validity is {0:n} but sin/cos for goal_bottom_angle are {1:n} and {2:n}".format(
           state[0],state[19],state[20]))
    if state_dict['self_pos_valid']:
      state_dict['goal_bottom_proximity'] = state[21]
    elif state_dict['collides_post']:
      state_dict['goal_bottom_proximity'] = 1.0
    else:
      state_dict['goal_bottom_proximity'] = None

    landmark_nums = {'penalty_box_center': 22,
                     'penalty_box_top': 25,
                     'penalty_box_bottom': 28,
                     'center_field': 31,
                     'corner_top_left': 34,
                     'corner_top_right': 37,
                     'corner_bottom_right': 40,
                     'corner_bottom_left': 43}
    for name, num in landmark_nums:
      state_dict.update(self._analyze_angle(state[num],
                                            state[num+1],
                                            name))
      if state_dict['self_pos_valid'] != state_dict[name + '_angle_valid']:
        raise RuntimeError(
          "Self position validity is {0:n} but sin/cos for {1}_angle are {2:n} and {3:n}".format(
            state[0], name, state[num], state[num+1]))
      if state_dict['self_pos_valid']:
        state_dict[name + '_proximity'] = state[num+2]
      else:
        state_dict[name + '_proximity'] = None

    if state_dict['self_pos_valid']:
      state_dict['OOB_left_proximity'] = state[46]
      state_dict['OOB_right_proximity'] = state[47]
      state_dict['OOB_top_proximity'] = state[48]
      state_dict['OOB_bottom_proximity'] = state[49]
    else:
      if state_dict['collides_post']:
        state_dict['OOB_left_proximity'] = -1.0
        state_dict['OOB_right_proximity'] = 1.0
      else:
        state_dict['OOB_left_proximity'] = None
        state_dict['OOB_right_proximity'] = None
      state_dict['OOB_top_proximity'] = None
      state_dict['OOB_bottom_proximity'] = None

    state_dict['ball_pos_valid'] = bool(state[50] > 0)
    state_dict.update(self._analyze_angle(state[51],state[52],'ball_pos'))
    if state_dict['ball_pos_valid'] != state_dict['ball_pos_angle_valid']:
      raise RuntimeError(
        "Ball position validity is {0:n} but sin/cos for angle are {1:n} and {2:n}".format(
          state[50], state[51], state[52]))
    if state_dict['ball_pos_valid']:
      state_dict['ball_proximity'] = state[53]
    elif state_dict['collides_ball'] or state_dict['kickable']:
      state_dict['ball_proximity'] = 1.0
    else: # pessimize
      state_dict['ball_proximity'] = -1.0
    state_dict['ball_vel_valid'] = bool(state[54] > 0)
    if state_dict['ball_vel_valid']:
      state_dict['ball_vel_magnitude'] = state[55]
    else:
      state_dict['ball_vel_magnitude'] = None
    state_dict.update(self._analyze_angle(state[56],state[57],'ball_vel'))
    if state_dict['ball_vel_valid'] != state_dict['ball_vel_angle_valid']:
      raise RuntimeError(
        "Ball velocity validity is {0:n} but sin/cos for vel_angle are {1:n} and {2:n}".format(
          state[54],state[56],state[57]))

    state_dict['teammates_list'] = []

    for i in range(num_teammates):
      teammate_dict = {}
      teammate_dict.update(self._analyze_angle(state[58+(8*i)],state[58+(8*i)+1],'pos'))
      if teammate_dict['pos_angle_valid']:
        teammate_dict['proximity'] = state[58+(8*i)+2]
      elif state_dict['collides_player'] and ((num_teammates+num_opponents) == 1):
        teammate_dict['proximity'] =  1.0
      else:
        teammate_dict['proximity'] = None
      teammate_dict.update(self._analyze_angle(state[58+(8*i)+3],state[58+(8*i)+4],'body'))
      # doing vel_magnitude slightly later
      teammate_dict.update(self._analyze_angle(state[58+(8*i)+6],state[58+(8*i)+7],'vel'))
      teammate_dict['vel_valid'] = teammate_dict['vel_angle_valid']
      if teammate_dict['vel_valid'] and not teammate_dict['pos_angle_valid']:
        raise RuntimeError(
          "Teammate (by dist {0:n}) pos sin/cos are {1:n} and {2:n} but vel sin/cos are {3:n} and {4:n}".format(
            i, state[58+(8*i)], state[58+(8*i)+1], state[58+(8*i)+6], state[58+(8*i)+7]))
      if teammate_dict['vel_valid']:
        teammate_dict['vel_magnitude'] = state[58+(8*i)+5]
      else:
        teammate_dict['vel_magnitude'] = None
      teammate_dict['unum'] = self._analyze_unum(state[58+(8*num_teammates)+(8*num_opponents)+i],
                                                 teammate_dict['pos_angle_valid'])
      state_dict['teammates_list'].append(teammate_dict)

    state_dict['opponents_list'] = []

    for i in range(num_opponents):
      base_num = 58+(8*num_teammates)+i
      opponent_dict = {}
      opponent_dict.update(self._analyze_angle(state[base_num],state[base_num+1],'pos'))
      if opponent_dict['pos_angle_valid']:
        opponent_dict['proximity'] = state[base_num+2]
      elif state_dict['collides_player'] and ((num_teammates+num_opponents) == 1):
        opponent_dict['proximity'] = 1.0
      elif state_dict['frozen'] and (num_opponents == 1):
        opponent_dict['proximity'] = 1.0
      else:
        opponent_dict['proximity'] = None
      opponent_dict.update(self._analyze_angle(state[base_num+3],state[base_num+4],'body'))
      # doing vel_magnitude slightly later
      opponent_dict.update(self._analyze_angle(state[base_num+6],state[base_num+7],'vel'))
      opponent_dict['vel_valid'] = opponent_dict['vel_angle_valid']
      if opponent_dict['vel_valid'] and not opponent_dict['pos_angle_valid']:
        raise RuntimeError(
          "Opponent (by dist {0:n}) pos sin/cos are {1:n} and {2:n} but vel sin/cos are {3:n} and {4:n}".format(
            i, state[base_num], state[base_num+1], state[base_num+6], state[base_num+7]))
      if opponent_dict['vel_valid']:
        opponent_dict['vel_magnitude'] = state[base_num+5]
      else:
        opponent_dict['vel_magnitude'] = None
      opponent_dict['unum'] = self._analyze_unum(state[base_num+(8*num_opponents)+num_teammates+i],
                                                 opponent_dict['pos_angle_valid'])
      state_dict['opponents_list'].append(opponent_dict)

    return state_dict

  @staticmethod
  def _check_hl_feature(feature):
    if abs(feature) > 1:
      return None
    return feature

  def _parse_high_level_state(self, state, num_teammates, num_opponents):
    length_needed = 10 + (6*num_teammates) + (3*num_opponents)
    self._check_state_len(state, length_needed)
    state_dict = {}

    state_dict['x_pos'] = self._check_hl_feature(state[0])
    state_dict['y_pos'] = self._check_hl_feature(state[1])

    state_dict['body_angle'] = self._check_hl_feature(state[2])
    if state_dict['body_angle'] is not None:
      state_dict['body_angle_degrees'] = state_dict['body_angle']*180
    else:
      state_dict['body_angle_degrees'] = None

    state_dict['ball_x_pos'] = self._check_hl_feature(state[3])
    state_dict['ball_y_pos'] = self._check_hl_feature(state[4])

    state_dict['kickable'] = bool(state[5] > 0)

    state_dict['goal_dist'] = self._check_hl_feature(state[6])
    state_dict['goal_angle'] = self._check_hl_feature(state[7])
    if state_dict['goal_angle'] is not None:
      state_dict['goal_angle_degrees'] = state_dict['goal_angle']*180
    else:
      state_dict['goal_angle_degrees'] = None

    state_dict['goal_open_angle'] = self._check_hl_feature(state[8])
    if state_dict['goal_open_angle'] is not None:
      state_dict['goal_open_angle_degrees'] = (state_dict['goal_open_angle']+1)*90
    else:
      state_dict['goal_open_angle_degrees'] = None

    state_dict['closest_opponent_dist'] = self._check_hl_feature(state[9])

    state_dict['teammates_list'] = []

    for i in range(num_teammates):
      teammate_dict = {}
      teammate_dict['goal_open_angle'] = self._check_hl_feature(state[10+i])
      if teammate_dict['goal_open_angle'] is not None:
        teammate_dict['goal_open_angle_degrees'] = (teammate_dict['goal_open_angle']+1)*90
      else:
        teammate_dict['goal_open_angle_degrees'] = None
      teammate_dict['closest_opponent_dist'] = self._check_hl_feature(state[10+num_teammates+i])
      teammate_dict['pass_open_angle'] = self.check_hl_feature(state[10+(2*num_teammates)+i])
      if teammate_dict['pass_open_angle'] is not None:
        teammate_dict['pass_open_angle_degrees'] = (teammate_dict['pass_open_angle']+1)*90
      else:
        teammate_dict['pass_open_angle_degrees'] = None
      teammate_dict['x_pos'] = self._check_hl_feature(state[10+(3*num_teammates)+(3*i)])
      teammate_dict['y_pos'] = self._check_hl_feature(state[10+(3*num_teammates)+(3*i)+1])
      if 1 <= state[10+(3*num_teammates)+(3*i)+2] <= 11:
        teammate_dict['unum'] = state[10+(3*num_teammates)+(3*i)+2]
      else:
        teammate_dict['unum'] = None
      state_dict['teammates_list'].append(teammate_dict)

    state_dict['opponents_list'] = []

    for i in range(num_opponents):
      opponent_dict = {}
      opponent_dict['x_pos'] = self._check_hl_feature(state[10+(6*num_teammates)+(3*i)])
      opponent_dict['y_pos'] = self._check_hl_feature(state[10+(6*num_teammates)+(3*i)+1])
      if 1 <= state[10+(6*num_teammates)+(3*i)+2] <= 11:
        opponent_dict['unum'] = state[10+(6*num_teammates)+(3*i)+2]
      else:
        opponent_dict['unum'] = None
      state_dict['opponents_list'].append(opponent_dict)

    return state_dict

  def parse_state(self, state, feature_set=None):
    if feature_set is None:
      feature_set = self.feature_set
    num_teammates = self.getNumTeammates()
    num_opponents = self.getNumOpponents()

    if feature_set == LOW_LEVEL_FEATURE_SET:
      return self._parse_low_level_state(state, num_teammates, num_opponents)
    elif feature_set == HIGH_LEVEL_FEATURE_SET:
      return self._parse_high_level_state(state, num_teammates, num_opponents)
    else:
      raise RuntimeError("Unknown feature_set {!r}".format(feature_set))
