"""Half Field Offense in 2D RoboCup Soccer"""
from ctypes import cdll, Structure, c_void_p, c_int, c_char_p, c_bool, POINTER, c_float
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
  [Mid-Level?] Intercept(): Intercept the ball
  [High-Level] Move(): Reposition player according to strategy
  [High-Level] Shoot(): Shoot the ball
  [High-Level] Pass(teammate_unum): Pass to teammate
  [High-Level] Dribble(): Offensive dribble
  [High-Level] Catch(): Catch the ball (Goalie Only)
  [High-Level] Reduce_Angle_To_Goal(): Cover biggest open angle to goal, moving to between goal and opponents
  [High-Level] Mark_Player(opponent_unum): Go between kicker and marked player
  [High-Level] Defend_Goal(): Move along line between goalposts to cover goal
  [High-Level?] Go_To_Ball(): Go directly to the ball
  [High-Level] Reorient(): Pay close attention to surroundings, in particular dealing with loss of self/ball information
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

class HFOEnvironment(object):
  def __init__(self):
    self.obj = hfo_lib.HFO_new()
    self.feature_set = None
    self.playing_offense = None

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
    self.playing_offense = team_name.endswith("left")

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

# Utility functions - may move to separate file

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

  @staticmethod
  def _compare_valid_angle(first_valid, second_valid,
                           validity, sin_angle, cos_angle,
                           what1, what2):
    if first_valid != second_valid:
      raise RuntimeError(
        "{w1} validity is {valid:n} but sin/cos for {w2} are {sin:n} and {cos:n}".format(
          w1=what1, valid=validity, w2=what2, sin=sin_angle, cos=cos_angle))

  def _get_player_stats(self, state, state_dict, num_teammates, num_opponents,
                        i, base_num, is_teammate, strict, unum_num):
    player_dict = {}
    player_dict.update(self._analyze_angle(state[base_num],state[base_num+1],'pos'))
    player_dict['pos_valid'] = player_dict['pos_angle_valid']
    if player_dict['pos_angle_valid']:
      player_dict['proximity'] = state[base_num+2]
    elif strict:
      player_dict['proximity'] = None
    elif state_dict['collides_player'] and ((num_teammates+num_opponents) == 1):
      player_dict['proximity'] = 1.0
    elif (not is_teammate) and state_dict['frozen'] and (num_opponents == 1):
      player_dict['proximity'] = 1.0
    elif self.playing_offense and is_teammate: # pessimize
      player_dict['proximity'] = -1.0
    else:
      player_dict['proximity'] = None
    player_dict.update(self._analyze_angle(state[base_num+3],state[base_num+4],'body'))
    # doing vel_magnitude slightly later
    player_dict.update(self._analyze_angle(state[base_num+6],state[base_num+7],'vel'))
    player_dict['vel_valid'] = player_dict['vel_angle_valid']
    if player_dict['vel_valid'] and not player_dict['pos_angle_valid']:
      if is_teammate:
        which = "Teammate (by dist {0:n})".format(i)
      else:
        which = "Opponent (by dist {0:n})".format(i)
      raise RuntimeError(
        "{0} pos sin/cos are {1:n} and {2:n} but vel sin/cos are {3:n} and {4:n}".format(
          which, state[base_num], state[base_num+1], state[base_num+6], state[base_num+7]))
    if player_dict['vel_valid']:
      player_dict['vel_magnitude'] = state[base_num+5]
    elif strict:
      player_dict['vel_magnitude'] = None
    elif player_dict['proximity'] == 1.0:
      player_dict['vel_magnitude'] = -1.0
    else:
      player_dict['vel_magnitude'] = None
    player_dict['unum'] = self._analyze_unum(state[unum_num],
                                             player_dict['pos_angle_valid'])
    return player_dict

  def _parse_low_level_state(self, state, num_teammates, num_opponents, strict=False):
    length_needed = 59 + (9*num_teammates) + (9*num_opponents)
    self._check_state_len(state, length_needed)
    state_dict = {}

    bool_nums = {'self_pos_valid': 0,
                 'self_vel_valid': 1,
                 'frozen': 8,
                 'collides_ball': 9,
                 'collides_player': 10,
                 'collides_post': 11,
                 'kickable': 12,
                 'ball_pos_valid': 50,
                 'ball_vel_valid': 54,
                 'last_action_success_possible': (58+(9*num_teammates)+(9*num_opponents))}
    for name, num in bool_nums:
      state_dict[name] = bool(state[num] > 0)

    state_dict.update(self._analyze_angle(state[2],state[3],'self_vel'))
    self._compare_valid_angle(state_dict['self_vel_valid'], state_dict['self_vel_angle_valid'],
                              state[1], state[2], state[3], "Self velocity", "angle")
    if state_dict['self_vel_valid']:
      state_dict['self_vel_magnitude'] = state[4]
    elif strict:
      state_dict['self_vel_magnitude'] = None
    else: # most likely 0 magnitude...
      state_dict['self_vel_magnitude'] = -1.0

    state_dict.update(self._analyze_angle(state[5],state[6],'self_body'))

    state_dict['stamina'] = state[7]

    goal_nums = {'goal_center': 13,
                 'goal_top': 16,
                 'goal_bottom': 19}

    for name, num in goal_nums:
      state_dict.update(self._analyze_angle(state[num], state[num+1], name))
      self._compare_valid_angle(state_dict['self_pos_valid'], state_dict[name + '_angle_valid'],
                                state[0], state[num], state[num+1],
                                "Self position", name + '_angle')
      if state_dict['self_pos_valid']:
        state_dict[name + '_proximity'] = state[num+2]
      elif strict:
        state_dict[name + '_proximity'] = None
      elif state_dict['collides_post']:
        state_dict[name + '_proximity'] = 1.0 # approximation
      elif self.playing_offense: # pessimize
        state_dict[name + '_proximity'] = -1.0
      else:
        state_dict[name + '_proximity'] = None

    landmark_nums = {'penalty_box_center': 22,
                     'penalty_box_top': 25,
                     'penalty_box_bottom': 28,
                     'center_field': 31,
                     'corner_top_left': 34,
                     'corner_top_right': 37,
                     'corner_bottom_right': 40,
                     'corner_bottom_left': 43}
    for name, num in landmark_nums:
      state_dict.update(self._analyze_angle(state[num], state[num+1], name))
      self._compare_valid_angle(state_dict['self_pos_valid'], state_dict[name + '_angle_valid'],
                                state[0], state[num], state[num+1],
                                "Self position", name + '_angle')
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
      if (not strict) and state_dict['collides_post']:
        state_dict['OOB_left_proximity'] = -1.0
        state_dict['OOB_right_proximity'] = 1.0
      else:
        state_dict['OOB_left_proximity'] = None
        state_dict['OOB_right_proximity'] = None
      state_dict['OOB_top_proximity'] = None
      state_dict['OOB_bottom_proximity'] = None

    state_dict.update(self._analyze_angle(state[51],state[52],'ball_pos'))
    self._compare_valid_angle(state_dict['ball_pos_valid'], state_dict['ball_pos_angle_valid'],
                              state[50], state[51], state[52],
                              "Ball position", "ball_pos_angle")
    if state_dict['ball_pos_valid']:
      state_dict['ball_proximity'] = state[53]
    elif strict:
      state_dict['ball_proximity'] = None
    elif state_dict['collides_ball'] or state_dict['kickable']:
      state_dict['ball_proximity'] = 1.0
    else: # pessimize
      state_dict['ball_proximity'] = -1.0
    if state_dict['ball_vel_valid']:
      state_dict['ball_vel_magnitude'] = state[55]
    elif strict:
      state_dict['ball_vel_magnitude'] = None
    elif state_dict['collides_ball']:
      state_dict['ball_vel_magnitude'] = -1.0
    else:
      state_dict['ball_vel_magnitude'] = None
    state_dict.update(self._analyze_angle(state[56],state[57],'ball_vel'))
    self._compare_valid_angle(state_dict['ball_vel_valid'], state_dict['ball_vel_angle_valid'],
                              state[54], state[56], state[57],
                              "Ball velocity", "ball_vel_angle")

    state_dict['teammates_list'] = []

    for i in range(num_teammates):
      state_dict['teammates_list'].append(
        self._get_player_stats(state,
                               state_dict=state_dict,
                               num_teammates=num_teammates,
                               num_opponents=num_opponents,
                               i=i,
                               base_num=58+(8*i),
                               is_teammate=True,
                               strict=strict,
                               unum_num=58+(8*num_teammates)+(8*num_opponents)+i))



    state_dict['opponents_list'] = []

    for i in range(num_opponents):
      state_dict['opponents_list'].append(
        self._get_player_stats(state,
                               state_dict=state_dict,
                               num_teammates=num_teammates,
                               num_opponents=num_opponents,
                               i=i,
                               base_num=58+(8*num_teammates)+(8*i),
                               is_teammate=False,
                               strict=strict,
                               unum_num=58+(8*num_teammates)+(8*num_opponents)+num_teammates+i))

    return state_dict

  @staticmethod
  def _check_hl_feature(feature):
    if abs(feature) > 1:
      return None
    return feature

  def _parse_high_level_state(self, state, num_teammates, num_opponents):
    length_needed = 11 + (6*num_teammates) + (3*num_opponents)
    self._check_state_len(state, length_needed)
    state_dict = {}

    hl_features_nums = {'x_pos': 0,
                        'y_pos': 1,
                        'body_angle': 2,
                        'ball_x_pos': 3,
                        'ball_y_pos': 4,
                        'goal_dist': 6,
                        'goal_angle': 7,
                        'goal_open_angle': 8,
                        'closest_opponent_dist': 9}

    for name, num in hl_features_nums:
      state_dict[name] = self._check_hl_feature(state[num])

    if state_dict['body_angle'] is not None:
      state_dict['body_angle_degrees'] = state_dict['body_angle']*180
    else:
      state_dict['body_angle_degrees'] = None

    state_dict['kickable'] = bool(state[5] > 0)

    if state_dict['goal_angle'] is not None:
      state_dict['goal_angle_degrees'] = state_dict['goal_angle']*180
    else:
      state_dict['goal_angle_degrees'] = None

    if state_dict['goal_open_angle'] is not None:
      state_dict['goal_open_angle_degrees'] = (state_dict['goal_open_angle']+1)*90
    else:
      state_dict['goal_open_angle_degrees'] = None

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

    state_dict['last_action_success_possible'] = bool(state[10+(6*num_teammates)+
                                                            (3*num_opponents)])

    return state_dict

  def parse_state(self, state, feature_set=None, strict=False):
    if feature_set is None:
      feature_set = self.feature_set
    num_teammates = self.getNumTeammates()
    num_opponents = self.getNumOpponents()

    if feature_set == LOW_LEVEL_FEATURE_SET:
      return self._parse_low_level_state(state, num_teammates, num_opponents, strict=strict)
    elif feature_set == HIGH_LEVEL_FEATURE_SET:
      return self._parse_high_level_state(state, num_teammates, num_opponents)
    else:
      raise RuntimeError("Unknown feature_set {!r}".format(feature_set))
