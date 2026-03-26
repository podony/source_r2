import numpy as np
from cereal import log
from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.common.numpy_fast import interp
from openpilot.common.realtime import DT_MDL
from openpilot.system.swaglog import cloudlog
from openpilot.selfdrive.hardware import EON


TRAJECTORY_SIZE = 33

if EON:
  CAMERA_OFFSET = -0.06
  PATH_OFFSET = 0.0
else:
  CAMERA_OFFSET = 0.04
  PATH_OFFSET = 0.04


class LanePlanner:
  def __init__(self):
    self.ll_t = np.zeros((TRAJECTORY_SIZE,))
    self.ll_x = np.zeros((TRAJECTORY_SIZE,))
    self.lll_y = np.zeros((TRAJECTORY_SIZE,))
    self.rll_y = np.zeros((TRAJECTORY_SIZE,))

    self.lane_width_estimate = FirstOrderFilter(2.7, 9.95, DT_MDL)
    self.lane_width_certainty = FirstOrderFilter(1.0, 0.95, DT_MDL)

    self.lane_width = 2.7

    self.lll_prob = 0.
    self.rll_prob = 0.
    self.d_prob = 0.

    self.lll_std = 0.
    self.rll_std = 0.

    self.l_lane_change_prob = 0.
    self.r_lane_change_prob = 0.

    self.camera_offset = CAMERA_OFFSET
    self.path_offset = PATH_OFFSET

    # 🔥 EUV 튜닝 핵심: d_prob 필터 (좌우 튐 방지)
    self.d_prob_filter = FirstOrderFilter(0.0, 0.8, DT_MDL)


  def parse_model(self, md):
    lane_lines = md.laneLines
    if len(lane_lines) == 4 and len(lane_lines[0].t) == TRAJECTORY_SIZE:
      self.ll_t = (np.array(lane_lines[1].t) + np.array(lane_lines[2].t)) / 2
      self.ll_x = lane_lines[1].x

      # 🔧 카메라 오프셋 적용
      self.lll_y = np.array(lane_lines[1].y) + self.camera_offset
      self.rll_y = np.array(lane_lines[2].y) + self.camera_offset

      self.lll_prob = md.laneLineProbs[1]
      self.rll_prob = md.laneLineProbs[2]

      self.lll_std = md.laneLineStds[1]
      self.rll_std = md.laneLineStds[2]

    desire_state = md.meta.desireState
    if len(desire_state):
      self.l_lane_change_prob = desire_state[log.LateralPlan.Desire.laneChangeLeft]
      self.r_lane_change_prob = desire_state[log.LateralPlan.Desire.laneChangeRight]


  def get_d_path(self, v_ego, path_t, path_xyz):
    path_xyz[:, 1] += self.path_offset

    l_prob, r_prob = self.lll_prob, self.rll_prob

    width_pts = self.rll_y - self.lll_y

    # 🔥 미래 lane 신뢰도 감소 (멀리 있는 lane 영향 줄이기)
    prob_mods = []
    for t_check in (0.0, 1.5, 3.0):
      width_at_t = interp(t_check * (v_ego + 7), self.ll_x, width_pts)
      prob_mods.append(interp(width_at_t, [4.0, 5.0], [1.0, 0.0]))
    mod = min(prob_mods)

    l_prob *= mod
    r_prob *= mod

    # 🔥 std 기반 신뢰도 보정 강화
    l_std_mod = interp(self.lll_std, [.1, .3], [1.0, 0.0])
    r_std_mod = interp(self.rll_std, [.1, .3], [1.0, 0.0])

    l_prob *= l_std_mod
    r_prob *= r_std_mod

    # 🔥 속도 기반 lane 의존도 조절 (EUV 핵심)
    speed_lane_factor = interp(v_ego, [0., 10., 25.], [0.3, 0.7, 1.0])
    l_prob *= speed_lane_factor
    r_prob *= speed_lane_factor

    # 🔧 lane width 계산 안정화
    self.lane_width_certainty.update(l_prob * r_prob)

    current_lane_width = abs(self.rll_y[0] - self.lll_y[0])
    self.lane_width_estimate.update(current_lane_width)

    speed_lane_width = interp(v_ego, [0., 31.], [2.7, 3.5])

    self.lane_width = (
      self.lane_width_certainty.x * self.lane_width_estimate.x +
      (1 - self.lane_width_certainty.x) * speed_lane_width
    )

    clipped_lane_width = min(4.0, self.lane_width)

    path_from_left_lane = self.lll_y + clipped_lane_width / 2.0
    path_from_right_lane = self.rll_y - clipped_lane_width / 2.0

    # 🔥 핵심: lane probability 안정화 (좌우 흔들림 제거)
    raw_d_prob = l_prob + r_prob - l_prob * r_prob
    self.d_prob_filter.update(raw_d_prob)
    self.d_prob = self.d_prob_filter.x

    # 🔥 중앙 유지 bias (EUV 느낌 핵심)
    center_bias = 0.05
    lane_path_y = (
      l_prob * path_from_left_lane +
      r_prob * path_from_right_lane
    ) / (l_prob + r_prob + 1e-4)

    # lane_path_y -= center_bias * np.sign(lane_path_y)
    # ✅ 오른쪽으로 5cm 이동 (EUV 기준 안정값)
    RIGHT_BIAS = 0.05  # 5cm

    lane_path_y += RIGHT_BIAS
    
    safe_idxs = np.isfinite(self.ll_t)

    if safe_idxs[0]:
      lane_path_y_interp = np.interp(
        path_t,
        self.ll_t[safe_idxs],
        lane_path_y[safe_idxs]
      )

      # 🔥 최종 blending (lane vs model path)
      path_xyz[:, 1] = (
        self.d_prob * lane_path_y_interp +
        (1.0 - self.d_prob) * path_xyz[:, 1]
      )

    else:
      cloudlog.warning("Lateral mpc - NaNs in laneline times, ignoring")

    return path_xyz
