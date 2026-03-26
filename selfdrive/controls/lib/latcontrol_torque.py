import math

from cereal import log
from openpilot.common.numpy_fast import interp
from openpilot.common.numpy_fast import clip
from openpilot.selfdrive.controls.lib.latcontrol import LatControl
from openpilot.selfdrive.controls.lib.pid import PIDController
from openpilot.selfdrive.controls.lib.vehicle_model import ACCELERATION_DUE_TO_GRAVITY


# 🔧 EUV 튜닝: 저속 보정 강화
LOW_SPEED_X = [0, 5, 10, 20]
LOW_SPEED_Y = [20, 15, 10, 0]


class LatControlTorque(LatControl):
  def __init__(self, CP, CI):
    super().__init__(CP, CI)

    self.torque_params = CP.lateralTuning.torque

    # 🔧 PID 튜닝 (안정성 + 반응성)
    self.pid = PIDController(
      self.torque_params.kp * 1.1,
      self.torque_params.ki * 0.9,
      k_f=self.torque_params.kf,
      pos_limit=self.steer_max,
      neg_limit=-self.steer_max
    )

    self.torque_from_lateral_accel = CI.torque_from_lateral_accel()
    self.use_steering_angle = self.torque_params.useSteeringAngle
    self.steering_angle_deadzone_deg = self.torque_params.steeringAngleDeadzoneDeg


  def update_live_torque_params(self, latAccelFactor, latAccelOffset, friction):
    self.torque_params.latAccelFactor = latAccelFactor
    self.torque_params.latAccelOffset = latAccelOffset

    # 🔧 friction 강화 (핸들 "먹는 느낌")
    self.torque_params.friction = friction * 1.2


  def update(self, active, CS, VM, params, last_actuators, steer_limited,
             desired_curvature, desired_curvature_rate, llk):

    pid_log = log.ControlsState.LateralTorqueState.new_message()

    if not active:
      output_torque = 0.0
      pid_log.active = False

    else:
      if self.use_steering_angle:
        actual_curvature = -VM.calc_curvature(
          math.radians(CS.steeringAngleDeg - params.angleOffsetDeg),
          CS.vEgo, params.roll
        )
        curvature_deadzone = abs(
          VM.calc_curvature(
            math.radians(self.steering_angle_deadzone_deg),
            CS.vEgo, 0.0
          )
        )
      else:
        actual_curvature_vm = -VM.calc_curvature(
          math.radians(CS.steeringAngleDeg - params.angleOffsetDeg),
          CS.vEgo, params.roll
        )
        actual_curvature_llk = llk.angularVelocityCalibrated.value[2] / CS.vEgo
        actual_curvature = interp(
          CS.vEgo,
          [2.0, 5.0],
          [actual_curvature_vm, actual_curvature_llk]
        )
        curvature_deadzone = 0.0

      # 🔧 lateral accel 계산
      desired_lateral_accel = desired_curvature * CS.vEgo ** 2
      actual_lateral_accel = actual_curvature * CS.vEgo ** 2
      lateral_accel_deadzone = curvature_deadzone * CS.vEgo ** 2

      # 🔧 저속 보정 (EUV 핵심)
      low_speed_factor = interp(CS.vEgo, LOW_SPEED_X, LOW_SPEED_Y) ** 2

      setpoint = desired_lateral_accel + low_speed_factor * desired_curvature
      measurement = actual_lateral_accel + low_speed_factor * actual_curvature

      gravity_adjusted_lateral_accel = desired_lateral_accel - params.roll * ACCELERATION_DUE_TO_GRAVITY

      torque_from_setpoint = self.torque_from_lateral_accel(
        setpoint,
        self.torque_params,
        setpoint,
        lateral_accel_deadzone,
        friction_compensation=False
      )

      torque_from_measurement = self.torque_from_lateral_accel(
        measurement,
        self.torque_params,
        measurement,
        lateral_accel_deadzone,
        friction_compensation=False
      )

      pid_log.error = torque_from_setpoint - torque_from_measurement

      # 🔥 핵심: friction + feedforward 강화
      ff = self.torque_from_lateral_accel(
        gravity_adjusted_lateral_accel,
        self.torque_params,
        desired_lateral_accel - actual_lateral_accel,
        lateral_accel_deadzone,
        friction_compensation=True
      ) * 1.15   # 🔧 EUV 느낌 핵심 포인트

      freeze_integrator = steer_limited or CS.steeringPressed or CS.vEgo < 5

      output_torque = self.pid.update(
        pid_log.error,
        feedforward=ff,
        speed=CS.vEgo,
        freeze_integrator=freeze_integrator
      )

      # 🔧 토크 제한 (급조향 방지)
      output_torque = clip(output_torque, -0.8, 0.8)

      pid_log.active = True
      pid_log.p = self.pid.p
      pid_log.i = self.pid.i
      pid_log.d = self.pid.d
      pid_log.f = self.pid.f
      pid_log.output = -output_torque
      pid_log.actualLateralAccel = actual_lateral_accel
      pid_log.desiredLateralAccel = desired_lateral_accel

      pid_log.saturated = self._check_saturation(
        self.steer_max - abs(output_torque) < 1e-3,
        CS,
        steer_limited
      )

    return -output_torque, 0.0, pid_log
