import math
import do_mpc
from casadi import *
import matplotlib.pyplot as plt


class Env:
	def __init__(self, env_type):
		self.type = env_type
		self.angles = np.array([0, 0])
		self.r = np.array([0, -1,0])
		self.model_type = 'continuous'
		self.model = do_mpc.model.Model(self.model_type)
		# describing the model
		T = self.model.set_variable('_tvp', 'thrust', 3)
		r = self.model.set_variable('_x', 'linear', 3)
		v = self.model.set_variable('_x', 'dlinear', 3)
		m = self.model.set_variable('_x', 'mass', 1)
		q = self.model.set_variable('_x', 'attitude', 4)
		omega = self.model.set_variable('_x', 'omega', 3)
		self.g = np.array([0, 0, 0])
		d = self.model.set_variable('_u', 'tvc', 3)

		f_b = d * T

		f_n = SX.sym('force_in', 3)

		x, y, z = vertsplit(f_b)
		qw, qx, qy, qz = vertsplit(q)
		f_n[0] = x * (qx * qx + qw * qw - qy * qy - qz * qz) + y * (2 * qx * qy - 2 * qw * qz) + z * (
				2 * qx * qz + 2 * qw * qy)
		f_n[1] = x * (2 * qw * qz + 2 * qx * qy) + y * (qw * qw - qx * qx + qy * qy - qz * qz) + z * (
				-2 * qw * qx + 2 * qy * qz)
		f_n[2] = x * (-2 * qw * qy + 2 * qx * qz) + y * (2 * qw * qx + 2 * qy * qz) + z * (
				qw * qw - qx * qx - qy * qy + qz * qz)

		J = SX.sym('moi', 3, 3)
		J[0, 0] = 10
		J[0, 1] = 0
		J[0, 2] = 0
		J[1, 0] = 0
		J[1, 1] = 10
		J[1, 2] = 0
		J[2, 0] = 0
		J[2, 1] = 0
		J[2, 2] = 10

		op = SX.sym('op', 3, 3)
		op[0, 0] = 0
		op[0, 1] = -omega[2]
		op[0, 2] = omega[1]
		op[1, 0] = omega[2]
		op[1, 1] = 0
		op[1, 2] = -omega[0]
		op[2, 0] = -omega[1]
		op[2, 1] = omega[0]
		op[2, 2] = 0

		l_b = cross(r, f_b)

		oh = SX.sym('oh', 4, 4)
		oh[0, 0] = 0
		oh[0, 1] = -omega[0]
		oh[0, 2] = -omega[1]
		oh[0, 3] = -omega[2]
		oh[1, 0] = omega[0]
		oh[1, 1] = 0
		oh[1, 2] = omega[2]
		oh[1, 3] = -omega[1]
		oh[2, 0] = omega[1]
		oh[2, 1] = -omega[2]
		oh[2, 2] = 0
		oh[2, 3] = omega[0]
		oh[3, 0] = omega[2]
		oh[3, 1] = omega[1]
		oh[3, 2] = -omega[0]
		oh[3, 3] = 0

		J_inv = inv(J)
		domega = -J_inv @ op @ J @ omega + l_b
		dq = (oh @ q) / 2
		dv = (f_n / m) + self.g
		dm =SX.sym('dm',1)
		dm[0]=0# 100 is I_sp*g_ref

		self.model.set_rhs('attitude', dq)
		self.model.set_rhs('omega', domega)
		self.model.set_rhs('dlinear', dv)
		self.model.set_rhs('linear', v)
		self.model.set_rhs('mass', dm)

		self.model.setup()
		print("Model setup successful")
		self.simulator = do_mpc.simulator.Simulator(self.model)

		params_simulator = {
			'integration_tool': 'cvodes',
			'abstol': 1e-10,
			'reltol': 1e-10,
			't_step': 0.005
		}

		self.simulator.set_param(**params_simulator)
		self.tvp_template = self.simulator.get_tvp_template()
		self.simulator.set_tvp_fun(self.tvp_fun)
		self.simulator.setup()
		self.estimator = do_mpc.estimator.StateFeedback(self.model)
		self.x0 = np.array([0, 0, 0, 0, 0, 0, 10, 1, 0, 0, 0, 0, 0, 0]).reshape(-1, 1)
		self.simulator.x0 = self.x0
		self.estimator.x0 = self.x0
		self.buffer = [self.x0]
		self.state_dims = len(self.x0)

	def action(self, action):
		if self.type == 'discrete':
			if action == 0:
				self.angles[0] += .5
			if action == 1:
				self.angles[0] -= .5
			if action == 2:
				self.angles[1] += .5
			if action == 3:
				self.angles[1] -= .5
			action = np.cos(self.angles)
		y_next = self.simulator.make_step(action)
		new_state = self.estimator.make_step(y_next)
		reward = None
		done = None
		self.buffer.append(new_state)
		return new_state, reward, done

	def tvp_fun(self, t_now):
		self.tvp_template['thrust'] = 10000
		return self.tvp_template

	def reset(self):
		self.simulator = self.x0
		self.estimator = self.x0
		self.buffer = [self.x0]
		print("Env is reset")
		return self.x0

	def getBuffer(self):
		return self.buffer


