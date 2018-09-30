import tensorflow as tf
import numpy as np
from osim.env import ProstheticsEnv

from ddpg_agent import *

from process_obs import *




if __name__=='__main__':

	env = ProstheticsEnv(visualize=False)

	env.change_model(model='3D', prosthetic=True, difficulty=0, seed=None)

	obs_d = env.reset(project = False)

	obs = process_obs_dict(obs_d)

	obs_size = len(obs)
	actions = 19

	agent = ddpg_agent(obs_size, actions)

	noise_level = 0.99
	noise_decay_rate = 0.01

	#print(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = 'a_m'))


	def train(eps, frameskip = 4):

		global noise_level, env
		import matplotlib.pyplot as plt 

		plt.figure()
		reward_collect = []

		for ep in range(eps):

			noise_level *= (1 - noise_decay_rate)

			print('Episode', ep, 'Noise level:', noise_level)

			z = agent.play_one_episode(env, noise_level, frameskip)
			reward_collect.append(z[0])

			plt.clf()
			plt.plot(reward_collect)
			plt.grid()
			plt.pause(0.001)

			#print(len(agent.replay_memory.buffer), len(agent.replay_memory.buffer[-1]))

			agent.train()
	
	'''
	def update_test():

		import matplotlib.pyplot as plt 
		a = []
		b = []

		#print(len(tf.trainable_variables()))

		for i in range(1100):

			z = agent.sess.run(tf.trainable_variables()[10].value())
			zz = z[0][:2]
			y = agent.sess.run(tf.trainable_variables()[32].value())
			yy = y[0][:2]

			print(zz,yy)
			agent.sess.run(agent.target_update_ops)

			a.append(zz[0])
			b.append(yy[0])

		plt.figure()
		plt.plot(a)
		plt.plot(b)
		plt.show()
	'''