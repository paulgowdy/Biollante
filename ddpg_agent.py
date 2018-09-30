import tensorflow as tf
import numpy as np

from osim.env import ProstheticsEnv

from process_obs import *
from rpm import *

'''
class actor_network():

	def __init__(self, inputdims, outputdims, name):

		with tf.variable_scope(name):

			self.obs_input = tf.placeholder(shape=[None, inputdims], dtype=tf.float32)

			fc1 = tf.layers.dense(self.obs_input, 100, activation = tf.nn.selu)
			fc1 = tf.layers.batch_normalization(fc1)

			fc2 = tf.layers.dense(fc1, 100, activation = tf.nn.selu)
			fc2 = tf.layers.batch_normalization(fc2)

			action_output = tf.layers.dense(fc2, outputdims, activation = tf.tanh)
			self.action_output = 0.5 * action_output + 0.5

class critic_network():

	def __init__(self, inputdims, outputdims, name):

		with tf.variable_scope(name):

			self.obs_input = tf.placeholder(shape=[None, inputdims], dtype=tf.float32)
			self.action_input = tf.placeholder(shape=[None, outputdims], dtype=tf.float32)

			fc1 = tf.layers.dense(self.obs_input, 100, activation = tf.nn.selu)
			fc1 = tf.layers.batch_normalization(fc1)

			fc1 = tf.concat([fc1, self.action_input], 1)

			fc2 = tf.layers.dense(fc1, 100, activation = tf.nn.selu)
			fc2 = tf.layers.batch_normalization(fc2)

			self.value_output = tf.layers.dense(fc2, 1, activation = None)
'''

class actor_network():

	def __init__(self, inputdims, outputdims, batch_size, trace_length, name):

		h_size = 100
		rnn_units = 100
		self.batch_size = batch_size
		self.trace_length = trace_length

		with tf.variable_scope(name):

			rnn_cell = tf.contrib.rnn.BasicLSTMCell(num_units=rnn_units, state_is_tuple=True)

			self.trace_length = tf.placeholder(dtype=tf.int32)
			self.batch_size = tf.placeholder(dtype=tf.int32,shape=[])

			self.obs_input = tf.placeholder(shape=[None, inputdims], dtype=tf.float32)

			fc1 = tf.layers.dense(self.obs_input, 100, activation = tf.nn.selu)
			fc1 = tf.layers.batch_normalization(fc1)

			fc2 = tf.layers.dense(fc1, h_size, activation = tf.nn.selu)
			fc2 = tf.layers.batch_normalization(fc2)

			# group traces for RNN

			traces = tf.reshape(fc2, [self.batch_size, self.trace_length, h_size])

			self.state_in = rnn_cell.zero_state(self.batch_size, tf.float32)

			self.rnn, self.rnn_state = tf.nn.dynamic_rnn(cell= rnn_cell, inputs = traces, dtype=tf.float32, initial_state = self.state_in)

			rnn_out = tf.reshape(self.rnn, shape=[-1, rnn_units])

			action_output = tf.layers.dense(rnn_out, outputdims, activation = tf.tanh)
			self.action_output = 0.5 * action_output + 0.5

			# actor loss

			#critic_q = tf.placeholder(dtype=tf.float32)
			#actor_loss = tf.reduce_mean(-1.0 * critic_q)
			#self.trainer = tf.train.AdamOptimizer(learning_rate=0.0001)
			#self.train_actor = self.trainer.minimize(actor_loss)

			self.input_critic_grad = tf.placeholder(tf.float32, [None, 64, 19])

			actor_model_weights = tf.trainable_variables(scope = name)

			self.actor_grads = tf.gradients(self.action_output, actor_model_weights, -self.input_critic_grad)
			grads = zip(self.actor_grads, actor_model_weights)
			self.optimize =  tf.train.AdamOptimizer(0.0001).apply_gradients(grads)


class critic_network():

	def __init__(self, inputdims, outputdims, name):

		with tf.variable_scope(name):

			self.obs_input = tf.placeholder(shape=[None, inputdims], dtype=tf.float32)
			self.action_input = tf.placeholder(shape=[None, outputdims], dtype=tf.float32)

			fc1 = tf.layers.dense(self.obs_input, 100, activation = tf.nn.selu)
			fc1 = tf.layers.batch_normalization(fc1)

			fc1 = tf.concat([fc1, self.action_input], 1)

			fc2 = tf.layers.dense(fc1, 100, activation = tf.nn.selu)
			fc2 = tf.layers.batch_normalization(fc2)

			self.value_output = tf.layers.dense(fc2, 1, activation = None)

			# critic loss

			#q2 = tf.placeholder(dtype=tf.int32)
			#r1 = 
			#q1_target = r1 + (1-isdone) * self.discount_factor * q2
			self.q1_target = tf.placeholder(dtype=tf.float32)
			critic_loss = tf.reduce_mean((self.q1_target - self.value_output)**2)
			self.trainer = tf.train.AdamOptimizer(learning_rate=0.0001)
			self.train_critic = self.trainer.minimize(critic_loss)

			#self.critic_grads = tf.reduce_mean(tf.gradients(self.value_output, self.action_input), axis = 1)
			self.critic_grads = tf.gradients(self.value_output, self.action_input)
		





class ddpg_agent(object):

	def __init__(self,
		observation_space_dims,
		nb_actions,
		tau = 0.005,
		memory_size = 100000):

		# replay memory
		self.batch_size = 8
		self.trace_length = 8

		self.replay_memory = rpm(memory_size)

		self.discount_factor = 0.99


		self.inputdims = observation_space_dims
		self.outputdims = nb_actions

		self.actor = actor_network(self.inputdims,self.outputdims, self.batch_size, self.trace_length, 'a_m')
		self.critic = critic_network(self.inputdims,self.outputdims, 'c_m')
		
		self.actor_target = actor_network(self.inputdims,self.outputdims, self.batch_size, self.trace_length, 'a_t')
		self.critic_target = critic_network(self.inputdims,self.outputdims, 'c_t')

		self.sess = tf.Session()
		self.sess.run(tf.global_variables_initializer())

		self.target_update_ops = self.update_target_networks(tau)
		#self.train_and_update_feed = self.train_and_update_op()

		

	def clamp_actions(self, actions):

			return np.clip(actions, a_max = 1.0 , a_min = 0.0)

	def act(self, observation, state_input):

		obs_feed = np.array([observation])

		#print(obs_feed.shape)

		actions, state1 = self.sess.run([self.actor.action_output, self.actor.rnn_state], feed_dict = {self.actor.obs_input : obs_feed, self.actor.trace_length : 1, self.actor.batch_size : 1, self.actor.state_in : state_input})

		return self.clamp_actions(actions[0]), state1

	def critique(self, observation, action):

		obs_feed = np.array([observation])
		a_feed = np.array([action])

		v = self.sess.run(self.critic.value_output, feed_dict = {self.critic.obs_input : obs_feed, self.critic.action_input : a_feed})

		return v[0]

	def update_target_networks(self, tau):

		#print(tf.trainable_variables())

		total_vars = len(tf.trainable_variables())
		tfVars = tf.trainable_variables()

		#g = tf.get_default_graph()

		op_holder = []

		#for z in range(len(tf.trainable_variables())//2):

		for idx, var in enumerate(tfVars[0 : total_vars//2]):

			op_holder.append(tfVars[idx + total_vars//2].assign((var.value() * tau) + ((1 - tau) * tfVars[idx + total_vars//2].value())))
		
		return op_holder

	def play_one_episode(self, env, noise_level, frameskip, max_steps = 400):

		steps = 0
		ep_total_reward = 0
		ep_learning_reward = 0
		episode_memory = []

		#print('frameskip:', frameskip)

		# Reset Env
		observation_d = env.reset(project = False)
		observation = process_obs_dict(observation_d)

		done = False

		h_size = 100
		rnn_units = 50

		state_in = (np.zeros([1,h_size]), np.zeros([1, h_size]))

		while not done and steps < max_steps:

			observation_before_action = observation

			action, state_out = self.act(observation_before_action, state_in)
			state_in = state_out
			# Do noise phasing here...

			exploration_noise = np.random.normal(size=(self.outputdims,)) * noise_level # * phased_noise_amplitude

			action += exploration_noise
			action = self.clamp_actions(action)

			learning_reward = 0

			for _ in range(frameskip):

				steps +=1

				observation_d, reward, done, _info = env.step(action, project = False)

				ep_total_reward += reward
				#learning_reward += reward_shaping(observation_d, reward)
				#lr, done = reward_shaping(observation_d, reward, done)
				learning_reward += reward #lr

				if done:
					break

			ep_learning_reward += learning_reward

			observation = process_obs_dict(observation_d)

			isdone = 1 if done else 0

			transition = (observation_before_action, action, learning_reward, isdone, observation)

			# add individual transition to replay memory
			#self.add_transition_to_replay_memory(transition)

			# collect transitions for the whole episode
			# add whole episode to memory at end
			episode_memory.append(transition)


			# Train here? During on policy movement...

			# No, try training after each episode...

			if done:

				break

		self.add_episode_to_replay_memory(episode_memory)

		return ep_total_reward, steps

	def add_transition_to_replay_memory(self, transition):

		pass

	def add_episode_to_replay_memory(self, episode_memory):

		self.replay_memory.add(episode_memory)
	


	def train(self, min_memory_size = 9):

		if len(self.replay_memory.buffer) > min_memory_size:

			memory_batch = self.replay_memory.sample(self.batch_size, self.trace_length)

			#print(memory_batch.shape)
			#[s1,a1,r1,isdone,s2]
			#print(memory_batch[0,1:3])
			#self.train_and_update_op(memory_batch)

			s1 = np.vstack(memory_batch[:,0])
			a1 = np.vstack(memory_batch[:,1])
			r1 = np.vstack(memory_batch[:,2])
			isdone = np.vstack(memory_batch[:,3])
			s2 = np.vstack(memory_batch[:,4])

			h_size = 100

			state_train = (np.zeros([self.batch_size,h_size]),np.zeros([self.batch_size,h_size])) 

			a2 = self.sess.run(self.actor_target.action_output,
							feed_dict = {self.actor_target.obs_input : s2,
										self.actor_target.batch_size : self.batch_size,
										self.actor_target.trace_length : self.trace_length,
										self.actor_target.state_in : state_train})

			q2 = self.sess.run(self.critic_target.value_output,
							feed_dict = {self.critic_target.obs_input : s2,
										self.critic_target.action_input : a2})

			q1_target = r1 + (1 - isdone) * self.discount_factor * q2

			#print('training critic!')
			self.sess.run(self.critic.train_critic, 
						feed_dict = {self.critic.obs_input : s1,
									self.critic.action_input : a1,
									self.critic.q1_target: q1_target})

			#print('training actor!')
			a1_predict = self.sess.run(self.actor.action_output,
						feed_dict = {self.actor.obs_input : s1,
										self.actor.batch_size : self.batch_size,
										self.actor.trace_length : self.trace_length,
										self.actor.state_in : state_train})

			critic_grads = self.sess.run(self.critic.critic_grads,
						feed_dict = {self.critic.obs_input : s1,
										self.critic.action_input : a1_predict})

			self.sess.run(self.actor.optimize,
						feed_dict = {self.actor.obs_input : s1,
										self.actor.batch_size : self.batch_size,
										self.actor.trace_length : self.trace_length,
										self.actor.state_in : state_train,
										self.actor.input_critic_grad : critic_grads})
				





