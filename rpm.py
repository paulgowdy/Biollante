from collections import deque
import numpy as np
import random

class rpm(object):
	#replay memory
	def __init__(self, max_buffer_size):

		self.buffer_size = max_buffer_size
		self.buffer = []

	def add(self, experience):

		if len(self.buffer) + 1 >= self.buffer_size:

			self.buffer[0:(1 + len(self.buffer)) - self.buffer_size] = []

		self.buffer.append(experience)
			
	def sample(self, batch_size, trace_length):

		sampled_episodes = random.sample(self.buffer,batch_size)

		sampledTraces = []

		for episode in sampled_episodes:

			point = np.random.randint(0, len(episode) + 1 - trace_length)
			sampledTraces.append(episode[point : point + trace_length])

		sampledTraces = np.array(sampledTraces)

		return np.reshape(sampledTraces, [batch_size * trace_length, 5])


	'''	
	def add(self, obj):
		#self.lock.acquire()
		if self.size() > self.buffer_size:
			# self.buffer.popleft()
			# self.buffer = self.buffer[1:]
			# self.buffer.pop(0)

			#trim
			print('buffer size larger than set value, trimming...')
			self.buffer = self.buffer[(self.size()-self.buffer_size):]

		elif self.size() == self.buffer_size:
			self.buffer[self.index] = obj
			self.index += 1
			self.index %= self.buffer_size

		else:
			self.buffer.append(obj)

		#self.lock.release()

	def size(self):
		return len(self.buffer)
	'''