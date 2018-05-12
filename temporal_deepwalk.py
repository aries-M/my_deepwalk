import os
import sys
import random
import math
import collections

import numpy as np
import networkx as nx 
import tensorflow as tf 

import options
import temporal_graph
import tasks

class DeepWalk(object):

	def __init__(self, session, options):
		# Initialize, pass arguments in options
		self._session=session
		self._options=options
		

	def build_dataset(self):
		# Build dataset, save network information
		num_paths=self._options.num_paths
		path_length=self._options.path_length
		filename1=self._options.data1
		filename2=self._options.data2

		# Build network with call and msg information, save graph and nodeid dictionary in file
		g=temporal_graph.TemporalGraph()
		g.load_from_txt(filename1)
		g.load_from_txt(filename2)
		g.merge_temporal_graphs()
		self.graph_size=g.number_of_nodes()
		print('Data size: ',self.graph_size)
		self.graph_filename=g.save_graph_as_edgelist()
		self.dict_filename=g.save_dict()

		# Generate random walks concatenated all in data
		#self.data=g.build_deepwalk_corpus(num_paths,path_length)
		#self.data_index=0

		# Generate random walks started from each node in data list
		# data is a 2-dim list, (node,time)
		# Generate a new list from graph
		self.data=g.build_deepwalk_list(num_paths,path_length)
		# Or generate an existing list from file
		#self.data=temporal_graph.load_deepwalk_list()
		print(np.any(np.isnan(self.data)))

		self.data_index=0
		self.num_time=g.num_time


	def generate_batch(self):
		# Generate batch for data, suitable for word2vec or deepwalk
		batch_size=self._options.batch_size
		num_skips=self._options.num_skips
		skip_window=self._options.window_size
		assert num_skips<=2*skip_window

		batch=np.ndarray(shape=(batch_size),dtype=np.int32)
		labels=np.ndarray(shape=(batch_size,1),dtype=np.int32)
		span=2*skip_window+1  
		buffer=collections.deque(maxlen=span) 
		if self.data_index + span > len(self.data):
			self.data_index = 0
		buffer.extend(self.data[self.data_index:self.data_index + span])
		self.data_index += span
		for i in range(batch_size // num_skips):
			context_words = [w for w in range(span) if w != skip_window]
			words_to_use = random.sample(context_words, num_skips)
			for j, context_word in enumerate(words_to_use):
				batch[i * num_skips + j] = buffer[skip_window]
				labels[i * num_skips + j, 0] = buffer[context_word]
			if self.data_index == len(self.data):
				buffer.extend(self.data[0:span])
				self.data_index = span
			else:
				buffer.append(self.data[self.data_index])
				self.data_index += 1

		self.data_index = (self.data_index + len(self.data) - span) % len(self.data)
		return batch, labels


	def generate_batch_list(self):
		# Generate batch for data list, for each random walks, only use the first node as key
		# num_sequences: number of sequences in a batch
		# sequence_size: size of a sequence
		# batch_size: number of inputs in a batch
		num_sequences=self._options.num_sequences
		num_skips=self._options.num_skips
		sequence_size=self.num_time
		batch_size=num_sequences*num_skips*sequence_size
		graph_size=self.graph_size

		batch=np.ndarray(shape=(batch_size),dtype=np.int32)
		labels=np.ndarray(shape=(batch_size,1),dtype=np.int32)
		for i in range(num_sequences):
			for j in range(sequence_size):
				words_to_use=random.sample(range(0,len(self.data[self.data_index][j])),num_skips)
				for k,context_word in enumerate(words_to_use):
					batch[(i*num_skips+k)*sequence_size+j]=self.data[self.data_index][j][0]+j*graph_size
					labels[(i*num_skips+k)*sequence_size+j]=self.data[self.data_index][j][context_word]+j*graph_size
			if self.data_index==len(self.data)-1:
				self.data_index=0
			else:
				self.data_index+=1

		return batch, labels


	def build_model(self):
		# Build the tensorflow graph, return the input placeholder, return optimizer and loss as variables for object
		num_sequences=self._options.num_sequences
		num_skips=self._options.num_skips
		sequence_size=self.num_time
		batch_size=num_sequences*num_skips*sequence_size

		graph_size=self.graph_size
		embedding_size=self._options.embedding_size
		num_sampled=self._options.num_sampled
		hidden_size=self._options.hidden_size

		initial_learning_rate=self._options.learning_rate
		decay_rate=self._options.decay_rate

		with tf.name_scope('inputs'):
			train_inputs=tf.placeholder(tf.int32,shape=[batch_size])
			train_labels=tf.placeholder(tf.int32,shape=[batch_size,1])

		with tf.name_scope('embeddings'):
			embeddings=tf.Variable(tf.random_uniform([graph_size*sequence_size,embedding_size],-1.0,1.0))
			embed=tf.nn.embedding_lookup(embeddings,train_inputs)

			embed=tf.reshape(embed,[num_sequences*num_skips,sequence_size,embedding_size])

			# add LSTM/GRU/RNN layer for input embedding
			#lstm_cell=tf.contrib.rnn.BasicLSTMCell(num_units=embedding_size, forget_bias=1.0, state_is_tuple=True)
			#lstm_cell=tf.contrib.rnn.GRUCell(num_units=embedding_size)
			lstm_cell=tf.contrib.rnn.BasicRNNCell(num_units=embedding_size,activation=tf.nn.tanh)
			lstm_cell = tf.contrib.rnn.DropoutWrapper(cell=lstm_cell, input_keep_prob=1.0, output_keep_prob=0.5)
			init_state=lstm_cell.zero_state(num_sequences*num_skips,dtype=tf.float32)
			with tf.variable_scope("rcnn", reuse=None): 
				dyn_embed,state=tf.nn.dynamic_rnn(lstm_cell, inputs=embed, initial_state=init_state, time_major=False)

			dyn_embed=tf.reshape(dyn_embed,[batch_size,embedding_size])
			#dyn_embed=embed

		with tf.name_scope('weights'):
			nce_weights=tf.Variable(tf.truncated_normal([graph_size*sequence_size,embedding_size],stddev=1.0/math.sqrt(embedding_size)))

		with tf.name_scope('biases'):
			nce_biases=tf.Variable(tf.zeros([graph_size*sequence_size]))

		with tf.name_scope('loss'):
			self.loss=tf.reduce_mean(tf.nn.nce_loss(
				weights=nce_weights,
				biases=nce_biases,
				labels=train_labels,
				inputs=dyn_embed,
				num_sampled=num_sampled,
				num_classes=graph_size*sequence_size))

		with tf.name_scope('optimizer'):
			global_step = tf.Variable(0, trainable=False)
			learning_rate = tf.train.exponential_decay(initial_learning_rate,global_step,2000,decay_rate,staircase=True)
			self.optimizer=tf.train.GradientDescentOptimizer(learning_rate).minimize(self.loss)

		norm=tf.sqrt(tf.reduce_sum(tf.square(embeddings),1,keep_dims=True))
		self.normalized_embeddings=embeddings/norm 

		self.init=tf.global_variables_initializer()

		self.saver=tf.train.Saver()

		# embeddings/normalized_embeddings: [(sequence_size,graph_size),embedding_size]

		with tf.name_scope('test'):
			static_embeddings=tf.reshape(embeddings,[graph_size,sequence_size,embedding_size])
			init_state=lstm_cell.zero_state(graph_size,dtype=tf.float32)
			with tf.variable_scope("rcnn", reuse=True): 
				dynamic_embeddings,state=tf.nn.dynamic_rnn(lstm_cell, inputs=static_embeddings, initial_state=init_state, time_major=False)
			dynamic_embeddings=tf.reshape(dynamic_embeddings,[graph_size*sequence_size,embedding_size])
			norm=tf.sqrt(tf.reduce_sum(tf.square(dynamic_embeddings),1,keep_dims=True))
			self.normalized_dynamic_embeddings=dynamic_embeddings/norm


		return train_inputs, train_labels


	def train_model(self, train_inputs, train_labels):
		# Train the model with number of epochs
		num_epochs=self._options.epochs_to_train

		self.init.run()

		average_loss=0
		for epoch in range(num_epochs):
			batch_input,batch_labels=self.generate_batch_list()
			feed_dict={train_inputs:batch_input, train_labels:batch_labels}
			_,loss_val=self._session.run([self.optimizer,self.loss],feed_dict=feed_dict)
			average_loss+=loss_val

			if epoch%2000==0:
				if epoch>0:
					average_loss/=2000
				print('Average loss at step ',epoch,': ',average_loss)
				average_loss=0

			if epoch%2000==0:
				print('Predicting accuracy at step ',epoch,": ")
				self.final_embeddings=self.normalized_dynamic_embeddings.eval()
				#self.final_embeddings=self.normalized_embeddings.eval()
				single_embeddings=self.single_time_embeddings()
				tasks.run_task(self._options.task_name,single_embeddings,self.graph_filename,self.dict_filename)


		self.final_embeddings=self.normalized_dynamic_embeddings.eval()
		#self.final_embeddings=self.normalized_embeddings.eval()


	def single_time_embeddings(self, t=-1):
		graph_size=self.graph_size
		sequence_size=self.num_time

		if t==-1:
			t=sequence_size-1
		assert t<sequence_size

		time_idx=[i+graph_size*t for i in range(graph_size)]
		ret=self.final_embeddings[time_idx,:]
		return ret


def main(_):

	opt=options.Options()

	print('num_paths: ',opt.num_paths)
	print('path_length: ',opt.path_length)
	print('embedding_size: ',opt.embedding_size)
	print('learning_rate: ',opt.learning_rate)
	print('decay_rate: ',opt.decay_rate)
	print('epochs_to_train: ',opt.epochs_to_train)
	print('num_sequences: ',opt.num_sequences)
	print('num_skips: ',opt.num_skips)
	print('------------------------')

	# Build network, Train model, Obtain embeddings
	with tf.Graph().as_default(), tf.Session() as session:
		with tf.device('/cpu:0'):
			model=DeepWalk(session,opt)

			print("Build dataset...")
			model.build_dataset()

			print("Build model...")
			train_inputs,train_labels=model.build_model()
			model.train_model(train_inputs,train_labels)
			model.saver.save(session,os.path.join(opt.save_path,'model.ckpt'))

	# Get the embeddings at certain time
	for i in range(model.num_time):
		single_embeddings=model.single_time_embeddings(i)
		# Task for embeddings
		tasks.run_task(opt.task_name,single_embeddings,model.graph_filename,model.dict_filename)


if __name__=='__main__':
	tf.app.run()



