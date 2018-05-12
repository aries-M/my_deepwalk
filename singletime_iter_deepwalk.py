import os
import sys
import random
import math
import collections

import numpy as np
import networkx as nx 
import tensorflow as tf 

import options
import singletime_graph
import tasks
from deepwalk import DeepWalk

class SingleTimeIterDeepWalk(DeepWalk):

	def build_dataset(self, nodes, edges, begin_time, interval):
		# Build dataset, save network information
		num_paths=self._options.num_paths
		path_length=self._options.path_length
		filename1=self._options.data1
		filename2=self._options.data2

		# Build network with call and msg information, save graph and nodeid dictionary in file
		g=singletime_graph.SingleTimeGraph(nodes,edges)
		g.load_from_txt(filename1,begin_time,begin_time+interval)
		g.load_from_txt(filename2,begin_time,begin_time+interval)
		self.filenames.single_graph_filename=g.save_graph_as_edgelist('temp/single_graph.txt')
		g.merge_graphs(nodes,edges)
		self.graph_size=g.number_of_nodes()
		print('Data size: ',self.graph_size)
		self.filenames.graph_filename=g.save_graph_as_edgelist()
		self.filenames.dict_filename=g.save_dict()

		# Generate random walks started from each node in data list
		self.data=g.build_deepwalk_list(num_paths,path_length)
		self.data_index=0

		return list(g.graph.nodes), list(g.graph.edges)


	def build_model(self, states):
		# Build the tensorflow graph, return the input placeholder, return optimizer and loss as variables for object
		batch_size=self._options.batch_size
		graph_size=self.graph_size
		embedding_size=self._options.embedding_size
		num_sampled=self._options.num_sampled

		with tf.name_scope('inputs'):
			train_inputs=tf.placeholder(tf.int32,shape=[batch_size])
			train_labels=tf.placeholder(tf.int32,shape=[batch_size,1])

		with tf.name_scope('embeddings'):
			embeddings=tf.Variable(tf.random_uniform([graph_size,embedding_size],-1.0,1.0))
			embeddings_with_states=tf.concat([embeddings,states],1)

			weights=tf.Variable(tf.truncated_normal([embedding_size,2*embedding_size],stddev=1.0/math.sqrt(2*embedding_size)))
			embeddings_with_states=tf.nn.sigmoid(tf.matmul(embeddings_with_states,weights,transpose_b=True))

			embed=tf.nn.embedding_lookup(embeddings_with_states,train_inputs)

		with tf.name_scope('weights'):
			nce_weights=tf.Variable(tf.truncated_normal([graph_size,embedding_size],stddev=1.0/math.sqrt(embedding_size)))

		with tf.name_scope('biases'):
			nce_biases=tf.Variable(tf.zeros([graph_size]))

		with tf.name_scope('loss'):
			self.loss=tf.reduce_mean(tf.nn.nce_loss(
				weights=nce_weights,
				biases=nce_biases,
				labels=train_labels,
				inputs=embed,
				num_sampled=num_sampled,
				num_classes=graph_size))

		with tf.name_scope('optimizer'):
			self.optimizer=tf.train.GradientDescentOptimizer(1.0).minimize(self.loss)

		norm=tf.sqrt(tf.reduce_sum(tf.square(embeddings),1,keep_dims=True))
		self.normalized_embeddings=embeddings/norm 

		self.init=tf.global_variables_initializer()

		self.saver=tf.train.Saver()

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

		self.final_embeddings=self.normalized_embeddings.eval()


def main(_):

	opt=options.Options()

	with tf.Graph().as_default(), tf.Session() as session:
		with tf.device('/cpu:0'):
			# Initialize state
			embedding_size=opt.embedding_size
			init_model=SingleTimeIterDeepWalk(session,opt)
			nodes,_=init_model.build_dataset([],[],0,31)
			edges=[]
			states=np.random.rand(init_model.graph_size,embedding_size).astype(np.float32)

	time_interval=3
	end_time=25
	for i in range(0,end_time,time_interval):
		print('Model ',i,':')
		# Build network, Train model, Obtain embeddings
		with tf.Graph().as_default(), tf.Session() as session:
			with tf.device('/cpu:0'):
				model=SingleTimeIterDeepWalk(session,opt)
				_,edges=model.build_dataset(nodes,edges,i,time_interval)
				train_inputs,train_labels=model.build_model(states)
				model.train_model(train_inputs,train_labels)
				model.saver.save(session,os.path.join(opt.save_path,'model.ckpt'))

				# Task for embeddings
				if opt.task_name=='link prediction':
					tasks.run_task(opt.task_name,states,init_model.filenames)
				else:
					tasks.run_task(opt.task_name,model.final_embeddings,init_model.filenames)

				# Update states for next time
				states=model.final_embeddings
		print('--------------')


if __name__=='__main__':
	tf.app.run()

