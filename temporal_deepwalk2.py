import os
import sys
import random
import math
import json
import collections

import numpy as np
import networkx as nx 
import tensorflow as tf 

import options
import temporal_graph
import tasks


class TemporalDeepWalk(object):

	def __init__(self, session, options):
		# Initialize, pass arguments in options
		self._session=session
		self._options=options
		self.filenames=temporal_graph.FileNameContainer()
		

	def build_dataset(self, split_options, is_reconstruct=False, save_path=''):
		# Build dataset, save network information
		num_paths=self._options.num_paths
		path_length=self._options.path_length
		filename1=self._options.data1
		filename2=self._options.data2

		# Build network with call and msg information, save graph and nodeid dictionary in file
		g=temporal_graph.TemporalGraph(split_options)
		g.load_node_from_txt()
		g.load_edge_from_txt(filename1)
		g.load_edge_from_txt(filename2)
		#g.merge_temporal_graphs()
		g.merge_edges()
		self.graph_size=g.number_of_nodes()

		if is_reconstruct==True:
			self.filenames.next_graph_filename=g.save_graph_as_edgelist(os.path.join(save_path,'next_graph.txt'))
			g.delete_edge_from_graph()
		
		self.filenames.graph_filename=g.save_graph_as_edgelist(os.path.join(save_path,'graph.txt'))
		self.filenames.dict_filename=g.save_dict(os.path.join(save_path,'dict.txt'))

		# Generate random walks started from each node in data list
		# data is a 2-dim list, (node,time)
		# Generate a new list from graph
		self.data=g.build_deepwalk_list(num_paths,path_length)
		# Or generate an existing list from file
		#self.data=temporal_graph.load_deepwalk_list()
		self.data_index=0
		self.num_time=g.num_time


	def save_embeddings(self,filename='temp/embeddings.txt'):
		data=json.dumps(np.ndarray.tolist(self.embeddings.eval()))
		openFile=open(filename,'w')
		openFile.write(data)
		openFile.close()
		return filename


	def save_dynamic_embeddings(self,filename='temp/dynamic_embeddings.txt'):
		data=json.dumps(np.ndarray.tolist(self.final_embeddings))
		openFile=open(filename,'w')
		openFile.write(data)
		openFile.close()
		return filename


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
		graph_size=self.graph_size

		batch=np.ndarray(shape=(num_sequences*num_skips,sequence_size),dtype=np.int32)
		labels=np.ndarray(shape=(num_sequences*num_skips,sequence_size),dtype=np.int32)
		for i in range(num_sequences):
			for j in range(sequence_size):
				words_to_use=random.sample(range(0,len(self.data[self.data_index][j])),num_skips)
				for k,context_word in enumerate(words_to_use):
					batch[(i*num_skips+k),j]=self.data[self.data_index][j][0]
					labels[(i*num_skips+k),j]=self.data[self.data_index][j][context_word]
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

		graph_size=self.graph_size
		embedding_size=self._options.embedding_size
		num_sampled=self._options.num_sampled
		hidden_size=self._options.hidden_size

		initial_learning_rate=self._options.learning_rate
		decay_rate=self._options.decay_rate

		with tf.name_scope('inputs'):
			train_inputs=tf.placeholder(tf.int32,shape=[num_sequences*num_skips,sequence_size])
			train_labels=tf.placeholder(tf.int32,shape=[num_sequences*num_skips,sequence_size])

		with tf.name_scope('embeddings'):
			# embeddings: 0-axis: node(graph size); 1-axis: time(sequence size); 2-axis: feature(embedding size)
			# reshape embeddings and train_inputs into 2 dimensions, for embedding_loopup
			embeddings=tf.Variable(tf.random_uniform([graph_size,sequence_size,embedding_size],-1.0,1.0))
			self.embeddings=embeddings
			train_inputs_matrix=[[train_inputs[i,j]*sequence_size+j for j in range(sequence_size)] for i in range(num_sequences*num_skips)]
			embeddings_matrix=tf.reshape(embeddings,[graph_size*sequence_size,embedding_size])
			embed=tf.nn.embedding_lookup(embeddings_matrix,train_inputs_matrix)
			embed=tf.reshape(embed,[num_sequences*num_skips,sequence_size,embedding_size])

			# add LSTM/GRU/RNN layer for input embedding
			lstm_cell=tf.contrib.rnn.BasicLSTMCell(num_units=embedding_size, forget_bias=1.0, state_is_tuple=True)
			#lstm_cell=tf.contrib.rnn.GRUCell(num_units=embedding_size)
			#lstm_cell=tf.contrib.rnn.BasicRNNCell(num_units=embedding_size)
			#lstm_cell = tf.contrib.rnn.DropoutWrapper(cell=lstm_cell, input_keep_prob=1.0, output_keep_prob=0.5)
			init_state=lstm_cell.zero_state(num_sequences*num_skips,dtype=tf.float32)
			with tf.variable_scope("rcnn", reuse=False): 
				dyn_embed,state=tf.nn.dynamic_rnn(lstm_cell, inputs=embed, initial_state=init_state, time_major=False)
			#dyn_embed=embed

		self.loss=0.0

		for i in range(0,sequence_size):
			dyn_embed_i=dyn_embed[:,i,:]
			train_labels_i=train_labels[:,i:i+1]

			# each nce weights and biases represent the context vector of a time step
			nce_weights=tf.Variable(tf.truncated_normal([graph_size,embedding_size],stddev=1.0/math.sqrt(embedding_size)))
			nce_biases=tf.Variable(tf.zeros([graph_size]))
			self.loss=(tf.reduce_mean(tf.nn.nce_loss(
				weights=nce_weights,
				biases=nce_biases,
				labels=train_labels_i,
				inputs=dyn_embed_i,
				num_sampled=num_sampled,
				num_classes=graph_size)))

		with tf.name_scope('optimizer'):
			global_step = tf.Variable(0, trainable=False)
			learning_rate = tf.train.exponential_decay(initial_learning_rate,global_step,2000,decay_rate,staircase=True)
			self.optimizer=tf.train.GradientDescentOptimizer(learning_rate).minimize(self.loss)

		norm=tf.sqrt(tf.reduce_sum(tf.square(embeddings),2,keep_dims=True))
		self.normalized_embeddings=embeddings/norm 

		self.init=tf.global_variables_initializer()

		self.saver=tf.train.Saver()

		with tf.name_scope('test'):
			init_state=lstm_cell.zero_state(graph_size,dtype=tf.float32)
			with tf.variable_scope("rcnn", reuse=True): 
				dynamic_embeddings,state=tf.nn.dynamic_rnn(lstm_cell, inputs=embeddings, initial_state=init_state, time_major=False)
			#dynamic_embeddings=embeddings
			norm=tf.sqrt(tf.reduce_sum(tf.square(dynamic_embeddings),2,keep_dims=True))
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

			if epoch%2000==0 and self._options.task_name=='colleague relations':
				print('Predicting accuracy at step ',epoch,": ")
				self.final_embeddings=self.normalized_dynamic_embeddings.eval()
				tasks.run_task(self._options.task_name,self.final_embeddings[:,-1,:], self.filenames)

				self.save_embeddings()
				self.save_dynamic_embeddings()

		self.final_embeddings=self.normalized_dynamic_embeddings.eval()


def load_data(filename='temp/embeddings.txt'):
	data=json.load(open(filename))
	return data



def build_next(options, split_options):
	g=temporal_graph.TemporalGraph(split_options)
	g.load_node_from_txt()
	g.load_edge_from_txt(options.data1)
	g.load_edge_from_txt(options.data2)
	#g.merge_nodes_to_final_graph()
	graph_filename=g.save_graph_as_edgelist('temp/next_graph.txt')
	return graph_filename


def save_config(opt, save_path):
	openFile=open(os.path.join(save_path,'config'),'w')

	openFile.write('num_paths: '+str(opt.num_paths)+'\n')
	openFile.write('path_length: '+str(opt.path_length)+'\n')
	openFile.write('embedding_size: '+str(opt.embedding_size)+'\n')
	openFile.write('learning_rate: '+str(opt.learning_rate)+'\n')
	openFile.write('decay_rate: '+str(opt.decay_rate)+'\n')
	openFile.write('epochs_to_train: '+str(opt.epochs_to_train)+'\n')
	openFile.write('batch_size: '+str(opt.batch_size)+'\n')
	openFile.write('num_skips: '+str(opt.num_skips)+'\n')

	openFile.close()


def save_embeddings_for_multiscale(embeddings_tuple, filename):
	embeddings=np.concatenate(embeddings_tuple,axis=1)
	data=json.dumps(np.ndarray.tolist(embeddings))
	openFile=open(filename,'w')
	openFile.write(data)
	openFile.close()
	return filename


def singlescale_run(opt, is_reconstruct, save_path):

	print('task_name:',opt.task_name)
	print('num_paths: ',opt.num_paths)
	print('path_length: ',opt.path_length)
	print('embedding_size: ',opt.embedding_size)
	print('learning_rate: ',opt.learning_rate)
	print('decay_rate: ',opt.decay_rate)
	print('epochs_to_train: ',opt.epochs_to_train)
	print('num_sequences: ',opt.num_sequences)
	print('num_skips: ',opt.num_skips)
	print('------------------------')

	save_config(opt,save_path)


	# Build network, Train model, Obtain embeddings
	with tf.Graph().as_default(), tf.Session() as session:
		with tf.device('/cpu:0'):
			model=TemporalDeepWalk(session,opt)

			print("Build dataset...")
			begin_time_str='2010/10/01 00:00:00'
			end_time_str='2010/10/24 23:59:59'
			scale_len_str='2010/01/4 00:00:00'
			split_options=temporal_graph.GraphSplitOptions(begin_time_str,end_time_str,scale_len_str)
			model.build_dataset(split_options,is_reconstruct,save_path)

			print("Build model...")
			train_inputs,train_labels=model.build_model()
			model.train_model(train_inputs,train_labels)
			model.saver.save(session,os.path.join(save_path,'model.ckpt'))

			model.save_dynamic_embeddings(os.path.join(save_path,'embeddings.txt'))

	# Get the embeddings at certain time, shows that with more temporal information, accuracy imporves
	'''for i in range(model.num_time):
		single_embeddings=model.final_embeddings[:,i,:]
		# Task for embeddings
		tasks.run_task(opt.task_name,single_embeddings,model.filenames)'''

	# Task for embeddings	
	if not is_reconstruct:
		print('begin task relation inferring')
		return tasks.run_task('colleague_relations',model.final_embeddings[:,-1,:],model.filenames)
	else:
		print('begin task link reconstruction')
		return tasks.run_task('link_reconstruction',model.final_embeddings[:,-1,:],model.filenames)

	


def multiscale_run(opt, is_reconstruct, save_path):

	print('task_name:',opt.task_name)
	print('num_paths: ',opt.num_paths)
	print('path_length: ',opt.path_length)
	print('embedding_size: ',opt.embedding_size)
	print('learning_rate: ',opt.learning_rate)
	print('decay_rate: ',opt.decay_rate)
	print('epochs_to_train: ',opt.epochs_to_train)
	print('num_sequences: ',opt.num_sequences)
	print('num_skips: ',opt.num_skips)
	print('------------------------')

	save_config(opt,save_path)

	g1=tf.Graph()
	with g1.as_default(), tf.Session() as session:
		with tf.device('/cpu:0'):

			# HOUR scale
			model1=TemporalDeepWalk(session,opt)

			print("Build dataset...")
			begin_time_str='2010/10/24 00:00:00'
			end_time_str='2010/10/24 23:59:59'
			scale_len_str='2010/01/01 03:00:00'
			split_options=temporal_graph.GraphSplitOptions(begin_time_str,end_time_str,scale_len_str)
			model1.build_dataset(split_options,is_reconstruct,save_path)

			print("Build model...")
			train_inputs,train_labels=model1.build_model()
			model1.train_model(train_inputs,train_labels)
			model1.saver.save(session,os.path.join(save_path,'model1.ckpt'))

			#model.save_dynamic_embeddings(os.path.join(save_path,'embeddings1.txt'))


	g2=tf.Graph()
	with g2.as_default(), tf.Session() as session:
		with tf.device('/cpu:0'):
			# DAY scale
			model2=TemporalDeepWalk(session,opt)

			print("Build dataset...")
			begin_time_str='2010/10/01 00:00:00'
			end_time_str='2010/10/24 23:59:59'
			scale_len_str='2010/01/4 00:00:00'
			split_options=temporal_graph.GraphSplitOptions(begin_time_str,end_time_str,scale_len_str)
			model2.build_dataset(split_options,is_reconstruct,save_path)

			print("Build model...")
			train_inputs,train_labels=model2.build_model()
			model2.train_model(train_inputs,train_labels)
			model2.saver.save(session,os.path.join(save_path,'model2.ckpt'))

			#model2.save_embeddings(filename='temp/embeddings2.txt')

			filenames=model2.filenames

	embeddings=np.concatenate((model1.final_embeddings[:,-1,:],model2.final_embeddings[:,-1,:]),axis=1)
	save_embeddings((model1.final_embeddings[:,-1,:],model2.final_embeddings[:,-1,:]),os.path.join(save_path,'embeddings.txt'))

	# Task for embeddings	
	if not is_reconstruct:
		print('begin task relation inferring')
		return tasks.run_task('colleague_relations',embeddings,model.filenames)
	else:
		print('begin task link reconstruction')
		return tasks.run_task('link_reconstruction',embeddings,model.filenames)



id2model={0:'temporal_deepwalk', 1:'multiscale_deepwalk'}
model_id=0
id2task={0:'normal', 1:'reconstruction'}
task_id=1


def main(_):
	n=10
	is_reconstruct=(task_id==1)
	all_result=[]
	opt=options.Options()
	for i in range(n):
		save_path=os.path.join('tmpdata',id2model[model_id],id2task[task_id],str(i))
		if not os.path.exists(save_path):
			os.makedirs(save_path)
		if model_id==0:
			result=singlescale_run(opt,is_reconstruct,save_path)
		elif model_id==1:
			result=multiscale_run(opt,is_reconstruct,save_path)
		else:
			pass
		all_result.append(result)
	
	all_result=np.array(all_result)
	avg_result=np.mean(all_result,0)

	print('------------')
	print('accuracy\t\tprecision\t\trecall\t\tF1')
	print(avg_result[0],avg_result[1],avg_result[2],avg_result[3])
	print('-----------------------')


if __name__=='__main__':
	tf.app.run()



