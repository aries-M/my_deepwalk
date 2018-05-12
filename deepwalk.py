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
import graph
import tasks


class DeepWalk(object):

	def __init__(self, session, options):
		# Initialize, pass arguments in options
		self._session=session
		self._options=options
		self.filenames=graph.FileNameContainer()
		

	def build_dataset(self, split_options, is_reconstruct=False, save_path=''):
		# Build dataset, save network information
		num_paths=self._options.num_paths
		path_length=self._options.path_length
		filename1=self._options.data1
		filename2=self._options.data2

		# Build network with call and msg information, save graph and nodeid dictionary in file
		g=graph.Graph(split_options)
		g.load_node_from_txt()
		g.load_edge_from_txt(filename1)
		g.load_edge_from_txt(filename2)
		self.graph_size=g.number_of_nodes()
		print('Data size: ',self.graph_size)

		if is_reconstruct==True:
			self.filenames.next_graph_filename=g.save_graph_as_edgelist(os.path.join(save_path,'next_graph.txt'))
			g.delete_edge_from_graph()

		self.filenames.graph_filename=g.save_graph_as_edgelist(os.path.join(save_path,'graph.txt'))
		self.filenames.dict_filename=g.save_dict(os.path.join(save_path,'dict.txt'))

		# Generate random walks concatenated all in data
		self.data=g.build_deepwalk_corpus(num_paths,path_length)
		self.data_index=0

		# Generate random walks started from each node in data list
		'''self.data=g.build_deepwalk_list(num_paths,path_length)
		self.data_index=0'''

	def save_embeddings(self, filename='temp/embeddings.txt'):
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
		batch_size=self._options.batch_size
		num_skips=self._options.num_skips
		skip_window=self._options.window_size
		assert num_skips<skip_window

		batch=np.ndarray(shape=(batch_size),dtype=np.int32)
		labels=np.ndarray(shape=(batch_size,1),dtype=np.int32)
		for i in range(batch_size//num_skips):
			context_words=[w for w in range(1,skip_window)]
			words_to_use=random.sample(context_words,num_skips)
			for j,context_word in enumerate(words_to_use):
				batch[i*num_skips+j]=self.data[self.data_index][0]
				labels[i*num_skips+j,0]=self.data[self.data_index][context_word]
			if self.data_index==len(self.data)-1:
				self.data_index=0
			else:
				self.data_index+=1

		return batch, labels


	def build_model(self):
		# Build the tensorflow graph, return the input placeholder, return optimizer and loss as variables for object
		batch_size=self._options.batch_size
		graph_size=self.graph_size
		embedding_size=self._options.embedding_size
		num_sampled=self._options.num_sampled

		initial_learning_rate=self._options.learning_rate
		decay_rate=self._options.decay_rate

		with tf.name_scope('inputs'):
				train_inputs=tf.placeholder(tf.int32,shape=[batch_size])
				train_labels=tf.placeholder(tf.int32,shape=[batch_size,1])

		with tf.name_scope('embeddings'):
			embeddings=tf.Variable(tf.random_uniform([graph_size,embedding_size],-1.0,1.0))
			embed=tf.nn.embedding_lookup(embeddings,train_inputs)

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
			global_step = tf.Variable(0, trainable=False)
			learning_rate = tf.train.exponential_decay(initial_learning_rate,global_step,2000,decay_rate,staircase=True)
			self.optimizer=tf.train.GradientDescentOptimizer(learning_rate).minimize(self.loss)

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
			#batch_input,batch_labels=self.generate_batch_list()
			batch_input,batch_labels=self.generate_batch()
			feed_dict={train_inputs:batch_input, train_labels:batch_labels}
			_,loss_val=self._session.run([self.optimizer,self.loss],feed_dict=feed_dict)
			average_loss+=loss_val

			if epoch%2000==0:
				if epoch>0:
					average_loss/=2000
				print('Average loss at step ',epoch,': ',average_loss)
				average_loss=0

		self.final_embeddings=self.normalized_embeddings.eval()



def build_next(options, split_options):
	g=graph.Graph(split_options)
	g.load_node_from_txt()
	g.load_edge_from_txt(options.data1)
	g.load_edge_from_txt(options.data2)
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



# relation inferring, link prediction, potential link prediction
def normal_run(opt, save_path):
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
			model=DeepWalk(session,opt)

			begin_time_str='2010/10/01 00:00:00'
			stop_time_str='2010/10/24 23:59:59'
			split_options=graph.GraphSplitOptions(begin_time_str,stop_time_str)
			model.build_dataset(split_options,False,save_path)

			train_inputs,train_labels=model.build_model()
			model.train_model(train_inputs,train_labels)
			model.saver.save(session,os.path.join(save_path,'model.ckpt'))

			model.save_embeddings(os.path.join(save_path,'embeddings.txt'))

	# Task for embeddings	
	print('begin task relation inferring')
	inferring_task=tasks.run_task('colleague_relations',model.final_embeddings,model.filenames)

	print('begin task potential link prediction')
	prediction2_task=tasks.run_task('potential_link_prediction',model.final_embeddings,model.filenames)


	return (inferring_task, prediction2_task)


# link reconstruction
# link reconstruction remove some edges from the graph, so create another graph
def link_reconstruction_run(opt, save_path):

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
			model=DeepWalk(session,opt)

			begin_time_str='2010/10/01 00:00:00'
			stop_time_str='2010/10/24 23:59:59'
			split_options=graph.GraphSplitOptions(begin_time_str,stop_time_str)
			model.build_dataset(split_options,True,save_path)

			train_inputs,train_labels=model.build_model()
			model.train_model(train_inputs,train_labels)
			model.saver.save(session,os.path.join(save_path,'model.ckpt'))

			model.save_embeddings(os.path.join(save_path,'embeddings.txt'))

	# Task for embeddings	
	print('begin task link reconstruction')
	reconstruction_task=tasks.run_task('link_reconstruction',model.final_embeddings,model.filenames)

	return reconstruction_task


def normal_main():
	n=10
	result=np.zeros((2,4))
	opt=options.Options()
	for i in range(n):
		save_path=os.path.join('tmpdata/deepwalk/normal',str(i))
		if not os.path.exists(save_path):
			os.makedirs(save_path)
		tmp=normal_run(opt,save_path)
		for t in range(2):
			for m in range(4):
				result[t][m]=result[t][m]+tmp[t][m]
	result=result/n
	print('-----------------------')
	print('average\t\taccuracy\t\tprecision\t\trecall\t\tF1')
	print('inferring',result[0][0],result[0][1],result[0][2],result[0][3])
	print('predicting',result[1][0],result[1][1],result[1][2],result[1][3])
	print('-----------------------')


def reconstruction_main():
	n=10
	result=np.zeros(4)
	opt=options.Options()
	for i in range(n):
		save_path=os.path.join('tmpdata/deepwalk/reconstruction',str(i))
		if not os.path.exists(save_path):
			os.makedirs(save_path)
		tmp=link_reconstruction_run(opt,save_path)
		for m in range(4):
			result[m]=result[m]+tmp[m]
	result=result/n
	print('-----------------------')
	print('average\t\taccuracy\t\tprecision\t\trecall\t\tF1')
	print('reconstruction',result[0],result[1],result[2],result[3])
	print('-----------------------')


id2task={0:'normal', 1:'link_reconstruction'}
task_id=1


def main(_):
	if id2task[task_id]=='normal':
		normal_main()
	else:
		reconstruction_main()

	

if __name__=='__main__':
	tf.app.run()












