import sys
import os
import random
import json
import time
import math
import numpy as np 
import pymysql as sql 
import networkx as nx 
from networkx.readwrite import json_graph



class FileNameContainer(object):

	def __init__(self):
		self.graph_filename=''
		self.dict_filename=''
		self.next_graph_filename=''



class GraphSplitOptions(object):

	def __init__(self, begin_time_str='2010/10/01 00:00:00', end_time_str='2010/10/25 23:59:59', scale_len_str='2010/01/4 00:00:00'):
		self.begin_time=time_str2float(begin_time_str)
		self.end_time=time_str2float(end_time_str)
		self.scale_len=time_str2float(scale_len_str)

	
def time_str2float(time_str):
	time_tuple=time.strptime(time_str,'%Y/%m/%d %H:%M:%S')
	time_float=time.mktime(time_tuple)-time.mktime(time.strptime('2010/01/01 00:00:00','%Y/%m/%d %H:%M:%S'))
	return time_float


class TemporalGraph(object):

	def __init__(self, split_options):
		# graphs contain graph snapshots, the time scale is one day
		# scale_len: length of the temporal scale by unit.
		self.graphs=[]
		self.dictionary={}
		self.split_options=split_options
		self.num_time=math.ceil((split_options.end_time-split_options.begin_time)/split_options.scale_len)
		for i in range(self.num_time):
			self.graphs.append(nx.Graph())
		#print(self.num_time)

	def load_node_from_txt(self, filename='metadata/new_user_info.txt'):
		openFile=open(filename)
		openFile.readline()
		lines=openFile.readlines()
		for line in lines:
			items=line.strip().split('\t')
			if not items[0] in self.dictionary:
				self.dictionary[items[0]]=len(self.dictionary)
				for idx in range(self.num_time):
					self.graphs[idx].add_node(self.dictionary[items[0]])


	def load_edge_from_txt(self, filename):
		openFile=open(filename)
		openFile.readline()
		lines=openFile.readlines()

		for line in lines:
			items=line.strip().split('\t')
			# get the interaction time
			# add edge to the temporal graph
			interact_time=time_str2float(items[2])
			if interact_time>=self.split_options.begin_time and interact_time<=self.split_options.end_time:
				# get the users id, turn phone number into id
				if not items[0] in self.dictionary:
					continue
				if not items[1] in self.dictionary:
					continue
				idx=int((interact_time-self.split_options.begin_time)/self.split_options.scale_len)
				#print(interact_time,self.split_options.begin_time,self.split_options.scale_len,idx)
				self.graphs[idx].add_edge(self.dictionary[items[0]],self.dictionary[items[1]])
			
		openFile.close()


	def load_edge_from_sql(self, tablename):
		cur=conn.cursor()
		query_edge="select distinct call_from, call_to from tablename"
		aa=cur.execute(query_edge)
		info=cur.fetchmany(aa)
		for items in info:
			if not items[0] in self.dictionary:
				self.dictionary[items[0]]=len(self.dictionary)
			if not items[1] in self.dictionary:
				self.dictionary[items[1]]=len(self.dictionary)
			self.graph.add_edge(self.dictionary[items[0]],self.dictionary[items[1]])


	def merge_edges(self):
		for i in range(0,self.num_time-1):
			self.graphs[i+1].add_edges_from(self.graphs[i].edges)


	def merge_temporal_graphs(self):
		for i in range(0,self.num_time-1):
			self.graphs[i+1].add_edges_from(self.graphs[i].edges)
		for i in range(self.num_time-1,0,-1):
			self.graphs[i-1].add_nodes_from(self.graphs[i].nodes)


	def merge_nodes_to_final_graph(self):
		for i in range(0,self.num_time-1):
			self.graphs[-1].add_nodes_from(self.graphs[i].nodes)


	def delete_edge_from_graph(self, ratio=0.3):
		delete_num=int(self.graphs[-1].number_of_edges()*ratio)
		edge_list=list(self.graphs[-1].edges)
		delete_list=random.sample(edge_list,delete_num)
		for i in range(0,self.num_time):
			self.graphs[i].remove_edges_from(delete_list)


	def save_dict(self, filename='temp/dict.txt'):
		data=json.dumps(self.dictionary)
		openFile=open(filename,'w')
		openFile.write(data)
		openFile.close()
		return filename
	

	def save_graph_as_edgelist(self,filename='temp/graph.txt'):
		data=json.dumps(list(self.graphs[-1].edges()))
		openFile=open(filename,'w')
		openFile.write(data)
		openFile.close()
		return filename


	def save_deepwalk_list(self, walks, filename='temp/walks.txt'):
		data=json.dumps(walks)
		openFile=open(filename,'w')
		openFile.write(data)
		openFile.close()
		

	def number_of_nodes(self):
		return len(self.dictionary)


	def build_deepwalk_corpus(self, num_paths, path_length, alpha=0, rand=random.Random(0)):
		walks=[]
		nodes=list(self.graph.nodes())
		for cnt in range(num_paths):
			rand.shuffle(nodes)
			for node in nodes:
				walks.extend(self.random_walk(path_length,rand=rand,alpha=alpha,start=node))
		return walks


	def build_deepwalk_list(self, num_paths, path_length, alpha=0, rand=random.Random(0)):
		# build deep walk on all graphs, the walks are stored by start node and temporal graph id
		init_walks=[[[] for j in range(self.num_time)] for i in range(len(self.dictionary))]
		for i,g in enumerate(self.graphs):
			nodes=list(g.nodes())
			for cnt in range(num_paths):
				for node in nodes:
					init_walks[node][i].append(self.random_walk(i,path_length,rand=rand,alpha=alpha,start=node))

		#self.save_deepwalk_list(init_walks,'temp/init_walks.txt')

		# generate data list: each element is a list, containing a batch of a node in continuous time
		walks=[]
		for cnt in range(num_paths):
			nodes=list(self.graphs[0].nodes)
			rand.shuffle(nodes)
			for each_node in init_walks:
				element=[]
				for each_time in each_node:
					element.append(each_time[cnt])
				walks.append(element)

		self.save_deepwalk_list(walks)

		return walks


	def random_walk(self, graph_idx, path_length, alpha=0, rand=random.Random(), start=None):
		g=self.graphs[graph_idx]
		if start:
			path=[start]
		else:
			path=[rand.choice(list(g.nodes()))]

		while len(path)<path_length:
			cur=path[-1]
			if g.degree(cur)>0:
				if rand.random()>=alpha:
					path.append(rand.choice(list(g[cur])))
				else:
					path.append(path[0])
			else:
				path.append(path[0])
		return path



def load_dict(filename):
	data=json.load(open(filename))
	return data


def load_graph_as_edgelist(filename):
	data=json.load(open(filename))
	return data


def load_deepwalk_list(filename='temp/walks.txt'):
	data=json.load(open(filename))
	return data


def main(filename1,filename2):
	split_options=GraphSplitOptions()
	gs=TemporalGraph(split_options)
	gs.load_from_txt(filename1)
	gs.load_from_txt(filename2)
	gs.merge_temporal_graphs()
	for g in gs.graphs:
		print(g.number_of_nodes(),g.number_of_edges())


if __name__=='__main__':
	main("metadata/call_info.txt","metadata/msg_info.txt")





