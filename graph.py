import sys
import os
import random
import json
import time
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

	def __init__(self, begin_time_str='2010/10/01 00:00:00', stop_time_str='2010/10/25 23:59:59'):
		self.begin_time=self.time_str2float(begin_time_str)
		self.stop_time=self.time_str2float(stop_time_str)

	def time_str2float(self, time_str):
		time_tuple=time.strptime(time_str,'%Y/%m/%d %H:%M:%S')
		time_float=time.mktime(time_tuple)
		return time_float



class Graph(object):

	def __init__(self, split_options):
		self.graph=nx.Graph()
		self.dictionary={}
		self.split_options=split_options

	def load_node_from_txt(self, filename='metadata/new_user_info.txt'):
		openFile=open(filename)
		openFile.readline()
		lines=openFile.readlines()
		for line in lines:
			items=line.strip().split('\t')
			if not items[0] in self.dictionary:
				self.dictionary[items[0]]=len(self.dictionary)
				self.graph.add_node(self.dictionary[items[0]])

	def load_edge_from_txt(self, filename):
		openFile=open(filename)
		openFile.readline()
		lines=openFile.readlines()
		for line in lines:
			items=line.strip().split('\t')
			if not items[0] in self.dictionary:
				continue
			if not items[1] in self.dictionary:
				continue
			interact_time=time.mktime(time.strptime(items[2],'%Y/%m/%d %H:%M:%S'))
			if interact_time>=self.split_options.begin_time and interact_time<=self.split_options.stop_time:			
				self.graph.add_edge(self.dictionary[items[0]],self.dictionary[items[1]])
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


	def delete_edge_from_graph(self, ratio=0.3):
		delete_num=int(self.graph.number_of_edges()*ratio)
		edge_list=list(self.graph.edges)
		delete_list=random.sample(edge_list,delete_num)
		self.graph.remove_edges_from(delete_list)


	def save_dict(self, filename='temp/dict.txt'):
		data=json.dumps(self.dictionary)
		openFile=open(filename,'w')
		openFile.write(data)
		openFile.close()
		return filename
	

	def save_graph_as_edgelist(self, filename='temp/graph.txt'):
		data=json.dumps(list(self.graph.edges()))
		openFile=open(filename,'w')
		openFile.write(data)
		openFile.close()
		return filename
		

	def number_of_nodes(self):
		return self.graph.number_of_nodes()


	def build_deepwalk_corpus(self, num_paths, path_length, alpha=0, rand=random.Random(0)):
		walks=[]
		nodes=list(self.graph.nodes())
		for cnt in range(num_paths):
			rand.shuffle(nodes)
			for node in nodes:
				walks.extend(self.random_walk(path_length,rand=rand,alpha=alpha,start=node))
		return walks


	def build_deepwalk_list(self, num_paths, path_length, alpha=0, rand=random.Random(0)):
		walks=[]
		nodes=list(self.graph.nodes())
		for cnt in range(num_paths):
			rand.shuffle(nodes)
			for node in nodes:
				walks.append(self.random_walk(path_length,rand=rand,alpha=alpha,start=node))
		return walks


	def random_walk(self, path_length, alpha=0, rand=random.Random(), start=None):
		if start:
			path=[start]
		else:
			path=[rand.choice(list(self.graph.nodes()))]

		while len(path)<path_length:
			cur=path[-1]
			if self.graph.degree(cur)>0:
				if rand.random()>=alpha:
					path.append(rand.choice(list(self.graph[cur])))
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







