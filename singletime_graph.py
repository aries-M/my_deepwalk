import sys
import os
import random
import json
import time
import numpy as np 
import pymysql as sql 
import networkx as nx 
from networkx.readwrite import json_graph

from graph import Graph

class SingleTimeGraph(Graph):

	def __init__(self, nodes, edges):
		self.graph=nx.Graph()
		self.dictionary={}

	def load_from_txt(self, filename, begin_date, end_date):
		openFile=open(filename)
		openFile.readline()
		lines=openFile.readlines()
		for line in lines:
			items=line.strip().split('\t')
			if not items[0] in self.dictionary:
				self.dictionary[items[0]]=len(self.dictionary)
			if not items[1] in self.dictionary:
				self.dictionary[items[1]]=len(self.dictionary)
			interact_time=time.strptime(items[2],'%Y/%m/%d %H:%M:%S')
			interact_date=interact_time.tm_mday-1
			if interact_date>=begin_date and interact_date<end_date:
				self.graph.add_edge(self.dictionary[items[0]],self.dictionary[items[1]])
		openFile.close()

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


	def merge_graphs(self, nodes, edges):
		self.graph.add_nodes_from(nodes)
		self.graph.add_edges_from(edges)



def load_dict(filename):
	data=json.load(open(filename))
	return data


def load_graph_as_edgelist(filename):
	data=json.load(open(filename))
	return data

