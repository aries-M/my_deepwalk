import sys
import os
import networkx as nx
import numpy as np

graph=nx.Graph()

def load_edge_from_txt(filename):
	openFile=open(filename)
	openFile.readline()
	lines=openFile.readlines()
	for line in lines:
		items=line.strip().split('\t')
		graph.add_edge(items[0],items[1])
	openFile.close()

def save_edge_to_txt(filename):
	openFile=open(filename,'w')
	for node in graph.nodes:
		openFile.write(node+'\n')
	openFile.close()

load_edge_from_txt('metadata/call_info.txt')
load_edge_from_txt('metadata/msg_info.txt')
save_edge_to_txt('metadata/new_user_info.txt')
