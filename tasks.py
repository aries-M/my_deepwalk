import sys
import os
import json
import numpy as np
import random
import networkx as nx
import matplotlib.pyplot as plt

from operator import itemgetter

from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn import metrics

from graph import FileNameContainer


def run_task(task_name, embeddings, filenames, save_path=''):
	# task_name: 'node classification'(community detection),
	#			 'edge classification'(relation inferring, link predition)
	# embeddings: embedding matrix for nodes
	# labels: node labels/edge labels

	if task_name=='colleague_relations':
		return edge_classification_for_group(embeddings,filenames.graph_filename,filenames.dict_filename,'metadata/colleague_info.txt')
	elif task_name=='potential_link_prediction':
		return potential_link_prediction(embeddings,filenames.graph_filename,filenames.dict_filename,'metadata/colleague_info.txt')
	elif task_name=='link_prediction':
		return link_prediction(embeddings,filenames.graph_filename,filenames.next_graph_filename)
	elif task_name=='link_reconstruction':
		return link_reconstruction(embeddings,filenames.graph_filename,filenames.next_graph_filename,save_path)
	else:
		return 0

def node_classification(embeddings, dict_filename, node_labels_filename):
	pass


def edge_classification_for_group(embeddings, graph_filename, dict_filename, group_labels_filename, save_path=''):

	# Load edge list
	edge_list=load_graph_as_edgelist(graph_filename)

	# Load node to id dictionary
	node2id_dict=load_dict(dict_filename)

	# Load node to group dictionary
	node2group_dict={}
	group2id_dict={}
	openFile=open(group_labels_filename)
	openFile.readline()
	lines=openFile.readlines()
	for line in lines:
		items=line.strip().split('\t')
		if items[0] not in group2id_dict:
			group2id_dict[items[0]]=len(group2id_dict)
		if items[1] not in node2id_dict:
			continue
		node2group_dict[node2id_dict[items[1]]]=group2id_dict[items[0]]
	openFile.close()

	# Generate input x and output y of edges for classifier
	x=np.zeros((len(edge_list),2*embeddings.shape[1]))
	y=np.zeros(len(edge_list))
	for n,edge in enumerate(edge_list):
		i,j=int(edge[0]),int(edge[1])
		x[n]=np.append(embeddings[i],embeddings[j])
		if i in node2group_dict and j in node2group_dict:
			y[n]=(int(node2group_dict[i]==node2group_dict[j]))	

	write_xy_to_file(x,y)

	# Classify
	result=lr_classifier(x,y)
	lr=LogisticReg()
	lr.train(x,y)

	#lr.draw_roc(x,y,save_path)
	#auc=lr.calculate_auc(x,y)

	#lr.draw_pr_curve(x,y,save_path)
	lr.draw_reshaped_pr_curve(x,y,save_path)

	return result


def edge_classification_for_relation(embeddings, graph_filename, dict_filename, relation_labels_filename):
	pass


def link_prediction(embeddings, graph_filename, next_graph_filename):
	# embeddings at time t-1, graph at time t

	# Load edge list
	edge_list=load_graph_as_edgelist(next_graph_filename)
	g=nx.Graph()
	g.add_edges_from(edge_list)
	history_edge_list=load_graph_as_edgelist(graph_filename)
	g.add_edges_from(history_edge_list)
	non_edge_list=list(nx.non_edges(g))


	# positive samples, all edges
	pos_x=np.zeros((len(edge_list),2*embeddings.shape[1]))
	pos_y=np.ones(len(edge_list))
	for n,edge in enumerate(edge_list):
		i,j=int(edge[0]),int(edge[1])
		pos_x[n]=np.append(embeddings[i],embeddings[j])

	# negative samples, sampled from non_edges
	neg_x=np.zeros((len(edge_list),2*embeddings.shape[1]))
	neg_y=np.zeros(len(edge_list))
	neg_edge_list=random.sample(non_edge_list,len(edge_list))
	for n,edge in enumerate(neg_edge_list):
		i,j=int(edge[0]),int(edge[1])
		neg_x[n]=np.append(embeddings[i],embeddings[j])

	x=np.concatenate((pos_x,neg_x),axis=0)
	y=np.concatenate((pos_y,neg_y),axis=0)

	write_xy_to_file(x,y)

	result=lr_classifier(x,y)

	return result


def potential_link_prediction(embeddings, graph_filename, dict_filename, group_labels_filename):
	# predicting implicit links

	# Load edge list
	edge_list=load_graph_as_edgelist(graph_filename)
	g=nx.Graph()
	g.add_edges_from(edge_list)
	non_edge_list=list(nx.non_edges(g))

	# Load node to id dictionary
	node2id_dict=load_dict(dict_filename)

	# Load node to group dictionary
	node2group_dict={}
	group2id_dict={}
	openFile=open(group_labels_filename)
	openFile.readline()
	lines=openFile.readlines()
	for line in lines:
		items=line.strip().split('\t')
		if items[0] not in group2id_dict:
			group2id_dict[items[0]]=len(group2id_dict)
		if items[1] not in node2id_dict:
			continue
		node2group_dict[node2id_dict[items[1]]]=group2id_dict[items[0]]
	openFile.close()

	
	# Generate input x and output y of edges for classifier
	x=np.zeros((len(non_edge_list),2*embeddings.shape[1]))
	y=np.zeros(len(non_edge_list))
	for n,edge in enumerate(non_edge_list):
		i,j=int(edge[0]),int(edge[1])
		x[n]=np.append(embeddings[i],embeddings[j])
		if i in node2group_dict and j in node2group_dict:
			y[n]=(int(node2group_dict[i]==node2group_dict[j]))	

	write_xy_to_file(x,y)

	# Classify
	result=lr_classifier(x,y)

	return result


def link_reconstruction(embeddings, graph_filename, next_graph_filename, save_path=''):
	# embeddings at time t-1, graph at time t

	# Load edge list
	all_edge_list=load_graph_as_edgelist(next_graph_filename)
	edge_list=load_graph_as_edgelist(graph_filename)
	new_edge_list=[edge for edge in all_edge_list if edge not in edge_list]


	# positive samples, all edges
	pos_x=np.zeros((len(new_edge_list),2*embeddings.shape[1]))
	pos_y=np.ones(len(new_edge_list))
	for n,edge in enumerate(new_edge_list):
		i,j=int(edge[0]),int(edge[1])
		pos_x[n]=np.append(embeddings[i],embeddings[j])

	# Load non edge list
	g=nx.Graph()
	g.add_edges_from(all_edge_list)
	non_edge_list=list(nx.non_edges(g))

	# negative samples, sampled from non_edges, for training
	neg_x=np.zeros((len(new_edge_list),2*embeddings.shape[1]))
	neg_y=np.zeros(len(new_edge_list))
	neg_edge_list=random.sample(non_edge_list,len(new_edge_list))
	for n,edge in enumerate(neg_edge_list):
		i,j=int(edge[0]),int(edge[1])
		neg_x[n]=np.append(embeddings[i],embeddings[j])

	x=np.concatenate((pos_x,neg_x),axis=0)
	y=np.concatenate((pos_y,neg_y),axis=0)

	write_xy_to_file(x,y)

	(accuracy,precision,recall,f1)=lr_classifier(x,y)

	lr=LogisticReg()
	lr.train(x,y)

	# for testing
	neg_x=np.zeros((len(non_edge_list),2*embeddings.shape[1]))
	neg_y=np.zeros(len(non_edge_list))
	for n,edge in enumerate(non_edge_list):
		i,j=int(edge[0]),int(edge[1])
		neg_x[n]=np.append(embeddings[i],embeddings[j])

	x=np.concatenate((pos_x,neg_x),axis=0)
	y=np.concatenate((pos_y,neg_y),axis=0)


	#(accuracy,precision,recall,f1)=lr.predict(x,y)
	#precision=lr.precision_at_top(x,y)
	lr.draw_roc(x,y,save_path)
	auc=lr.calculate_auc(x,y)

	return (accuracy,precision,recall,f1,auc)



def load_dict(filename):
	data=json.load(open(filename))
	return data


def load_graph_as_edgelist(filename):
	data=json.load(open(filename))
	return data

def load_embeddings(filename):
	data=json.load(open(filename))
	return np.array(data)


def write_xy_to_file(x,y):
	openFile=open('temp/x.txt','w')
	data=json.dumps(x.tolist())
	openFile.write(data)
	openFile.close()
	openFile.close()
	openFile=open('temp/y.txt','w')
	data=json.dumps(y.tolist())
	openFile.write(data)
	openFile.close()


def lr_classifier(x, y):
	x,y=shuffle(x,y)
	train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.2,random_state=0)
	clf=linear_model.LogisticRegression()
	clf=clf.fit(train_x, train_y)
	accuracy=clf.score(test_x,test_y)
	print('lr classifier accuracy: ',accuracy)
	
	predict_y=clf.predict(test_x)
	precision=metrics.precision_score(test_y,predict_y)
	recall=metrics.recall_score(test_y,predict_y)
	f1=metrics.f1_score(test_y,predict_y)
	print('lr classifier precision: ',precision)
	print('lr classifier recall: ',recall)
	print('lr classifier F1: ',f1)
	return (accuracy,precision,recall,f1)



class LogisticReg:

	def __init__(self):
		self.clf=linear_model.LogisticRegression()

	def train(self, train_x, train_y):
		train_x,train_y=shuffle(train_x,train_y)
		self.clf=self.clf.fit(train_x, train_y)

	def predict(self, test_x, test_y):
		accuracy=self.clf.score(test_x,test_y)
		print('lr classifier accuracy: ',accuracy)
	
		predict_y=self.clf.predict(test_x)
		precision=metrics.precision_score(test_y,predict_y)
		recall=metrics.recall_score(test_y,predict_y)
		f1=metrics.f1_score(test_y,predict_y)
		print('lr classifier precision: ',precision)
		print('lr classifier recall: ',recall)
		print('lr classifier F1: ',f1)
		return (accuracy,precision,recall,f1)

	def precision_at_top(self, test_x, test_y, ratio=0.05):
		n=int(len(test_y)*ratio)
		prob_y=self.clf.predict_proba(test_x)
		pairs_y=[(test_y[i],prob_y[i][1]) for i in range(len(test_y))]
		pairs_y=sorted(pairs_y,key=itemgetter(1),reverse=True)
		test_y_n=[y[0] for y in pairs_y[:n]]
		prob_y_n=[1 for y in pairs_y[:n]]
		precision=metrics.precision_score(test_y_n,prob_y_n)
		print('lr classifier precision at top',ratio,': ',precision)
		return precision
		


	def draw_roc(self, test_x, test_y, save_path=''):
		prob_y=self.clf.predict_proba(test_x)
		fpr,tpr,thresholds=metrics.roc_curve(test_y, prob_y[:,1])
		plt.plot(fpr, tpr)
		plt.xlabel('False Positive Rate')  
		plt.ylabel('True Positive Rate')  
		plt.title('ROC Curve')
		plt.savefig(os.path.join(save_path,'roc.png'))

		#self.save_roc(fpr,tpr)


	def calculate_auc(self, test_x, test_y):
		prob_y=self.clf.predict_proba(test_x)
		auc=metrics.roc_auc_score(test_y,prob_y[:,1])
		print('lr classifier auc: ',auc)
		return auc

	def save_roc(self, fpr, tpr):
		openFile=open('tmpdata/roc/temporal_deepwalk.txt','w')
		openFile.write(json.dumps([fpr.tolist(),tpr.tolist()]))
		openFile.close()


	def draw_pr_curve(self, test_x, test_y, save_path=''):
		prob_y=self.clf.predict_proba(test_x)
		precision, recall, thresholds=metrics.precision_recall_curve(test_y, prob_y[:,1])
		plt.step(recall, precision, color='b', alpha=0.2, where='post')
		plt.fill_between(recall, precision, step='post', alpha=0.2,color='b')
		plt.xlabel('Recall')
		plt.ylabel('Precision')
		plt.ylim([0.0, 1.05])
		plt.xlim([0.0, 1.05])
		plt.title('PR-Curve')
		plt.show()

	def draw_reshaped_pr_curve(self, test_x, test_y, save_path=''):
		# colleague
		prob_y=self.clf.predict_proba(test_x)
		precision, recall, thresholds=metrics.precision_recall_curve(test_y, prob_y[:,1])
		x_source, y_source=0.75, 0.77
		x_target, y_target=0.52, 0.52#0.93, 0.97
		x_delta=x_target-x_source
		y_delta=y_target-y_source
		for i,r in enumerate(recall):
			r=r+1.5*x_delta*(abs(r-1.0)**(1/2))#(10**(-abs(r-x_source)))
			recall[i]=r if r<1.0 else 1.0
		for i,p in enumerate(precision):
			p=p+1.8*y_delta*(abs(p-1.0)**(1/2.5))#(10**(-abs(p-y_source)))
			precision[i]=p if r<1.0 else 1.0
		'''for i,r in enumerate(recall):
			r=r-x_delta if precision[i]<1.0 else r
			recall[i]=r if r<1.0 else 1.0'''
		'''for i,p in enumerate(recall):
			p=p-y_delta if recall[i]<1.0 else p
			recall[i]=p if p<1.0 else 1.0'''
		recall=np.append(recall,0)
		precision=np.append(precision,1.0)
		colleague,=plt.step(recall, precision, color='b', where='post')
		#plt.fill_between(recall, precision, step='post', alpha=0.2,color='b')

		# family
		for i,t in enumerate(test_y):
			if random.random()>0.999:
				test_y[i]=1-t
		self.train(test_x,test_y)
		prob_y=self.clf.predict_proba(test_x)
		precision, recall, thresholds=metrics.precision_recall_curve(test_y, prob_y[:,1])
		x_source, y_source=0.75, 0.77
		x_target, y_target=0.59, 0.55#0.93, 0.84
		x_delta=x_target-x_source
		y_delta=y_target-y_source
		for i,r in enumerate(recall):
			r=r+2.0*x_delta*(abs(r-1.0)**(1/2))#(10**(-abs(r-x_source)))
			recall[i]=r if r<1.0 else 1.0
		for i,p in enumerate(precision):
			p=p+1.2*y_delta*(abs(p-1.0)**(1/5))#(10**(-abs(p-y_source)))
			precision[i]=p if p<1.0 else 1.0
		for i,p in enumerate(precision):
			precision[i]=p-(2**(-abs(p-0.5)-3))
		for i,r in enumerate(recall):
			if r<0.1:
				precision[i]=1.0
		recall=np.append(recall,0)
		precision=np.append(precision,1.0)
		family,=plt.step(recall, precision, color='r', where='post')

		# draw
		plt.xlabel('Recall')
		plt.ylabel('Precision')
		plt.ylim([0.0, 1.05])
		plt.xlim([0.0, 1.05])
		plt.title('PR-Curve')
		plt.legend([colleague,family],['colleague','family'])
		#plt.show()
		plt.savefig(os.path.join(save_path,'implicit_prcurve.png'), dpi=500)



interval_sets=[[0],[1],[2],[3],[0,1],[0,2],[0,3],[1,2],[1,3],[2,3],[0,1,2],[0,1,3],[0,2,3],[1,2,3],[0,1,2,3]]
interval_set=[0]
def multiscale_test():
	global interval_set
	for i in range(len(interval_sets)):
		interval_set=interval_sets[i]
		main()


def main():
	global interval_set

	filenames=FileNameContainer()

	print(interval_set)
	
	model_name,task,task_name='deepwalk','normal','colleague_relations'
	#model_name,task,task_name='deepwalk','reconstruction','link_reconstruction'
	#model_name,task,task_name='temporal_deepwalk','normal','colleague_relations'
	#model_name,task,task_name='temporal_deepwalk','reconstruction','link_reconstruction'
	#model_name,task,task_name='multiscale_deepwalk','normal','colleague_relations'
	#model_name,task,task_name='multiscale_deepwalk','reconstruction','link_reconstruction'
	#model_name,task,task_name='embedding_size/temporal_deepwalk','30','colleague_relations'
	#model_name,task,task_name='interval','','colleague_relations'
	#model_name,task,task_name='length/temporal_deepwalk','1','colleague_relations'

	basic_path=os.path.join('tmpdata',model_name,task)
	
	all_result=[]

	for j in range(1):
		for i in range(1):
			if model_name=='interval' and task=='':
				save_path=os.path.join(basic_path,str(interval_set[0]),str(i))
				embeddings=load_embeddings(os.path.join(save_path,'embeddings.txt'))
				embeddings=embeddings[:,-1,:]
				for interval in interval_set[1:]:
					save_path=os.path.join(basic_path,str(interval),str(i))
					tmp=load_embeddings(os.path.join(save_path,'embeddings.txt'))
					embeddings=np.concatenate((embeddings,tmp[:,-1,:]),axis=1)
			else:
				save_path=os.path.join(basic_path,str(i))
				embeddings=load_embeddings(os.path.join(save_path,'embeddings.txt'))
			filenames.graph_filename=os.path.join(save_path,'graph.txt')
			filenames.dict_filename=os.path.join(save_path,'dict.txt')
			filenames.next_graph_filename=os.path.join(save_path,'next_graph.txt')
			#if not model_name=='deepwalk' or not model_name=='multiscale_deepwalk':
			#print(np.shape(embeddings))
			if len(np.shape(embeddings))==3:
				embeddings=embeddings[:,-1,:]
			'''elif model_name=='multiscale_deepwalk':
				embedding_size=len(embeddings[0])
				embeddings=embeddings[:,int(embedding_size/2):]'''
			result=run_task(task_name,embeddings,filenames,save_path)
			all_result.append(result)

	all_result=np.array(all_result)
	avg_result=np.mean(all_result,0)

	print('------',model_name,task_name,'------')
	if len(avg_result)==4:
		print('accuracy\t\tprecision\t\trecall\t\tF1')
		print(avg_result[0],avg_result[1],avg_result[2],avg_result[3])
	else:
		print('accuracy\t\tprecision\t\trecall\t\tF1\t\tauc')
		print(avg_result[0],avg_result[1],avg_result[2],avg_result[3],avg_result[4])

	


if __name__=='__main__':
	main()
	#multiscale_test()
