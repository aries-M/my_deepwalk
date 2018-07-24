import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

basic_path='/Users/susanna/Desktop/pkuthss-1.7.4/pkuthss/doc/example/fig'

def draw_img(x, y, xlabel, ylabel, title, name, label):
	#pdf = PdfPages(name+'.pdf') 

	plt.figure() 
	for i in range(len(x)):
		plt.plot(x[i],y[i],"--",linewidth=1,label=label[i])  
	plt.xlabel(xlabel) 
	plt.ylabel(ylabel) 
	plt.title(title) 
	plt.legend()
	#plt.show() 
	plt.savefig(os.path.join(basic_path,name),dpi=500)
	#pdf.savefig(plt) 
	#plt.close()
	#pdf.close()

def draw_bar(value, label, xlabel, ylabel, title, name):
	plt.barh(range(len(value)),value,tick_label=label)
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	plt.title(title)
	#plt.ylim(0.8,0.9)
	plt.xlim(0.80,0.88)
	#plt.show()

	plt.tight_layout()

	plt.savefig(os.path.join(basic_path,name),dpi=500)


def for_embedding_size():
	x1=[10,20,30,40,50,60,70,80]
	y1=[0.810747663551, 0.813551402, 0.823878504673, 0.827710280374, 0.833644859813 , 0.834112149533 , 0.843457943925 , 0.844859813084]
	label1='DeepWalk'

	x2=[10,20,30,40,50,60,70,80]
	y2=[0.806542056, 0.836448598, 0.83808411215, 0.839672897196, 0.845887850467, 0.84785046729, 0.849719626168, 0.851869159]
	label2='Temporal DeepWalk'

	draw_img([x1,x2],[y1,y2],'embedding size','Accuracy','Embedding Size','embedding_size',[label1,label2])

def for_time_length():
	x1=[12,15,18,21,24]
	y1=[0.823726708075,0.826333333333,0.842291666667,0.83362745098,0.816261682243]
	label1='DeepWalk'

	x2=[12,15,18,21,24]
	y2=[0.835652173913,0.844111111111,0.846458333333,0.836568627451,0.832056074766]
	label2='Temporal DeepWalk'

	draw_img([x1,x2],[y1,y2],'time length','Accuracy','Time Length','time_length',[label1,label2])

def for_roc_curve():


	data1=json.load(open('../tmpdata/roc/deepwalk.txt'))
	x1=data1[0]
	y1=data1[1]
	label1='DeepWalk'

	data2=json.load(open('../tmpdata/roc/temporal_deepwalk.txt'))
	x2=data2[0]
	y2=data2[1]
	label2='Temporal DeepWalk'

	data3=json.load(open('../tmpdata/roc/multiscale_deepwalk.txt'))
	x3=data3[0]
	y3=data3[1]
	label3='Multi-scaled Temporal DeepWalk'

	draw_img([x1,x2,x3],[y1,y2,y3],'False Positive Rate','True Positive Rate','ROC Curve for Link Prediction','roc_curve_predict',[label1,label2,label3])


def for_time_interval():

	value=[0.836448598,0.827102803738,0.835186915888,0.819158878505,
			0.851401869159,0.841121495327,0.855140186916,0.853738317757,0.843457943925,0.85046728972,
			0.860280373832,0.855607476636,0.849065420561,0.847196261682,
			0.868224299065]
	label=['3','6','9','12','3+6','3+9','3+12','6+9','6+12','9+12','3+6+9','3+6+12','3+9+12','6+9+12','3+6+9+12']

	draw_bar(value,label,'Accuracy','Comination of Time Scale', '', 'multiscale')


if __name__ == '__main__':
	#for_embedding_size()
	#for_time_length()
	for_roc_curve()
	#for_time_interval()

