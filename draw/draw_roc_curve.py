import matplotlib.pyplot as plt

def draw():
	plt.plot(fpr, tpr)
	plt.xlabel('False Positive Rate')  
	plt.ylabel('True Positive Rate')  
	plt.title('ROC Curve')
	plt.savefig(os.path.join(save_path,'roc.png'))
