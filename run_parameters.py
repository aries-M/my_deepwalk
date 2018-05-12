import os


cnt=0


task_name='colleague_relations'
for num_paths in [10]:
	for path_length in [10]:
		for embedding_size in [20]:
			for learning_rate in [1.0]:
				for decay_rate in [0.96]:
					for epochs_to_train in [30001]:
						for num_sequences in [50]:
							for num_skips in [2]:
								command='python temporal_deepwalk2.py'
								command+=(' --task_name='+str(task_name))
								command+=(' --num_paths='+str(num_paths))
								command+=(' --path_length='+str(path_length))
								command+=(' --embedding_size='+str(embedding_size))
								command+=(' --learning_rate='+str(learning_rate))
								command+=(' --decay_rate='+str(decay_rate))
								command+=(' --epochs_to_train='+str(epochs_to_train))
								command+=(' --num_sequences='+str(num_sequences))
								command+=(' --num_skips='+str(num_skips))
								command+=(' > log/results/'+str(cnt))
								cnt+=1
								os.system(command)


task_name='link_prediction'
for num_paths in [10]:
	for path_length in [10]:
		for embedding_size in [20]:
			for learning_rate in [1.0]:
				for decay_rate in [0.96]:
					for epochs_to_train in [30001]:
						for num_sequences in [50]:
							for num_skips in [2]:
								command='python temporal_deepwalk2.py'
								command+=(' --task_name='+str(task_name))
								command+=(' --num_paths='+str(num_paths))
								command+=(' --path_length='+str(path_length))
								command+=(' --embedding_size='+str(embedding_size))
								command+=(' --learning_rate='+str(learning_rate))
								command+=(' --decay_rate='+str(decay_rate))
								command+=(' --epochs_to_train='+str(epochs_to_train))
								command+=(' --num_sequences='+str(num_sequences))
								command+=(' --num_skips='+str(num_skips))
								command+=(' > log/results/'+str(cnt))
								cnt+=1
								os.system(command)
