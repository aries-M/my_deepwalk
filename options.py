import os
import sys
import tensorflow as tf

flags = tf.app.flags

flags.DEFINE_string("save_path", "log/", "Directory to write the model and "
                    "training summaries.")
flags.DEFINE_string("data1", "metadata/call_info.txt", "Training text file. ")
flags.DEFINE_string("data2", "metadata/msg_info.txt", "Training text file. ")
flags.DEFINE_string("task_name", "link_prediction", "Task using embeddings for testing. ")
flags.DEFINE_integer("num_paths", 10, "Number of random walks to start at each node. ")
flags.DEFINE_integer("path_length", 10, "Length of the random walk started at each node. ")
flags.DEFINE_integer("embedding_size", 20, "The embedding dimension size.")
flags.DEFINE_integer(
    "epochs_to_train", 20001,
    "Number of epochs to train. Each epoch processes a batch of the training data.")
flags.DEFINE_float("learning_rate", 1.0, "Initial learning rate.")
flags.DEFINE_float("decay_rate", 0.96, "Decay learning rate.")
flags.DEFINE_integer("num_neg_samples", 10,
                     "Negative samples per training example.")
flags.DEFINE_integer("hidden_size", 20, "Number of cells in hidden layer for LSTM. ")
flags.DEFINE_integer("batch_size", 50,
                     "Number of training examples processed per step "
                     "(size of a minibatch).")
flags.DEFINE_integer("num_sequences", 50,
                     "Number of sequences when training RNN/LSTM per step .")
flags.DEFINE_integer("window_size", 5,
                     "The number of words to predict to the left and right "
                     "of the target word.")
flags.DEFINE_integer("num_skips", 2, "How many times to reuse an input to generate a label.")
flags.DEFINE_integer("statistics_interval", 5,
                     "Print statistics every n seconds.")
flags.DEFINE_integer("summary_interval", 5,
                     "Save training summary to file every n seconds (rounded "
                     "up to statistics interval).")
flags.DEFINE_integer("checkpoint_interval", 600,
                     "Checkpoint the model (i.e. save the parameters) every n "
                     "seconds (rounded up to statistics interval).")
flags.DEFINE_integer("begin_time", 0,
                     "Time length of graph.")

FLAGS = flags.FLAGS

class Options(object):

    def __init__(self):

        self.data1=FLAGS.data1
        self.data2=FLAGS.data2
        self.task_name=FLAGS.task_name
        self.num_paths=FLAGS.num_paths
        self.path_length=FLAGS.path_length
        self.embedding_size=FLAGS.embedding_size
        self.num_sampled=FLAGS.num_neg_samples
        self.hidden_size=FLAGS.hidden_size
        self.learning_rate=FLAGS.learning_rate
        self.decay_rate=FLAGS.decay_rate
        self.epochs_to_train=FLAGS.epochs_to_train
        self.batch_size=FLAGS.batch_size
        self.num_sequences=FLAGS.num_sequences
        self.window_size=FLAGS.window_size
        self.num_skips=FLAGS.num_skips
        self.statistics_interval=FLAGS.statistics_interval
        self.summary_interval=FLAGS.summary_interval
        self.checkpoint_interval = FLAGS.checkpoint_interval
        self.save_path=FLAGS.save_path
        self.begin_time=FLAGS.begin_time
        if len(self.save_path)>0 and not os.path.exists(self.save_path):
            os.makedirs(self.save_path)











