from thainlplib import ThaiWordSegmentLabeller
from os import listdir
from os.path import isfile, join
import random
import tensorflow as tf

# Split tokens
def process_line(line):
    inputs = []
    outputs = []
    for token in line.split('|'):
        if len(token) == 0: continue
        inputs += ThaiWordSegmentLabeller.get_input_labels(token)
        outputs += ThaiWordSegmentLabeller.get_output_labels(token)
    return inputs, outputs

# Create a record for a sentence
def make_sequence_example(sequence, labels):
    token_features = [tf.train.Feature(int64_list=tf.train.Int64List(value=[x])) for x in sequence]
    label_features = [tf.train.Feature(int64_list=tf.train.Int64List(value=[x])) for x in labels]
    example = tf.train.SequenceExample(
        context = tf.train.Features(feature={
            'length': tf.train.Feature(int64_list=tf.train.Int64List(value=[len(sequence)]))
        }),
        feature_lists = tf.train.FeatureLists(feature_list={
            'tokens': tf.train.FeatureList(feature=token_features),
            'labels': tf.train.FeatureList(feature=label_features)
        })
    )
    return example

# Read input data line by line and split to training and validation TFRecord files
def preprocess_files(input_files, training_output_file, validation_output_file, training_proportion):
    options = tf.python_io.TFRecordOptions(compression_type=tf.python_io.TFRecordCompressionType.ZLIB)
    training_writer = tf.python_io.TFRecordWriter(training_output_file, options=options)
    validation_writer = tf.python_io.TFRecordWriter(validation_output_file, options=options)
    file_number = 1
    for input_filename in input_files:
        print('Processing file {}/{}...'.format(file_number, len(input_files)))
        file_number += 1
        input_file = open(input_filename)
        for line in input_file.readlines():
            x, y = process_line(line)
            p = random.random()
            example = make_sequence_example(x, y)
            if p <= training_proportion:
                training_writer.write(example.SerializeToString())
            else:
                validation_writer.write(example.SerializeToString())
        input_file.close()
    training_writer.close()
    validation_writer.close()

def flatten_list(l):
    return [item for sublist in l for item in sublist]

def list_files(paths):
    return flatten_list([[path + '/' + file for file in listdir(path) if isfile(join(path, file))] for path in paths])

# Download and store the BEST corpus into data directory first
# Read and shuffle input files
files = list_files(['data/article', 'data/encyclopedia', 'data/news', 'data/novel'])
random.shuffle(files)

# Set the locations of the output files here
training_data_file = '/tmp/training.tf_record'
validation_data_file = '/tmp/validation.tf_record'

# Preprocess and split each sentence to training and validation files
preprocess_files(files, training_data_file, validation_data_file, .9)
print("Done")
