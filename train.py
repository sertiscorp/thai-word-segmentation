from thainlplib import ThaiWordSegmentLabeller, ThaiWordSegmentationModel

# Training and validation data configuration
training_data_file = '/tmp/training.tf_record'
validation_data_file = '/tmp/validation.tf_record'
vocabulary_size = ThaiWordSegmentLabeller.get_input_vocabulary_size()
num_output_labels = ThaiWordSegmentLabeller.get_output_vocabulary_size()

# Model hyperparameters
dropout = 0.50
state_size = 128
learning_rate = 0.001

# Other configuration
buffer_size = 150000 # Read all data to CPU memory
batch_size = 112 # Lower/increase this depending on your GPU memory size
validate_every_n_iterations = 100
checkpoint_path = 'checkpoints'

model = ThaiWordSegmentationModel(training_data_file, validation_data_file, buffer_size, batch_size,
                                  vocabulary_size, num_output_labels, state_size, dropout)
model.train(learning_rate, validate_every_n_iterations, checkpoint_path, restore_checkpoint=False)
