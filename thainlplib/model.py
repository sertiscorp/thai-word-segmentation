import tensorflow as tf
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
import os

class ThaiWordSegmentationModel:
    # Parse each sentence of input and output labels and length of the sentence from TFRecord file
    @staticmethod
    def _parse_record(example_proto):
        context_features = {
            "length": tf.FixedLenFeature([], dtype=tf.int64)
        }
        sequence_features = {
            "tokens": tf.FixedLenSequenceFeature([], dtype=tf.int64),
            "labels": tf.FixedLenSequenceFeature([], dtype=tf.int64)
        }

        context_parsed, sequence_parsed = tf.parse_single_sequence_example(serialized=example_proto,
            context_features=context_features, sequence_features=sequence_features)

        return context_parsed['length'], sequence_parsed['tokens'], sequence_parsed['labels']

    # Read training data from TFRecord file, shuffle, loop over data infinitely and
    # pad to the longest sentence
    @staticmethod
    def _read_training_dataset(data_file, batch_size, buffer_size=10000):
        return tf.data.TFRecordDataset([data_file], compression_type="ZLIB") \
            .map(ThaiWordSegmentationModel._parse_record) \
            .shuffle(buffer_size) \
            .repeat() \
            .padded_batch(batch_size, padded_shapes=([], [None], [None]))

    # Read validation data from TFRecord file and pad to the longest sentence
    @staticmethod
    def _read_validation_dataset(data_file, batch_size):
        return tf.data.TFRecordDataset([data_file], compression_type="ZLIB") \
            .map(ThaiWordSegmentationModel._parse_record) \
            .padded_batch(batch_size, padded_shapes=([], [None], [None]))
    
    # Get iterators for training and validation data, a handle variable for alternating
    # between the iterators and a data batch
    @staticmethod
    def _init_iterators(training_dataset, validation_dataset):
        handle = tf.placeholder(tf.string, shape=[])
        iterator = tf.data.Iterator.from_string_handle(handle,
                                                       training_dataset.output_types,
                                                       training_dataset.output_shapes)
        batch = iterator.get_next()
        
        training_iterator = training_dataset.make_one_shot_iterator()
        validation_iterator = validation_dataset.make_initializable_iterator()
        
        return training_iterator, validation_iterator, handle, batch
    
    # Build layers for character embeddings and bi-directional RNN with GRU cells
    @staticmethod
    def _build_embedding_rnn(tokens, lengths, vocabulary_size, state_size, dropout):
        embedding_weights = tf.Variable(tf.random_uniform([vocabulary_size, state_size], -1.0, 1.0))
        embedding_vectors = tf.nn.embedding_lookup(embedding_weights, tokens)

        cell = tf.nn.rnn_cell.GRUCell(state_size)
        cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=1-dropout)

        (forward_output, backward_output), _ = \
            tf.nn.bidirectional_dynamic_rnn(cell, cell, inputs=embedding_vectors,
                                            sequence_length=lengths, dtype=tf.float32)
        outputs = tf.concat([forward_output, backward_output], axis=2)
        
        return embedding_vectors, outputs
    
    # Build fully connected output layer and postprocess output data
    @staticmethod
    def _build_classifier(inputs, labels, lengths, num_output_labels):
        logits = tf.layers.dense(inputs=inputs, units=num_output_labels, activation=None)
        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
        
        mask = tf.sequence_mask(lengths)
        loss = tf.reduce_mean(tf.boolean_mask(losses, mask))
        masked_prediction = tf.boolean_mask(tf.argmax(logits, axis=2), mask)
        masked_labels = tf.boolean_mask(labels, mask)
        
        return loss, mask, masked_prediction, masked_labels
    
    # Define training optimizer to minimize prediction loss
    @staticmethod
    def _build_optimizer(loss):
        global_step = tf.Variable(0, trainable=False)
        learning_rate = tf.placeholder(tf.float32, shape=[])
        optimizer = tf.contrib.layers.optimize_loss(loss=loss, global_step=global_step,
            learning_rate=learning_rate, optimizer='Adam')
        return optimizer, global_step, learning_rate

    @staticmethod
    def _build_graph(tokens, labels, lengths, vocabulary_size, num_output_labels, dropout, state_size):
        training = tf.placeholder(tf.bool, shape=[])
        
        # Embedding and bidirectional RNN layer
        _, rnn_outputs = ThaiWordSegmentationModel._build_embedding_rnn(lengths=lengths, tokens=tokens,
            vocabulary_size=vocabulary_size, state_size=state_size, 
            dropout=tf.cast(training, tf.float32) * dropout)
        
        # Output layer
        loss, mask, masked_prediction, masked_labels = \
            ThaiWordSegmentationModel._build_classifier(inputs=rnn_outputs, labels=labels,
            lengths=lengths, num_output_labels=num_output_labels)
        
        # Optimizer
        optimizer, global_step, learning_rate = ThaiWordSegmentationModel._build_optimizer(loss=loss)
        
        return loss, masked_prediction, masked_labels, optimizer, global_step, learning_rate, training
    
    def __init__(self, training_data_file, validation_data_file, buffer_size, batch_size, vocabulary_size, 
                 num_output_labels, state_size, dropout):
        tf.reset_default_graph()

        # Load training and validation data sets
        training_dataset = ThaiWordSegmentationModel._read_training_dataset(training_data_file, batch_size, buffer_size)
        validation_dataset = ThaiWordSegmentationModel._read_validation_dataset(validation_data_file, batch_size)
        
        # Initialize training and validation data iterators
        self.tf_training_iterator, self.tf_validation_iterator, self.tf_iterator_handle, tf_batch = \
            ThaiWordSegmentationModel._init_iterators(training_dataset=training_dataset, 
                                                      validation_dataset=validation_dataset)
        # Break the batch into a batch of each variable
        self.tf_lengths_batch, self.tf_tokens_batch, self.tf_labels_batch = tf_batch

        # Build the neural graph
        self.tf_loss, self.tf_masked_prediction, self.tf_masked_labels, self.tf_optimizer, \
            self.tf_global_step, self.tf_learning_rate, self.tf_training = \
            ThaiWordSegmentationModel._build_graph(tokens=self.tf_tokens_batch, labels=self.tf_labels_batch,
            lengths=self.tf_lengths_batch, vocabulary_size=vocabulary_size, num_output_labels=num_output_labels,
            dropout=dropout, state_size=state_size)
    
    # Save the trained model to a file in case you want to stop and continue training
    # (with different hyperparameters)
    def _restore_checkpoint(self, session, saver, checkpoint_path):
        saver.restore(session, tf.train.get_checkpoint_state(checkpoint_path).model_checkpoint_path)
        return session.run(self.tf_global_step)
    
    # Measure accuracy of the word boundary prediction using precision, recall and F1 score
    @staticmethod
    def _evaluate(tag, iteration, loss, observed, predicted):
        precision = precision_score(observed, predicted) * 100
        recall = recall_score(observed, predicted) * 100
        f1 = f1_score(observed, predicted) * 100
        print('{}: Iteration {}, Loss {:.5f}, Precision {:2.2f}, Recall {:2.2f}, F1 {:2.2f}' \
              .format(tag, iteration, loss, precision, recall, f1))
        return precision, recall, f1

    def train(self, learning_rate, validate_every_n_iterations, checkpoint_path, restore_checkpoint=False):
        # Start session and initialize variables
        global_init_op = tf.global_variables_initializer()
        session = tf.Session()
        session.run(global_init_op)
        
        # Get handles of training and validation iterators
        training_handle = session.run(self.tf_training_iterator.string_handle())
        validation_handle = session.run(self.tf_validation_iterator.string_handle())
        
        last_global_step = 0
        saver = tf.train.Saver(pad_step_number=True)
        
        # Restore checkpoint if requested
        if restore_checkpoint == True:
            last_global_step = self._restore_checkpoint(session=session, saver=saver, 
                                                        checkpoint_path=checkpoint_path)
        
        # Train forever, hence stop the training when you are happy with the result
        iteration = last_global_step
        while True:
            for _ in range(validate_every_n_iterations):
                iteration += 1
                labels_batch, loss, masked_labels, masked_prediction, optimizer = \
                    session.run([self.tf_labels_batch, self.tf_loss, self.tf_masked_labels,
                                 self.tf_masked_prediction, self.tf_optimizer],
                    feed_dict = {self.tf_iterator_handle: training_handle, self.tf_training: True, 
                                 self.tf_learning_rate: learning_rate})
                ThaiWordSegmentationModel._evaluate('Training', iteration, loss, masked_labels, masked_prediction)

            # Validate and save checkpoint
            masked_labels_all = np.empty([0], dtype=int)
            masked_prediction_all = np.empty([0], dtype=int)
            session.run(self.tf_validation_iterator.initializer)
            try:
                while True:
                    labels_batch, loss, masked_labels, masked_prediction = \
                        session.run([self.tf_labels_batch, self.tf_loss, self.tf_masked_labels,
                                     self.tf_masked_prediction],
                        feed_dict = {self.tf_iterator_handle: validation_handle, self.tf_training: False})
                    masked_labels_all = np.append(masked_labels_all, masked_labels)
                    masked_prediction_all = np.append(masked_prediction_all, masked_prediction)
            except tf.errors.OutOfRangeError:
                _, _, f1 = ThaiWordSegmentationModel._evaluate('Validation', iteration, loss, masked_labels_all, 
                                                               masked_prediction_all)
                saver.save(session, os.path.join(checkpoint_path, 'model_{:2.2f}'.format(f1)), global_step=iteration)

        session.close()

    # Read checkpoint and save model to file for predictions
    def save_model(self, checkpoint_path, saved_model_path,
                   signature_name = tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY):
        
        with tf.Session() as session:
            global_init_op = tf.global_variables_initializer()
            session.run(global_init_op)
            
            saver = tf.train.Saver()
            saver.restore(session, checkpoint_path)

            inputs = {'inputs': tf.saved_model.utils.build_tensor_info(self.tf_tokens_batch),
                      'lengths': tf.saved_model.utils.build_tensor_info(self.tf_lengths_batch),
                      'training': tf.saved_model.utils.build_tensor_info(self.tf_training)}
            outputs = {'outputs': tf.saved_model.utils.build_tensor_info(self.tf_masked_prediction)}

            builder = tf.saved_model.builder.SavedModelBuilder(saved_model_path)
            prediction_signature = (tf.saved_model.signature_def_utils.build_signature_def(
                inputs = inputs,
                outputs = outputs,
                method_name = tf.saved_model.signature_constants.PREDICT_METHOD_NAME
            ))
            builder.add_meta_graph_and_variables(
                session,
                [tf.saved_model.tag_constants.SERVING],
                signature_def_map={
                  signature_name: prediction_signature
                }
            )
            
            builder.save()
