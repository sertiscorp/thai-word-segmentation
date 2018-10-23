from thainlplib import ThaiWordSegmentLabeller
from thainlplib.model import ThaiWordSegmentationModel

import tensorflow as tf
import fire
import pandas as pd
import numpy as np
import preprocess

saved_model_path = 'saved_model'


BATCH_SIZE = 500


def build_sample(txt):
    encoded_txt = ThaiWordSegmentLabeller.get_input_labels(txt)
    dummy_label = [False]*len(encoded_txt)

    return preprocess.make_sequence_example(encoded_txt, dummy_label)


def main(input_csv, output_csv, separator="|", verbose=0):
    df_input = pd.read_csv(input_csv, encoding='utf-8', sep=';', names=['txt'])[:2000]

    texts = df_input.txt.values

    print('Processing samples')
    samples = df_input.txt.apply(build_sample).values

    options = tf.python_io.TFRecordOptions(compression_type=tf.python_io.TFRecordCompressionType.ZLIB)

    # todo: tmp file
    training_writer = tf.python_io.TFRecordWriter('sample.tfrecord', options=options)
    for s in samples:
        training_writer.write(s.SerializeToString())
    training_writer.close()

    tokenized_texts = []

    no_batches = int(np.ceil(len(texts)*1.0 / BATCH_SIZE))

    with tf.Session() as sess:
        model = tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], saved_model_path)
        signature = model.signature_def[tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
        graph = tf.get_default_graph()

        g_lengths = graph.get_tensor_by_name(signature.inputs['lengths'].name)
        g_training = graph.get_tensor_by_name(signature.inputs['training'].name)
        g_outputs = graph.get_tensor_by_name(signature.outputs['outputs'].name)

        g_handle = graph.get_tensor_by_name('Placeholder:0')

        dataset = tf.data.TFRecordDataset(['sample.tfrecord'], compression_type="ZLIB") \
            .map(ThaiWordSegmentationModel._parse_record) \
            .padded_batch(BATCH_SIZE, padded_shapes=([], [None], [None]))

        iter = dataset.make_initializable_iterator()

        sess.run(iter.initializer)

        ehandle = sess.run(iter.string_handle())

        probs = []
        for j in range(no_batches):
            print('Batch-%d' %(j+1))
            prob, seq = sess.run([g_outputs, g_lengths], feed_dict={g_training: False, g_handle: ehandle})

            st_idx = 0
            for l in (seq):
                st = st_idx
                sp = st_idx + l
                probs.append(prob[st:sp])

                st_idx = sp

    for t, p in zip(texts, probs):
        tokenized_text = separator.join(tokenize(t, p))
        tokenized_texts.append(tokenized_text)

    pd.DataFrame(tokenized_texts).to_csv(output_csv, index=False)


def tokenize(s, probs):
    indices = np.argwhere(probs > 0).reshape(-1).tolist()

    return [s[i:j] for i,j in zip(indices, indices[1:]+[None])]


if __name__ == '__main__':
    fire.Fire(main)

