import tensorflow as tf
import fire
import pandas as pd
import numpy as np
import preprocess
import os

from tqdm import tqdm

from thainlplib import ThaiWordSegmentLabeller
from thainlplib.model import ThaiWordSegmentationModel

SAVED_MODEL_PATH = 'saved_model'
TMP_FILE = '.tmptfrecord'


def build_sample(txt):
    encoded_txt = ThaiWordSegmentLabeller.get_input_labels(txt)
    dummy_label = [False]*len(encoded_txt)

    return preprocess.make_sequence_example(encoded_txt, dummy_label)


def main(input_path, output_path, separator="|", batch_size=250):
    df_input = pd.read_csv(input_path, encoding='utf-8', sep=';', names=['txt'])

    texts = df_input.txt.values

    print('Retrieving samples')
    samples = df_input.txt.apply(build_sample).values

    options = tf.python_io.TFRecordOptions(compression_type=tf.python_io.TFRecordCompressionType.ZLIB)

    # todo: tmp file
    training_writer = tf.python_io.TFRecordWriter(TMP_FILE, options=options)
    for s in samples:
        training_writer.write(s.SerializeToString())
    training_writer.close()

    tokenized_texts = []

    no_batches = int(np.ceil(len(texts) * 1.0 / batch_size))

    with tf.Session() as sess:
        model = tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], SAVED_MODEL_PATH)
        signature = model.signature_def[tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
        graph = tf.get_default_graph()

        g_lengths = graph.get_tensor_by_name(signature.inputs['lengths'].name)
        g_training = graph.get_tensor_by_name(signature.inputs['training'].name)
        g_outputs = graph.get_tensor_by_name(signature.outputs['outputs'].name)

        g_handle = graph.get_tensor_by_name('Placeholder:0')

        dataset = tf.data.TFRecordDataset([TMP_FILE], compression_type="ZLIB") \
            .map(ThaiWordSegmentationModel._parse_record) \
            .padded_batch(batch_size, padded_shapes=([], [None], [None]))

        data_iter = dataset.make_initializable_iterator()

        sess.run(data_iter.initializer)

        data_handle = sess.run(data_iter.string_handle())

        probs = []
        for j in tqdm(range(no_batches)):
            prob, seq = sess.run([g_outputs, g_lengths], feed_dict={g_training: False, g_handle: data_handle})

            st_idx = 0
            for l in (seq):
                st = st_idx
                sp = st_idx + l
                probs.append(prob[st:sp])

                st_idx = sp

    for t, p in zip(texts, probs):
        tokenized_text = separator.join(tokenize(t, p))
        tokenized_texts.append(tokenized_text)

    pd.DataFrame(tokenized_texts).to_csv(output_path, index=False, header=False)

    os.remove(TMP_FILE)


def tokenize(s, probs):
    indices = np.argwhere(probs > 0).reshape(-1).tolist()

    return [s[i:j] for i,j in zip(indices, indices[1:]+[None])]


if __name__ == '__main__':
    fire.Fire(main)

