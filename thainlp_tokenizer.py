from thainlplib import ThaiWordSegmentLabeller

import tensorflow as tf
import fire
import pandas as pd
import numpy as np

saved_model_path = 'saved_model'


def main(input_csv, output_csv, separator="|", verbose=0):
    df_input = pd.read_csv(input_csv, encoding='utf-8', sep=';', names=['txt'])

    texts = df_input.txt.values
    inputs = df_input.txt.apply(ThaiWordSegmentLabeller.get_input_labels).values[:10]
    lengths = df_input.txt.apply(lambda x: len(x)).values[:10]

    tokenized_texts = []

    with tf.Session() as session:
        model = tf.saved_model.loader.load(session, [tf.saved_model.tag_constants.SERVING], saved_model_path)
        signature = model.signature_def[tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
        graph = tf.get_default_graph()

        g_inputs = graph.get_tensor_by_name(signature.inputs['inputs'].name)
        g_lengths = graph.get_tensor_by_name(signature.inputs['lengths'].name)
        g_training = graph.get_tensor_by_name(signature.inputs['training'].name)
        g_outputs = graph.get_tensor_by_name(signature.outputs['outputs'].name)

        # todo: batching
        for i, tp in enumerate(zip(inputs, lengths)):
            c_inputs = [tp[0]]
            c_lengths = [tp[1]]
            prob = session.run(g_outputs, feed_dict = {g_inputs: c_inputs, g_lengths: c_lengths, g_training: False})
            tokenized_text = separator.join(tokenize(texts[i], prob))
            tokenized_texts.append(tokenized_text)

            if verbose > 0:
                print('>>> Original:\n%s\n' % texts[i])
                print('>>> Tokenized:\n%s\n' % tokenized_text)

    pd.DataFrame(tokenized_texts).to_csv(output_csv, index=False)


def tokenize(s, probs):
    indices = np.argwhere(probs > 0).reshape(-1).tolist()

    return [s[i:j] for i,j in zip(indices, indices[1:]+[None])]


if __name__ == '__main__':
    fire.Fire(main)

