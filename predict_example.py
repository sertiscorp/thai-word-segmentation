from thainlplib import ThaiWordSegmentLabeller
import numpy as np
import tensorflow as tf

# Pretrained model weights location
saved_model_path='saved_model'

# Input text
text = """ประเทศไทยรวมเลือดเนื้อชาติเชื้อไทย
เป็นประชารัฐไผทของไทยทุกส่วน
อยู่ดำรงคงไว้ได้ทั้งมวล
ด้วยไทยล้วนหมายรักสามัคคี
ไทยนี้รักสงบแต่ถึงรบไม่ขลาด
เอกราชจะไม่ให้ใครข่มขี่
สละเลือดทุกหยาดเป็นชาติพลี
เถลิงประเทศชาติไทยทวีมีชัยชโย"""

# Convert text to labels
inputs = [ThaiWordSegmentLabeller.get_input_labels(text)]
lengths = [len(text)]

def nonzero(a):
    return [i for i, e in enumerate(a) if e != 0]

def split(s, indices):
    return [s[i:j] for i,j in zip(indices, indices[1:]+[None])]

with tf.Session() as session:
    # Read model weights
    model = tf.saved_model.loader.load(session, [tf.saved_model.tag_constants.SERVING], saved_model_path)

    # Get model input variables
    graph = tf.get_default_graph()
    g_inputs = graph.get_tensor_by_name('IteratorGetNext:1')
    g_lengths = graph.get_tensor_by_name('IteratorGetNext:0')
    g_training = graph.get_tensor_by_name('Placeholder_1:0')
    g_outputs = graph.get_tensor_by_name('boolean_mask_1/Gather:0')
    
    # Segment the text
    y = session.run(g_outputs, feed_dict = {g_inputs: inputs, g_lengths: lengths, g_training: False})

    # Mark word boundaries with pipe character
    for w in split(text, nonzero(y)): print(w, end='|')
    print()
