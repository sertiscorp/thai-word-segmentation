from thainlplib import ThaiWordSegmentLabeller
import tensorflow as tf

saved_model_path='saved_model'

text = """ประเทศไทยรวมเลือดเนื้อชาติเชื้อไทย
เป็นประชารัฐไผทของไทยทุกส่วน
อยู่ดำรงคงไว้ได้ทั้งมวล
ด้วยไทยล้วนหมายรักสามัคคี
ไทยนี้รักสงบแต่ถึงรบไม่ขลาด
เอกราชจะไม่ให้ใครข่มขี่
สละเลือดทุกหยาดเป็นชาติพลี
เถลิงประเทศชาติไทยทวีมีชัยชโย"""
inputs = [ThaiWordSegmentLabeller.get_input_labels(text)]
lengths = [len(text)]

def nonzero(a):
    return [i for i, e in enumerate(a) if e != 0]

def split(s, indices):
    return [s[i:j] for i,j in zip(indices, indices[1:]+[None])]

with tf.Session() as session:
    model = tf.saved_model.loader.load(session, [tf.saved_model.tag_constants.SERVING], saved_model_path)
    signature = model.signature_def[tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
    graph = tf.get_default_graph()

    g_inputs = graph.get_tensor_by_name(signature.inputs['inputs'].name)
    g_lengths = graph.get_tensor_by_name(signature.inputs['lengths'].name)
    g_training = graph.get_tensor_by_name(signature.inputs['training'].name)
    g_outputs = graph.get_tensor_by_name(signature.outputs['outputs'].name)
    y = session.run(g_outputs, feed_dict = {g_inputs: inputs, g_lengths: lengths, g_training: False})

    for w in split(text, nonzero(y)): print(w, end='|')
    print()
