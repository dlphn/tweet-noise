import tensorflow as tf
import numpy as np
import os
import deepLearning.data_helpersBinary as data_helpers
import json
import csv

# Parameters
# ==================================================

# Data loading params
tf.flags.DEFINE_string("pos_dir", "test_data_actualite.json", "Data source for the positive data.")
tf.flags.DEFINE_string("neg_dir", "test_data_spam.json", "Data source for the negative data.")

# Eval Parameters
"best run : 1555165791 : gru, dropout 0.9, l2 : 3.0, num_epoch : 500 , batch size : 32, learning_rate : 1e-3"
tf.flags.DEFINE_integer("batch_size", 1054, "Batch Size (Default: 64)")
tf.flags.DEFINE_string("checkpoint_dir", "./runs/1555165791/checkpoints", "Checkpoint directory from training run")
tf.flags.DEFINE_string("voc_dir", "./runs/1555165791/text_vocab", "Checkpoint directory from training run")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")


FLAGS = tf.flags.FLAGS
# FLAGS._parse_flags()
# print("\nParameters:")
# for attr, value in sorted(FLAGS.__flags.items()):
#     print("{}={}".format(attr.upper(), value))
# print("")

jsonFileActu = open("test_data_actualite.json", "r")
jsonObjActu = json.load(jsonFileActu)
nbActu = len(jsonObjActu)

jsonFileSpam = open("test_data_spam.json", "r")
jsonObjSpam = json.load(jsonFileSpam)
nbSpam = len(jsonObjSpam)

def eval():
    with tf.device('/cpu:0'):
        x_text, y = data_helpers.load_data_and_labels(FLAGS.pos_dir, FLAGS.neg_dir)

    # Map data into vocabulary
    text_path = os.path.join(FLAGS.voc_dir)
    text_vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor.restore(text_path)

    x_eval = np.array(list(text_vocab_processor.transform(x_text)))
    print(len(x_eval))
    print(x_eval)
    y_eval = np.argmax(y, axis=1)
    print(len(y_eval))
    print(y_eval)

    checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)

    graph = tf.Graph()
    with graph.as_default():
        session_conf = tf.ConfigProto(
            allow_soft_placement=FLAGS.allow_soft_placement,
            log_device_placement=FLAGS.log_device_placement)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            # Load the saved meta graph and restore variables
            saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
            saver.restore(sess, checkpoint_file)

            # Get the placeholders from the graph by name
            input_text = graph.get_operation_by_name("input_text").outputs[0]
            # input_y = graph.get_operation_by_name("input_y").outputs[0]
            dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

            # Tensors we want to evaluate
            predictions = graph.get_operation_by_name("output/predictions").outputs[0]

            # Generate batches for one epoch
            batches = data_helpers.batch_iter(list(x_eval), FLAGS.batch_size, 1, shuffle=False)

            # Collect the predictions here
            all_predictions = []
            for x_batch in batches:
                batch_predictions = sess.run(predictions, {input_text: x_batch,
                                                           dropout_keep_prob: 1.0})
                all_predictions = np.concatenate([all_predictions, batch_predictions])
            correct_predictions = float(sum(all_predictions == y_eval))
            true_positive = float(sum(all_predictions[: nbActu] == y_eval[ : nbActu]))
            precision = true_positive / sum(all_predictions)
            recall = true_positive / float(len(y_eval[: nbActu]))
            fscore = 2 * precision * recall / (precision + recall)
            print("Total number of test examples: {}".format(len(y_eval)))
            print("Accuracy: {:g}".format(correct_predictions / float(len(y_eval))))
            print("Precision: {:g}".format(precision))
            print("Recall: {:g}".format(recall))
            print("F Score: {:g}".format(fscore))

            # Save the evaluation to a csv
            predictions_human_readable = np.column_stack((np.array(x_text), all_predictions))
            out_path = os.path.join(FLAGS.checkpoint_dir, "prediction.csv")
            print("Saving evaluation to {0}".format(out_path))

            with open(out_path, 'w') as f:
                csv.writer(f).writerows(predictions_human_readable)

def main(_):
    eval()


if __name__ == "__main__":
    tf.app.run()