import tensorflow as tf
import numpy as np
import os
import deepLearning.data_helpersBinary as data_helpers
from tensorflow.contrib import learn
import csv
import json

# Parameters
# ==================================================

# Directory of the run : MODIFY AFTER EACH MODEL TRAINING
tf.flags.DEFINE_string("model_dir", "./runs/1555164573/", "Model directory from training run")

# Data loading params
tf.flags.DEFINE_string("actu_dir", "test_data_actualite.json", "Data source for the actuality tweets.")
tf.flags.DEFINE_string("spam_dir", "test_data_spam.json", "Data source for the spam tweets.")

# Eval Parameters
"best run : 1555164573 : dropout 0.9, l2 : 0.0, num_epoch : 500 , batch size : 32"
tf.flags.DEFINE_integer("batch_size", 1054,"batch Size (default: 64)")
tf.flags.DEFINE_boolean("eval_train", False, "Evaluate on all training data")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS

def eval_cnn_binary():
    jsonFileActu = open(FLAGS.actu_dir, "r")
    jsonObjActu = json.load(jsonFileActu)
    nbActu = len(jsonObjActu)

    x_raw, y_test = data_helpers.load_data_and_labels(FLAGS.actu_dir, FLAGS.spam_dir)
    y_test = np.argmax(y_test, axis=1)

    # Map data into vocabulary
    vocab_path = os.path.join(FLAGS.model_dir + "text_vocab")
    vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)
    x_test = np.array(list(vocab_processor.transform(x_raw)))

    checkpoint_file = tf.train.latest_checkpoint(FLAGS.model_dir + "checkpoints")
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
            input_x = graph.get_operation_by_name("input_x").outputs[0]
            # input_y = graph.get_operation_by_name("input_y").outputs[0]
            dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

            # Tensors we want to evaluate
            predictions = graph.get_operation_by_name("output/predictions").outputs[0]

            # Generate batches for one epoch
            batches = data_helpers.batch_iter(list(x_test), FLAGS.batch_size, 1, shuffle=False)

            # Collect the predictions here
            all_predictions = []

            for x_test_batch in batches:
                batch_predictions = sess.run(predictions, {input_x: x_test_batch, dropout_keep_prob: 1.0})
                all_predictions = np.concatenate([all_predictions, batch_predictions])

    if y_test is not None:
        correct_predictions = float(sum(all_predictions == y_test))
        true_positive = float(sum(all_predictions[: nbActu] == y_test[: nbActu]))
        precision = true_positive / sum(all_predictions)
        recall = true_positive / float(len(y_test[: nbActu]))
        fscore = 2 * precision * recall / (precision + recall)
        print("Total number of test examples: {}".format(len(y_test)))
        print("Accuracy: {:g}".format(correct_predictions / float(len(y_test))))
        print("Precision: {:g}".format(precision))
        print("Recall: {:g}".format(recall))
        print("F Score: {:g}".format(fscore))

        # Save the evaluation to a csv
        predictions_human_readable = np.column_stack((np.array(x_raw), all_predictions))
        out_path = os.path.join(FLAGS.model_dir + "checkpoints", "prediction.csv")
        print("Saving evaluation to {0}".format(out_path))

        with open(out_path, 'w') as f:
            csv.writer(f).writerows(predictions_human_readable)

if __name__ == '__main__':
    eval_cnn_binary()
