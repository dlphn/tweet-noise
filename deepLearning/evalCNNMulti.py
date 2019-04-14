import os
import sys
import json
import logging
import deepLearning.data_helpersMulti as data_helpers
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.contrib import learn
import csv

logging.getLogger().setLevel(logging.INFO)

# Parameters
# ==================================================

# Data Parameters
tf.flags.DEFINE_string("test_data_file", "test_data_multi.json", "Data source for the test data.")

# Eval Parameters
"best run : 1555227545 : dropout 0.75, l2 : 0.0, batch size : 16"
tf.flags.DEFINE_integer("batch_size", 1054,"batch Size (default: 64)")
tf.flags.DEFINE_string("checkpoint_dir", "./runs/1555227545/checkpoints", "Checkpoint directory from training run")
tf.flags.DEFINE_string("voc_dir", "./runs/1555227545/text_vocab", "Checkpoint directory from training run")
tf.flags.DEFINE_boolean("eval_train", False, "Evaluate on all training data")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS

def predict_unseen_data():
    """Step 0: load trained model and parameters"""
    checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
    logging.critical('Loaded the trained model: {}'.format(checkpoint_file))

    """Step 1: load data for prediction"""
    test_examples = json.loads(open("test_data_multi.json").read())

    # labels.json was saved during training, and it has to be loaded during prediction
    labels = json.loads(open('./labels.json').read())
    one_hot = np.zeros((len(labels), len(labels)), int)
    np.fill_diagonal(one_hot, 1)
    label_dict = dict(zip(labels, one_hot))

    x_raw = [example['text'] for example in test_examples]
    x_test = [data_helpers.clean_str(x) for x in x_raw]
    logging.info('The number of x_test: {}'.format(len(x_test)))

    y_test = None
    if 'class' in test_examples[0]:
        y_raw = [example['class'] for example in test_examples]
        y_test = [label_dict[y] for y in y_raw]
        logging.info('The number of y_test: {}'.format(len(y_test)))

    vocab_path = os.path.join(FLAGS.voc_dir)
    vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)
    x_test = np.array(list(vocab_processor.transform(x_test)))

    """Step 2: compute the predictions"""
    graph = tf.Graph()
    with graph.as_default():
        session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        sess = tf.Session(config=session_conf)

        with sess.as_default():
            saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
            saver.restore(sess, checkpoint_file)

            input_x = graph.get_operation_by_name("input_x").outputs[0]
            dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]
            predictions = graph.get_operation_by_name("output/predictions").outputs[0]

            batches = data_helpers.batch_iter(list(x_test), FLAGS.batch_size, 1, shuffle=False)
            all_predictions = []
            for x_test_batch in batches:
                batch_predictions = sess.run(predictions, {input_x: x_test_batch, dropout_keep_prob: 1.0})
                all_predictions = np.concatenate([all_predictions, batch_predictions])

    if y_test is not None:
        print(y_test)
        y_test = np.argmax(y_test, axis=1)
        correct_predictions = sum(all_predictions == y_test)

        # Save the actual labels back to file
        actual_labels = [labels[int(prediction)] for prediction in all_predictions]

        for idx, example in enumerate(test_examples):
            example['new_prediction'] = actual_labels[idx]

        true_prediction = [0, 0, 0, 0, 0, 0]
        total_prediction = [0, 0, 0, 0, 0, 0]
        true_results = [0, 0, 0, 0, 0, 0]
        for i in range (len(all_predictions)) :
            true_results[int(all_predictions[i])] += 1
            total_prediction[y_test[i]] += 1
            if all_predictions[i] == y_test[i] :
                true_prediction[y_test[i]] += 1
        print(true_prediction)
        print(total_prediction)
        print(true_results)
        print("Total number of test examples: {}".format(len(y_test)))
        print("Accuracy: {:g}".format(correct_predictions / float(len(y_test))))
        precision = (true_prediction[0] + true_prediction[5]) / (total_prediction[0] + total_prediction[5])
        recall = (true_prediction[0] + true_prediction[5]) / (true_results[0] + true_results[5])
        fscore = 2 * precision * recall / (precision + recall)
        print("Precision: {:g}".format(precision))
        print("Recall: {:g}".format(recall))
        print("F Score: {:g}".format(fscore))


        # Save the evaluation to a csv
        predictions_human_readable = np.column_stack((np.array(x_raw), all_predictions))
        out_path = os.path.join(FLAGS.checkpoint_dir, "prediction.csv")
        print("Saving evaluation to {0}".format(out_path))

        with open(out_path, 'w') as f:
            csv.writer(f).writerows(predictions_human_readable)

if __name__ == '__main__':
#python3 predict.py ./trained_model_1478649295/ ./data/small_samples.json
    predict_unseen_data()