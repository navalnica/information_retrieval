import logging
from typing import List
import numpy as np
import os
import csv

from .siamese import CustomBinaryClassificationEvaluator


logger = logging.getLogger(__name__)

class CustomCEBinaryClassificationEvaluator:
    """
    This evaluator can be used with the CrossEncoder class. Given sentence pairs and binary labels (0 and 1),
    it compute the average precision and the best possible f1 score
    """
    def __init__(self, sentence_pairs: List[List[str]], labels: List[int], batch_size: int, write_csv: bool = True):
        assert len(sentence_pairs) == len(labels)
        for label in labels:
            assert (label == 0 or label == 1)

        self.sentence_pairs = sentence_pairs
        self.labels = np.asarray(labels)
        self.batch_size = batch_size

        self.csv_file = "CEBinaryClassificationEvaluator_results.csv"
        self.csv_headers = ["epoch", "steps", "Accuracy", "F1", "F1_Threshold", "Precision", "Recall"]
        self.write_csv = write_csv

    def __call__(self, model, output_path: str = None, epoch: int = -1, steps: int = -1) -> float:
        if epoch != -1:
            if steps == -1:
                out_txt = " after epoch {}:".format(epoch)
            else:
                out_txt = " in epoch {} after {} steps:".format(epoch, steps)
        else:
            out_txt = ":"

        logger.info("CEBinaryClassificationEvaluator: Evaluating the model on " + " dataset" + out_txt)
        pred_scores = model.predict(
            self.sentence_pairs, batch_size=self.batch_size, 
            convert_to_numpy=True, show_progress_bar=True
        )
        output_scores = CustomBinaryClassificationEvaluator.find_best_f1_and_threshold(pred_scores, self.labels, True)

        logger.info("F1:                 {:.2f}\t(Threshold: {:.4f})".format(output_scores['f1'] * 100, output_scores['f1_threshold']))
        logger.info("Accuracy:           {:.2f}".format(output_scores['accuracy'] * 100))
        logger.info("Precision:          {:.2f}".format(output_scores['precision'] * 100))
        logger.info("Recall:             {:.2f}".format(output_scores['recall'] * 100))

        if output_path is not None and self.write_csv:
            csv_path = os.path.join(output_path, self.csv_file)
            output_file_exists = os.path.isfile(csv_path)
            with open(csv_path, mode="a" if output_file_exists else 'w', encoding="utf-8") as f:
                writer = csv.writer(f)
                if not output_file_exists:
                    writer.writerow(self.csv_headers)

                writer.writerow([
                    epoch, steps, 
                    output_scores['accuracy'], 
                    output_scores['f1'], output_scores['f1_threshold'], 
                    output_scores['precision'], 
                    output_scores['recall']
                ])

        return output_scores['recall']
