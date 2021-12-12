import logging
import os
import csv
from sklearn.metrics.pairwise import paired_cosine_distances
from sklearn.metrics import average_precision_score
import numpy as np
from typing import List

logger = logging.getLogger(__name__)

class CustomBinaryClassificationEvaluator():

    def __init__(
        self, 
        sentences1: List[str], 
        sentences2: List[str], 
        labels: List[int],
        name: str = 'validation_evaluator', 
        batch_size: int = 32, 
        show_progress_bar: bool = True,
        write_csv: bool = True,
    ):
        self.sentences1 = sentences1
        self.sentences2 = sentences2
        self.labels = labels

        assert len(self.sentences1) == len(self.sentences2)
        assert len(self.sentences1) == len(self.labels)
        for label in labels:
            assert (label == 0 or label == 1)

        self.write_csv = write_csv
        self.name = name
        self.batch_size = batch_size
        assert show_progress_bar is not None
        self.show_progress_bar = show_progress_bar

        self.csv_filename = f"{name}_results.csv"
        self.csv_headers = ["epoch", "steps", "f1_threshold", "f1", "precision", "recall", "accuracy", "ap"]

    def __call__(self, model, output_path: str = None, epoch: int = -1, steps: int = -1) -> float:

        if epoch != -1:
            if steps == -1:
                out_txt = f" after epoch {epoch}:"
            else:
                out_txt = f" in epoch {epoch} after {steps} steps:"
        else:
            out_txt = ":"

        logger.info(f"evaluator: '{self.name}' {out_txt}")

        scores = self.compute_metrices(model)

        # append record to .csv log file

        file_output_data = [epoch, steps]
        for metric in self.csv_headers[2:]:
            file_output_data.append(scores[metric])

        if output_path is not None and self.write_csv:
            os.makedirs(output_path, exist_ok=True)
            csv_fp = os.path.join(output_path, self.csv_filename)
            if not os.path.isfile(csv_fp):
                with open(csv_fp, newline='', mode="w", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(self.csv_headers)
                    writer.writerow(file_output_data)
            else:
                with open(csv_fp, newline='', mode="a", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(file_output_data)

        return scores['f1']


    def compute_metrices(self, model):
        sentences = list(set(self.sentences1 + self.sentences2))
        embeddings = model.encode(sentences, batch_size=self.batch_size, show_progress_bar=self.show_progress_bar, convert_to_numpy=True)
        emb_dict = {sent: emb for sent, emb in zip(sentences, embeddings)}
        embeddings1 = [emb_dict[sent] for sent in self.sentences1]
        embeddings2 = [emb_dict[sent] for sent in self.sentences2]

        cosine_scores = 1 - paired_cosine_distances(embeddings1, embeddings2)  # 1d array
        labels = np.asarray(self.labels)
        output_scores = {}

        reverse = True
        output_scores= self.find_best_f1_and_threshold(cosine_scores, labels, reverse)
        output_scores['ap'] = average_precision_score(labels, cosine_scores * (1 if reverse else -1))

        logger.info("F1:                 {:.2f}\t(Threshold: {:.4f})".format(output_scores['f1'] * 100, output_scores['f1_threshold']))
        logger.info("Accuracy:           {:.2f}".format(output_scores['accuracy'] * 100))
        logger.info("Precision:          {:.2f}".format(output_scores['precision'] * 100))
        logger.info("Recall:             {:.2f}".format(output_scores['recall'] * 100))
        logger.info("Average Precision:  {:.2f}\n".format(output_scores['ap'] * 100))

        return output_scores

    @staticmethod
    def find_best_f1_and_threshold(scores, labels, high_score_more_similar: bool):
        assert len(scores) == len(labels)

        scores = np.asarray(scores)
        labels = np.asarray(labels)

        rows = list(zip(scores, labels))

        rows = sorted(rows, key=lambda x: x[0], reverse=high_score_more_similar)

        best_f1 = best_precision = best_recall = 0
        threshold = 0
        nextract = 0
        ncorrect = 0
        total_num_duplicates = sum(labels)

        for i in range(len(rows)-1):
            score, label = rows[i]
            nextract += 1

            if label == 1:
                ncorrect += 1

            if ncorrect > 0:
                precision = ncorrect / nextract
                recall = ncorrect / total_num_duplicates
                f1 = 2 * precision * recall / (precision + recall)
                if f1 > best_f1:
                    best_f1 = f1
                    best_precision = precision
                    best_recall = recall
                    threshold = (rows[i][0] + rows[i + 1][0]) / 2

        accuracy = (np.where(scores > threshold, 1, 0) == labels).sum() / labels.shape[0]

        return {'f1': best_f1, 'precision': best_precision, 'recall': best_recall, 'accuracy': accuracy, 'f1_threshold': threshold}
