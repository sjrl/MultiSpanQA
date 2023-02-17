import os
import json
import logging
import collections
from tqdm.auto import tqdm
from typing import Optional, Tuple, Dict, List

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

import numpy as np

from transformers import (
    BertPreTrainedModel,
    BertModel,
)

from eval_script import get_entities

logger = logging.getLogger(__name__)


# TODO Replace with a BertForTokenClassifcation
#  https://huggingface.co/docs/transformers/model_doc/bert#transformers.BertForTokenClassification
class TaggerForMultiSpanQA(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        labels=None,
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        outputs = (logits,) + outputs[:]
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            active_loss = attention_mask.view(-1) == 1
            active_logits = logits.view(-1, self.num_labels)
            active_labels = torch.where(
                active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
            )
            loss = loss_fct(active_logits, active_labels)
            outputs = (loss,) + outputs

        return outputs


def postprocess_tagger_predictions(
    examples: Dict,
    features: Dict,
    predictions: Tuple[np.ndarray, np.ndarray],
    id2label: Dict[int, str],
    output_dir: Optional[str] = None,
    prefix: Optional[str] = None,
    log_level: Optional[int] = logging.WARNING,
    save_embeds: bool = False,
) -> str:
    """
    Post-processes the predictions of a question-answering model to convert them to answers that are substrings of the
    original contexts. This is the base postprocessing functions for models that only return start and end logits.

    :param examples: The non-preprocessed dataset (see the main script for more information).
    :param features: The processed dataset (see the main script for more information).
    :param predictions: The predictions of the model: two arrays containing the start logits and the end logits
        respectively. Its first dimension must match the number of elements of :obj:`features`.
    :param id2label: Dictionary lookup to convert id to label.
    :param output_dir: If provided, the dictionaries of predictions, n_best predictions (with their scores and logits)
        and, if :obj:`version_2_with_negative=True`, the dictionary of the scores differences between best and null
        answers, are saved in `output_dir`.
    :param prefix: If provided, the dictionaries mentioned above are saved with `prefix` added to their names.
    :param log_level: ``logging`` log level (the default is ``logging.WARNING``)
    :param save_embeds: If True save the logits and hidden states to numpy files.
    """

    if len(predictions[0].shape) != 1:  # Not CRF output
        if predictions[0].shape[-1] != 3:
            raise RuntimeError(f"`predictions` should be in shape of (max_seq_length, 3).")
        all_logits = predictions[0]
        all_hidden = predictions[1]
        all_labels = np.argmax(predictions[0], axis=2)

        if len(predictions[0]) != len(features):
            raise ValueError(f"Got {len(predictions[0])} predictions and {len(features)} features.")
    else:
        all_logits = predictions

    if -100 not in id2label.values():
        id2label[-100] = 'O'

    # Build a map example to its corresponding features.
    example_id_to_index = {k: i for i, k in enumerate(examples["id"])}
    features_per_example = collections.defaultdict(list)
    for i, feature in enumerate(features):
        features_per_example[example_id_to_index[feature["example_id"]]].append(i)

    # The dictionaries we have to fill.
    all_predictions: Dict[str, List[str]] = collections.OrderedDict()
    all_ids = []
    all_valid_logits = []
    all_valid_hidden = []

    # Logging.
    logger.setLevel(log_level)
    logger.info(f"Post-processing {len(examples)} example predictions split into {len(features)} features.")

    # Let's loop over all the examples!
    for example_index, example in enumerate(tqdm(examples)):
        # Those are the indices of the features associated to the current example.
        feature_indices = features_per_example[example_index]
        prelim_predictions = []

        # Looping through all the features associated to the current example.
        for feature_index in feature_indices:
            # We grab the predictions of the model for this feature.
            sequence_ids = features[feature_index]['sequence_ids']
            word_ids = features[feature_index]['word_ids']
            logits = [l for l in all_logits[feature_index]]
            hidden = [l for l in all_hidden[feature_index]]
            labels = [id2label[l] for l in all_labels[feature_index]]
            prelim_predictions.append(
                {
                    "logits": logits,
                    "hidden": hidden,
                    "labels": labels,
                    "word_ids": word_ids,
                    "sequence_ids": sequence_ids
                }
            )

        previous_word_idx = -1
        ignored_index = []  # Some example tokens will disappear after tokenization.
        valid_labels = []
        valid_logits = []
        valid_hidden = []
        for x in prelim_predictions:
            logits = x['logits']
            hidden = x['hidden']
            labels = x['labels']
            word_ids = x['word_ids']
            sequence_ids = x['sequence_ids']

            # Start token index of the current span in the text.
            token_start_index = 0
            while sequence_ids[token_start_index] != 1:
                token_start_index += 1

            for word_idx, label, lo, hi in list(zip(word_ids, labels, logits, hidden))[token_start_index:]:
                # Special tokens have a word id that is None. We set the label to -100 so they are automatically
                # ignored in the loss function.
                if word_idx is None:
                    continue
                # We set the label for the first token of each word.
                elif word_idx > previous_word_idx:
                    ignored_index += range(previous_word_idx + 1, word_idx)
                    valid_labels.append(label)
                    valid_logits.append(lo)
                    valid_hidden.append(hi)
                    previous_word_idx = word_idx
                # For the other tokens in a word, we set the label to either the current label or -100, depending on
                # the label_all_tokens flag.
                else:
                    continue

        context = example["context"]
        for i in ignored_index[::-1]:
            context = context[:i] + context[i + 1:]
        assert len(context) == len(valid_labels)

        predict_entities = get_entities(valid_labels, context)
        predictions = [x[0] for x in predict_entities]
        all_predictions[example["id"]] = predictions

        all_ids.append(example["id"])
        all_valid_logits.append(valid_logits)
        all_valid_hidden.append(valid_hidden)

    all_valid_logits = np.array(all_valid_logits)
    all_valid_hidden = np.array(all_valid_hidden)

    # If we have an output_dir, let's save all those dicts.
    if output_dir is not None:
        if not os.path.isdir(output_dir):
            raise EnvironmentError(f"{output_dir} is not a directory.")

        prediction_file = os.path.join(
            output_dir, "predictions.json" if prefix is None else f"{prefix}_predictions.json"
        )

        logger.info(f"Saving predictions to {prediction_file}.")
        with open(prediction_file, "w") as writer:
            writer.write(json.dumps(all_predictions, indent=4) + "\n")

    if save_embeds:
        logger.info(f"Saving embeds for CRF.")
        ids_file = os.path.join(output_dir, "ids.json" if prefix is None else f"{prefix}_ids.json")
        with open(ids_file, "w") as writer:
            writer.write(json.dumps(all_ids, indent=4) + "\n")

        logits_file = os.path.join(output_dir, "logits.np" if prefix is None else f"{prefix}_logits.np")
        hidden_file = os.path.join(output_dir, "hidden.np" if prefix is None else f"{prefix}_hidden.np")

        np.save(logits_file, all_valid_logits)
        np.save(hidden_file, all_valid_hidden)

    return prediction_file
