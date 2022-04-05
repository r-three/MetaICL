# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import os
import torch

from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM


class MetaICLModel(object):
    def __init__(self, model_name, amp=True, n_gpu=0):
        # TODO: maybe add amp
        self.amp = amp
        self.n_gpu = n_gpu
        self.model = None
        self.mode = None
        self.model_name = model_name

    def is_none(self):
        return self.model is None

    def train(self):
        self.model.train()
        self.mode = "train"

    def eval(self):
        self.model.eval()
        self.mode = "eval"

    def cuda(self):
        if self.n_gpu > 1:
            encoder = self.model.encoder
            decoder = self.model.decoder
            encoder.to("cuda:0")
            decoder.to("cuda:1")
            # TODO: implement two gpu
        self.model.cuda()

    def load(self):
        """
        checkpoint can be either keyword of the model or path to the checkpoint file
        """
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
        self.mode = "eval"

    def do_inference(self, data, batch_size=1, verbose=False):
        dataloader = data.get_dataloader(batch_size, is_training=False)
        if verbose:
            dataloader = tqdm(dataloader)
        losses = []
        for batch in dataloader:
            input_ids = batch[0].cuda()
            attention_mask = batch[1].cuda()
            token_type_ids = batch[2].cuda()
            if len(batch) == 3:
                labels = None
            else:
                labels = batch[3].cuda()
            with torch.no_grad():
                loss = self.run_model(input_ids, attention_mask, token_type_ids, labels=labels)
            losses += loss.cpu().detach().numpy().tolist()
        return losses

    def do_predict(self, data, batch_size=1, losses=None, verbose=False):
        if losses is None:
            losses = self.do_inference(data, batch_size, verbose=verbose)
        losses = np.array(losses)
        assert len(losses) == len(data)
        predictions = []
        for idx, dp in enumerate(data.metadata):
            curr_label_losses = [np.sum(losses[indices]) for indices in dp["indices"]]
            prediction_idx = sorted(enumerate(curr_label_losses), key=lambda x: x[1])[0][0]
            prediction = dp["options"][prediction_idx]
            predictions.append(prediction.strip())
        return predictions

    def run_model(self, input_ids, attention_mask, token_type_ids, labels=None):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits[..., :-1, :].contiguous()

        if labels is None:
            labels = input_ids
        labels = labels[..., 1:].contiguous()
        label_mask = token_type_ids[..., 1:].contiguous()

        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
        losses = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))  # [batch_size, length]

        losses = losses.view(logits.size(0), logits.size(1)) * label_mask
        return torch.sum(losses, axis=1) / torch.sum(label_mask, axis=1)
