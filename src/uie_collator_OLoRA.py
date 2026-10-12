import logging

from uie_collator import DataCollatorForUIE as _SharedDataCollatorForUIE

import torch
from transformers.data.data_collator import *


logger = logging.getLogger(__name__)

SUPPORTED_DECODER_MODELS = ['codegen', 'bloomz', 'gpt-neox', 'llama']
SUPPORTED_SEQ2SEQ_MODELS = ['t5', 'flan-t5']


def check_model(model_name, supported_models):
    for sup_model in supported_models:
        if sup_model.lower() in model_name.lower():
            return True

    return False


@dataclass
class DataCollatorForUIE:
    decoder_pipeline_version = _SharedDataCollatorForUIE.decoder_pipeline_version
    tokenizer: PreTrainedTokenizerBase
    model: Optional[Any] = None
    padding: Union[bool, str, PaddingStrategy] = True
    max_source_length: Optional[int] = None
    max_target_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"
    add_task_name: bool = False
    add_dataset_name: bool = False
    common_dataset_name: str = None
    text_only: bool = False
    num_examples: int = 0
    input_record_file: str = None

    def __call__(self, batch, return_tensors=None):
        if return_tensors is None:
            return_tensors = self.return_tensors

        model_name = self.model.config._name_or_path
        # print(model_name)
        if check_model(model_name, SUPPORTED_DECODER_MODELS):
            model_inputs = self.decoder_call(batch, return_tensors)
        elif check_model(model_name, SUPPORTED_SEQ2SEQ_MODELS):
            model_inputs = self.seq2seq_call(batch, return_tensors)
        else:
            raise ValueError('Unsupport model {}!'.format(model_name))

        return model_inputs

    def _tokenize_labels(self, labels, return_tensors):
        label_kwargs = dict(
            max_length=self.max_target_length,
            padding=self.padding,
            return_tensors=return_tensors,
            truncation=True,
            pad_to_multiple_of=self.pad_to_multiple_of,
        )

        try:
            return self.tokenizer(text_target=labels, **label_kwargs)
        except TypeError:
            with self.tokenizer.as_target_tokenizer():
                return self.tokenizer(labels, **label_kwargs)

    def get_instruction(self, instance):
        # "instructions \n options \n {0} \n Answer: "
        instruction = instance['Instance']["instruction"]
        content = instance['Instance']['sentence']

        # add task/ds prefix
        prefix = ''
        if self.add_task_name:
            prefix += "Task:" + instance['Task'] + '\n'
        if self.add_dataset_name:
            ds_name = self.common_dataset_name if self.common_dataset_name else instance['Dataset']
            prefix = prefix + "Dataset:"
            prefix = prefix + ds_name + '\n' if prefix else instance['Dataset'] + '\n'
        if prefix:
            instruction = prefix + instruction

        # TODO, support few shot
        # add few shot samples
        samples = ''
        if len(instance['Samples']) > 0:
            raise Exception('Few shot is coming soon...')
        if samples:
            content = samples + content
        # TODO, fix bug
        try:
            instruction = instruction.format(content)
        finally:
            return instruction


    def seq2seq_call(self, batch, return_tensors):
        sources = []
        labels = []

        for instance in batch:
            label = instance['Instance']['label']
            labels.append(label)
            instruction = self.get_instruction(instance)

            source = instruction
            tokenized_source = self.tokenizer(source)["input_ids"]
            if len(tokenized_source) <= self.max_source_length:
                sources.append(source)
            else:
                sources.append(self.tokenizer.decode(tokenized_source[:self.max_source_length], skip_special_tokens=True))

        # TODO, support online demo
        if self.text_only:
            model_inputs = {"inputs": sources, "labels": labels}
        else:
            model_inputs = self.tokenizer(
                sources,
                max_length=self.max_source_length,
                padding=self.padding,
                return_tensors=return_tensors,
                truncation=True,
                pad_to_multiple_of=self.pad_to_multiple_of
            )
            labels = self._tokenize_labels(labels, return_tensors)
            label_mask = labels["attention_mask"].bool()
            model_inputs["labels"] = labels["input_ids"].masked_fill(~label_mask, self.label_pad_token_id)

            # prepare decoder_input_ids
            if self.model is not None:
                decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(labels=model_inputs["labels"])
                model_inputs["decoder_input_ids"] = decoder_input_ids

            self._save_samples(model_inputs, sources, labels)

        return model_inputs

    # Keep the legacy import path consistent with the main decoder implementation.
    # Seq2seq tokenization and the legacy constructor above remain unchanged.
    decoder_call = _SharedDataCollatorForUIE.decoder_call
    _save_samples = _SharedDataCollatorForUIE._save_samples
