import logging

import torch
from transformers.data.data_collator import *
import os

logger = logging.getLogger(__name__)
# logging.basicConfig(level=logging.DEBUG)

SUPPORTED_DECODER_MODELS = ['codegen', 'bloomz', 'gpt-neox', 'llama']
SUPPORTED_SEQ2SEQ_MODELS = ['t5', 'flan-t5']


def check_model(model_name, supported_models):
    for sup_model in supported_models:
        if sup_model.lower() in model_name.lower():
            return True

    return False


@dataclass
class DataCollatorForUIE:
    # Audit metadata only; never included in model inputs.
    decoder_pipeline_version = "decoder_token_ids_v2"
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
    use_probe: bool = False

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        model: Optional[Any] = None,
        padding: Union[bool, str, PaddingStrategy] = True,
        max_source_length: Optional[int] = None,
        max_target_length: Optional[int] = None,
        pad_to_multiple_of: Optional[int] = None,
        label_pad_token_id: int = -100,
        return_tensors: str = "pt",
        add_task_name: bool = False,
        add_dataset_name: bool = False,
        common_dataset_name: Optional[str] = None,
        text_only: bool = False,
        num_examples: int = 0,
        input_record_file: Optional[str] = None,
        use_probe: bool = False
    ):
        self.tokenizer = tokenizer
        self.model = model
        self.padding = padding
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.pad_to_multiple_of = pad_to_multiple_of
        self.label_pad_token_id = label_pad_token_id
        self.return_tensors = return_tensors
        self.add_task_name = add_task_name
        self.add_dataset_name = add_dataset_name
        self.common_dataset_name = common_dataset_name
        self.text_only = text_only
        self.num_examples = num_examples
        self.input_record_file = input_record_file
        self.use_probe = use_probe


    # def __post_init__(self):
    #     logger.info("Initialized DataCollatorForUIE with the following parameters:")
    #     logger.info(f"  tokenizer: {self.tokenizer.__class__.__name__}")
    #     logger.info(f"  model: {self.model.__class__.__name__ if self.model else 'None'}")
    #     logger.info(f"  padding: {self.padding}")
    #     logger.info(f"  max_source_length: {self.max_source_length}")
    #     logger.info(f"  max_target_length: {self.max_target_length}")
    #     logger.info(f"  pad_to_multiple_of: {self.pad_to_multiple_of}")
    #     logger.info(f"  label_pad_token_id: {self.label_pad_token_id}")
    #     logger.info(f"  return_tensors: {self.return_tensors}")
    #     logger.info(f"  add_task_name: {self.add_task_name}")
    #     logger.info(f"  add_dataset_name: {self.add_dataset_name}")
    #     logger.info(f"  common_dataset_name: {self.common_dataset_name}")
    #     logger.info(f"  text_only: {self.text_only}")
    #     logger.info(f"  num_examples: {self.num_examples}")
    #     logger.info(f"  input_record_file: {self.input_record_file}")

    def __call__(self, batch, return_tensors=None):
        if return_tensors is None:
            return_tensors = self.return_tensors

        model_name = self.model.config._name_or_path
        # print(model_name)
        # logger.info("Model name: {}".format(model_name))
        if "/" in model_name or "\\" in model_name or "finetune" in model_name:
            # 优先用 model_type 字段（如果有）
            model_type = getattr(self.model.config, "model_type", None)
            if model_type is not None:
                model_name = model_type
            else:
                # 否则用 basename 兜底
                model_name = os.path.basename(model_name)




        # 根据使用的模型不同进行判断
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

            # --- ✨新增 --- 把 class_labels 加入 batch
            if self.use_probe:
                model_inputs["class_labels"] = torch.tensor(
                [instance["class_label"] for instance in batch], dtype=torch.long
            )
                
            # prepare decoder_input_ids
            if self.model is not None:
                decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(labels=model_inputs["labels"])
                model_inputs["decoder_input_ids"] = decoder_input_ids

            self._save_samples(model_inputs, sources, labels)

        return model_inputs

    def decoder_call(self, batch, return_tensors):
        """Build causal examples in token space, without a prompt/answer BPE merge.

        Prompt and answer are encoded independently with automatic special tokens
        disabled. The prompt has one explicit BOS; the answer has one explicit
        EOS. Source and target budgets are independent, so long prompts cannot
        remove target supervision. Source truncation keeps BOS and the prompt
        suffix (including the answer cue), identically for training and inference.

        Our Llama loss applies loss_mask to logits[:, :-1], rather than labels
        [:, 1:]. Therefore a target at index j needs loss_mask[j - 1] = 1.
        """
        if not batch:
            raise ValueError("A decoder batch must contain at least one example")
        training = [instance['subset'] not in ['dev', 'test'] for instance in batch]
        if any(training) and not all(training):
            raise ValueError("Cannot mix training and inference examples in a decoder batch")
        for name in ("max_source_length", "max_target_length"):
            value = getattr(self, name)
            if value is not None and value < 1:
                raise ValueError(name + " must be positive or None")

        self.tokenizer.padding_side = 'left'
        bos_id = self.tokenizer.bos_token_id
        eos_id = self.tokenizer.eos_token_id
        sources, labels, sequences, prompt_lengths, target_lengths = [], [], [], [], []
        for instance, is_train in zip(batch, training):
            label = instance['Instance']['label']
            instruction = self.get_instruction(instance)
            prompt = self.tokenizer(instruction, add_special_tokens=False)["input_ids"]
            prompt = ([bos_id] if bos_id is not None else []) + prompt
            if self.max_source_length is not None and len(prompt) > self.max_source_length:
                if bos_id is None:
                    prompt = prompt[-self.max_source_length:]
                else:
                    suffix_budget = self.max_source_length - 1
                    prompt = [bos_id] + (prompt[-suffix_budget:] if suffix_budget else [])
            if not prompt:
                raise ValueError("A decoder prompt needs a token to predict its first answer")

            target = []
            if is_train:
                target = self.tokenizer(label, add_special_tokens=False)["input_ids"]
                # Reserve EOS inside the target budget; never insert BOS in the label.
                if eos_id is not None and target and target[-1] == eos_id:
                    target = target[:-1]
                if self.max_target_length is not None:
                    content_budget = self.max_target_length - int(eos_id is not None)
                    if target and content_budget < 1:
                        raise ValueError("max_target_length must retain an answer token as well as EOS")
                    target = target[:content_budget]
                if eos_id is not None:
                    target.append(eos_id)
                if not target:
                    raise ValueError("A decoder training target needs at least one token")

            sequence = prompt + target
            sequences.append(sequence)
            prompt_lengths.append(len(prompt))
            target_lengths.append(len(target))
            # Decoding is for display only; these strings are never re-tokenized.
            sources.append(self.tokenizer.decode(sequence, skip_special_tokens=False,
                                                  clean_up_tokenization_spaces=False))
            labels.append(label)

        if self.text_only:
            return {"inputs": sources, "labels": labels}

        limit_input_len = self.max_source_length
        if all(training) and limit_input_len is not None:
            limit_input_len = (limit_input_len + self.max_target_length
                               if self.max_target_length is not None else None)
        model_inputs = self.tokenizer.pad(
            {"input_ids": sequences},
            max_length=(limit_input_len if self.padding in ("max_length", PaddingStrategy.MAX_LENGTH) else None),
            padding=self.padding,
            return_tensors=return_tensors,
            pad_to_multiple_of=self.pad_to_multiple_of,
        )
        input_ids = model_inputs['input_ids']
        model_inputs['labels'] = torch.full_like(input_ids, self.label_pad_token_id)
        loss_mask = torch.zeros_like(input_ids, dtype=torch.float32)
        width = input_ids.shape[1]
        for row, (sequence, prompt_length, target_length) in enumerate(
                zip(sequences, prompt_lengths, target_lengths)):
            if not target_length:
                continue
            answer_start = width - len(sequence) + prompt_length
            answer_end = answer_start + target_length
            model_inputs['labels'][row, answer_start:answer_end] = input_ids[row, answer_start:answer_end]
            loss_mask[row, answer_start - 1:answer_end - 1] = 1
        model_inputs['loss_mask'] = loss_mask

        if getattr(self, "use_probe", False):
            model_inputs['class_labels'] = torch.tensor(
                [instance["class_label"] for instance in batch], dtype=torch.long)
        self._save_samples(model_inputs, sources, labels)
        return model_inputs

    def _save_samples(self, model_inputs, sources, labels):
        if not self.input_record_file:
            return

        if "loss_mask" in model_inputs:
            with open(self.input_record_file, 'a+', encoding='utf-8') as f:
                for text, label, token_labels, predictor_mask in zip(
                        sources, labels, model_inputs['labels'], model_inputs['loss_mask']):
                    target_mask = torch.zeros_like(predictor_mask, dtype=torch.bool)
                    target_mask[1:] = predictor_mask[:-1].bool()
                    mask_label = self.tokenizer.decode(
                        token_labels[target_mask].tolist(), clean_up_tokenization_spaces=False)
                    f.write(text + '\n')
                    f.write(label + '\n')
                    f.write(mask_label + '\n\n')
        else:
            with open(self.input_record_file, 'a+', encoding='utf-8') as f:
                for text, label in zip(sources, labels['input_ids']):
                    f.write(text + '\n')
                    f.write(self.tokenizer.decode(label, clean_up_tokenization_spaces=False) + '\n')
