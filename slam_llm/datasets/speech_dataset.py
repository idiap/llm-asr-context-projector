# SPDX-FileCopyrightText: Copyright (c) 2024 SLAM-LLM contributors
# SPDX-FileCopyrightText: Copyright © 2025-2026 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-License-Identifier: MIT
#
# Modifications by Idiap Research Institute (© 2025-2026):
#   - Enhanced prompt parsing to support <speech>, <p:N>, and <p:TEXT> tokens
#   - Added flexible speech token placement anywhere in the prompt
#   - Added support for learnable token insertion and text-initialized learnable tokens
#   - Added dialogue context: <ctx:N> prompt token, pre-extracted context embeddings for the
#     context projector, and raw-text context ({context}) from previous turns or their keywords

import os
import re
import json
import copy
import torch
import logging
import whisper
import numpy as np

logger = logging.getLogger(__name__)

TOKEN_SPEECH = "<speech>"


class SpeechDatasetJsonl(torch.utils.data.Dataset):

    def __init__(
        self, dataset_config, tokenizer=None, processor=None, split="train", context_dim=768, llm_name="vicuna-7b-v1.5"
    ):
        global stoken_ix
        super().__init__()
        self.dataset_config = dataset_config
        self.tokenizer = tokenizer
        self.processor = processor
        self.llm_name = llm_name
        # data_parallel_size = dist.get_world_size()
        data_parallel_size = 1
        
        # Context projection
        self.context_token = False
        self.context_dim = context_dim  # 768 for Dialog2Flow embeddings
        # -1 includes all previous utterances as context; a <ctx:N> prompt token limits it to the last N
        self.number_of_ctx_utts = -1

        # self.data_list = contents
        self.IGNORE_INDEX = -100  # The default setting in CrossEntropyLoss
        self.AUDIO_TOKEN_ID = -1
        self.CONTEXT_TOKEN_ID = -2
        self.mel_size = dataset_config.get(
            "mel_size", 80
        )  # 80 for whisper large v1 and v2, 128 for large v3

        self.prompt = dataset_config.get("prompt", None)
        if self.prompt is None:            
            self.prompt = "Transcribe speech to text. Output the transcription directly without redundant content. Ensure that the output is not duplicated. "        
        self.prompt_template = dataset_config.get("prompt_template", None)
        if self.prompt_template:
            self.prompt = self.prompt_template.format(prompt=self.prompt)
        # If p-prompt embeddings initialization...
        logger.info(f"Input prompt: {self.prompt}")
        p_token = dataset_config.prompt_token
        re_token = f"{p_token[:-1]}:.+?{p_token[-1]}"
        m = re.search(re_token, self.prompt, flags=re.IGNORECASE|re.DOTALL)
        if m:
            logger.info(f"  Embedding initialization detected.")
            prompt_new = []
            for piece in re.split(f"({re_token})", self.prompt, flags=re.IGNORECASE|re.DOTALL):
                if re.match(re_token, piece, flags=re.IGNORECASE|re.DOTALL):
                    init_text = re.match(f"{p_token[:-1]}:(.+?){p_token[-1]}", piece, flags=re.IGNORECASE|re.DOTALL).group(1)
                    input_ids = tokenizer(init_text, add_special_tokens=False)["input_ids"]
                    logger.info(f"    -> Segment '{init_text}' initialized with embeddings for {input_ids} tokens.")
                    prompt_new.append(p_token * len(input_ids))
                else:
                    prompt_new.append(piece)
            self.prompt = "".join(prompt_new)
            logger.info(f"New input prompt: {self.prompt}")

        if dataset_config.prompt_embeddings_path:
            stoken_ix = 0
            def token_number(_):
                global stoken_ix
                stoken_ix = stoken_ix + 1
                return f"<p{stoken_ix - 1}>"
            self.prompt = re.sub(p_token, token_number, self.prompt)
            logger.info(f"Prompt embeddings path detected -> New input prompt: {self.prompt}")
            new_tokens = [f"<p{ix}>" for ix in range(stoken_ix)]
            for new_token in new_tokens:
                # one by one to force the right order, to match the indexes in the Embedding matrix
                tokenizer.add_special_tokens({'additional_special_tokens': [new_token]})
            logger.info(f"Adding new tokens to tokenizer: {new_tokens}")

        self.speech_token_ix = 0

        # Where the context comes from: pre-extracted embeddings, raw text, or both
        if dataset_config.enable_context_projector and dataset_config.context_embeddings_path is not None:
            self.context_embeddings_path = dataset_config.context_embeddings_path
        elif not dataset_config.enable_context_projector and dataset_config.context_embeddings_path is None and dataset_config.enable_raw_text_context:
            logger.info(f"Raw text context flag is on, so context embeddings will be generated on-the-fly using the LLM.")
            self.context_embeddings_path = None
        elif dataset_config.enable_context_projector and dataset_config.context_embeddings_path is None and not dataset_config.enable_raw_text_context:
            logger.info(f"No path to context embeddings are provided, neither raw context flag is on.\n Errors can ocurr, double check!!")
        # If a <ctx:N> token is in the prompt, N sets how many previous utterances are used as context
        ctx_token = dataset_config.context_token
        re_ctx_token = f"{ctx_token[:-1]}:.+?{ctx_token[-1]}"
        m_ctx = re.search(re_ctx_token, self.prompt, flags=re.IGNORECASE|re.DOTALL)
        if m_ctx:
            for piece in re.split(f"({re_ctx_token})", self.prompt, flags=re.IGNORECASE|re.DOTALL):
                if re.match(re_ctx_token, piece, flags=re.IGNORECASE|re.DOTALL):
                    self.number_of_ctx_utts = int(re.match(f"{ctx_token[:-1]}:(.+?){ctx_token[-1]}", piece, flags=re.IGNORECASE|re.DOTALL).group(1))
                    logger.info(f"Context token detected in the input prompt: {ctx_token}, and will be used to append {self.number_of_ctx_utts} (at most) previous utterances as context data.")
                    self.prompt = self.prompt.replace(piece, ctx_token)  # Remove the <ctx:.+?> token from the prompt and replace it with <ctx> token
                    logger.info(f"New input prompt: {self.prompt}")

        # If prompt_template contains <speech>, set the speech tokens flag and add token to tokenizer
        if TOKEN_SPEECH not in self.prompt:
            logger.info(f"No speech token ('{TOKEN_SPEECH}') was found in the input prompt. "
                        "Speech embeddings will be automatically prepend to the input.")
            self.prompt_ids = self.tokenizer.encode(self.prompt)
            self.speech_token_prefix = True
        else:
            tokenizer.add_special_tokens({'additional_special_tokens': [TOKEN_SPEECH]})
            speech_token_id = tokenizer.get_vocab()[TOKEN_SPEECH]
            if ctx_token in self.prompt:  # the context token must be a special token of the tokenizer
                logger.info(f"Context token ('{ctx_token}') was found in the input prompt.")
                current_special_tokens = tokenizer.additional_special_tokens
                new_tokens = [ctx_token]
                all_tokens = list(set(current_special_tokens + new_tokens))
                tokenizer.add_special_tokens({'additional_special_tokens':  all_tokens})
                context_token_id = tokenizer.get_vocab()[ctx_token]
            self.prompt_ids = self.tokenizer.encode(self.prompt)
            self.speech_token_ix = self.prompt_ids.index(speech_token_id)
            del self.prompt_ids[self.speech_token_ix]
            self.speech_token_prefix = False
            logger.info(f"Speech token found ('{TOKEN_SPEECH}') at index {self.speech_token_ix}.")
            if ctx_token in self.prompt:  # locate the context token in the prompt ids
                self.context_token_ix = self.prompt_ids.index(context_token_id)
                del self.prompt_ids[self.context_token_ix]
                logger.info(f"Context token found ('{ctx_token}') at index {self.context_token_ix}.")
                self.context_token=True

        self.answer_template = "{}"
        self.fix_length_audio = dataset_config.get("fix_length_audio", -1)
        self.inference_mode = dataset_config.get("inference_mode", False)
        self.normalize = dataset_config.get("normalize", False)
        self.target_lowercase = dataset_config.get("target_lowercase", False)
        self.input_type = dataset_config.get("input_type", None)
        assert self.input_type in ["raw", "mel"], "input_type must be one of [raw, mel]"

        self.data_list = []
        if split == "train":
            with open(dataset_config.train_data_path, encoding="utf-8") as fin:
                for line in fin:
                    data_dict = json.loads(line.strip())
                    self.data_list.append(data_dict)
        else:
            with open(dataset_config.val_data_path, encoding="utf-8") as fin:
                for line in fin:
                    data_dict = json.loads(line.strip())
                    self.data_list.append(data_dict)

    def read_context_embeddings_from_file(
            self, conversation_id, utt_idx, ctx_utts, ctx_dim=768, conversations_folder = ""
        ):
        """
        Read the context embeddings from a file based on the conversation ID and utterance index.
        :param conversation_id: The ID of the conversation.
        :param utt_idx: The index of the current utterance in the conversation.
        :param ctx_utts: Number of previous utterances to include as context.
        :return: A list of context embeddings.
        """
        context_embeddings = []
        if conversation_id is None or utt_idx is None:
            return [""]
        context_file_path = os.path.join(
                                conversations_folder,
                                conversation_id,
                                f"{conversation_id}_embedding.npy"
                            )
        if not os.path.exists(context_file_path):
            logger.warning(f"Context file {context_file_path} does not exist.")
            return np.empty((0, ctx_dim), dtype=np.float32)
        full_conversation_embeddings = np.load(context_file_path)
        if utt_idx == 0 or ctx_utts == 0:
            return np.empty((0, ctx_dim), dtype=np.float32) # No previous utterances or no context required
        elif ctx_utts == -1:
            return full_conversation_embeddings[:utt_idx]  # All previous utterances
        elif utt_idx < ctx_utts:    
            return full_conversation_embeddings[:utt_idx]  # All previous utterances available, but less than ctx_utts
        else:   
            return full_conversation_embeddings[utt_idx - ctx_utts:utt_idx]  # Last ctx_utts previous utterances
    
    def get_new_prompt_with_context(self, context):
        """
        Create a new prompt by inserting the context into the original prompt template.
        :param context: The context string to insert.
        :return: The new prompt string with context inserted.
        """
        if context is not None:
            if self.dataset_config.context_token in self.prompt:
                prompt = self.prompt.replace(self.dataset_config.context_token,"{context}")
                prompt = prompt.format(context=context)
            else:
                prompt = self.prompt.format(context=context)
        else:
            prompt = self.prompt

        if TOKEN_SPEECH not in prompt:
            prompt_ids = self.tokenizer.encode(prompt)
            speech_token_prefix = True
            speech_token_idx = 0
        else:
            speech_token_id = self.tokenizer.get_vocab()[TOKEN_SPEECH]
            prompt_ids = self.tokenizer.encode(prompt)
            speech_token_idx = prompt_ids.index(speech_token_id)
            del prompt_ids[speech_token_idx]
            speech_token_prefix = False
        return prompt_ids, prompt, speech_token_prefix, speech_token_idx
    
    def get_new_prompt_with_hybrid_context(self, context):
        """
        Create a new prompt by inserting the context into the original prompt template.
        :param context: The context string to insert.
        :return: The new prompt string with context inserted.
        """
        if self.dataset_config.context_token and context is not None:
            prompt = self.prompt.format(context=context)
        else:
            prompt = self.prompt

        if TOKEN_SPEECH not in prompt:
            prompt_ids = self.tokenizer.encode(prompt)
            speech_token_prefix = True
            speech_token_idx = 0
        else:
            speech_token_id = self.tokenizer.get_vocab()[TOKEN_SPEECH]
            context_token_id = self.tokenizer.get_vocab()[self.dataset_config.context_token]
            prompt_ids = self.tokenizer.encode(prompt)
            speech_token_idx = prompt_ids.index(speech_token_id)
            context_token_idx = prompt_ids.index(context_token_id)
            del prompt_ids[speech_token_idx]
            del prompt_ids[context_token_idx]
            speech_token_prefix = False
            
        return prompt_ids, prompt, speech_token_prefix, speech_token_idx, context_token_idx

    def get_source_len(self, data_dict):
        return data_dict["source_len"]

    def get_target_len(self, data_dict):

        return data_dict["target_len"] if "target_len" in data_dict else 0

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        data_dict = self.data_list[index]
        target = data_dict.get("target", None)
        key = data_dict.get("key", None)

        context = None
        avail_ctx = -1
        prompt_ids = None
        prompt = None
        speech_token_prefix = self.speech_token_prefix
        speech_token_idx = self.speech_token_ix
        context_token_idx = None

        # =========================
        # CONTEXT HANDLING
        # =========================
        if self.dataset_config.enable_context_projector and not self.dataset_config.enable_raw_text_context:
            utt_idx = data_dict.get("utt_idx", None)
            conversation_id = data_dict.get("conversation_id", None)
            context = self.read_context_embeddings_from_file(
                conversation_id, utt_idx,
                self.number_of_ctx_utts, self.context_dim,
                self.context_embeddings_path
            )
            avail_ctx = context.shape[0]

        elif not self.dataset_config.enable_context_projector and self.dataset_config.enable_raw_text_context:
            avail_ctx = 0
            if self.dataset_config.context_format == "keywords_history":
                ctx_items = data_dict["keywords_history"]
                raw_context = ", ".join(ctx_items).strip()
            else:
                ctx_items = data_dict["context"] if self.number_of_ctx_utts == -1 else data_dict["context"][-self.number_of_ctx_utts:]
                raw_context = " ".join(ctx_items).strip()

            prompt_ids, prompt, speech_token_prefix, speech_token_idx = \
                self.get_new_prompt_with_context(raw_context)

        elif self.dataset_config.enable_context_projector and self.dataset_config.enable_raw_text_context:
            utt_idx = data_dict.get("utt_idx", None)
            conversation_id = data_dict.get("conversation_id", None)
            context = self.read_context_embeddings_from_file(
                conversation_id, utt_idx,
                self.number_of_ctx_utts, self.context_dim,
                self.context_embeddings_path
            )
            avail_ctx = context.shape[0]

            if self.dataset_config.context_format == "keywords_history":
                ctx_items = data_dict["keywords_history"]
                raw_context = ", ".join(ctx_items).strip()
            else:
                ctx_items = data_dict["context"]
                raw_context = " ".join(ctx_items).strip()

            prompt_ids, prompt, speech_token_prefix, speech_token_idx, context_token_idx = \
                self.get_new_prompt_with_hybrid_context(raw_context)

        # =========================
        # AUDIO LOADING
        # =========================
        audio_raw = None
        audio_mel = None
        audio_length = 0

        audio_path = data_dict.get("source")
        audio_raw = whisper.load_audio(audio_path, sr=16000)  ## sr added to make sure all is in 16KHz

        if self.input_type == "raw":
            audio_raw = torch.from_numpy(audio_raw)
            if self.normalize:
                audio_raw = torch.nn.functional.layer_norm(audio_raw, audio_raw.shape)
            audio_length = len(audio_raw) // 320  # ad-hoc for fairseq 320x downsample
            audio_length = audio_length // 5  # ad-hoc for 5x fc downsample

        elif self.input_type == "mel":
            audio_raw = whisper.pad_or_trim(audio_raw)
            audio_mel = whisper.log_mel_spectrogram(
                audio_raw, n_mels=self.mel_size
            ).permute(1, 0)

            audio_length = (audio_mel.shape[0] + 1) // 2  # ad-hoc for whisper for 2x downsample from mel to feats
            audio_length = audio_length // 5  # ad-hoc for 5x fc downsample

        if self.fix_length_audio > 0:
            audio_length = self.fix_length_audio

        # =========================
        # PROMPT LENGTH
        # =========================
        if self.dataset_config.enable_raw_text_context:
            prompt_length = len(prompt_ids)
        else:
            prompt_length = len(self.prompt_ids)

        # =========================
        # INFERENCE MODE
        # =========================
        if self.inference_mode:
            if not self.dataset_config.enable_raw_text_context:
                example_ids = (
                    self.prompt_ids[:self.speech_token_ix]
                    + ([self.AUDIO_TOKEN_ID] * audio_length)
                    + self.prompt_ids[self.speech_token_ix:]
                )
                if self.context_token:
                    example_ids = (
                        example_ids[:self.context_token_ix]
                        + ([self.CONTEXT_TOKEN_ID] * avail_ctx)
                        + example_ids[self.context_token_ix:]
                    )

            else:
                example_ids = (
                    prompt_ids[:speech_token_idx]
                    + ([self.AUDIO_TOKEN_ID] * audio_length)
                    + prompt_ids[speech_token_idx:]
                )
                if self.context_token and context_token_idx is not None:
                    example_ids = (
                        example_ids[:context_token_idx]
                        + ([self.CONTEXT_TOKEN_ID] * avail_ctx)
                        + example_ids[context_token_idx:]
                    )

            example_ids = torch.tensor(example_ids, dtype=torch.int64)
            example_mask = example_ids.ge(-2)

            return {
                "input_ids": example_ids,
                "attention_mask": example_mask,
                "audio": audio_raw if self.input_type == "raw" else None,
                "audio_mel": audio_mel if self.input_type == "mel" else None,
                "audio_length": audio_length,
                "key": key,
                "target": target,
                "prompt_length": prompt_length,
                "context_length": avail_ctx if self.dataset_config.enable_context_projector else -1,
                "context": context if self.dataset_config.enable_context_projector else None,
            }

        # =========================
        # TRAINING MODE
        # =========================
        if self.dataset_config.enable_raw_text_context:
            base_prompt = prompt
            audio_token_ix = speech_token_idx
            audio_token_prefix = speech_token_prefix
        else:
            base_prompt = self.prompt
            audio_token_ix = self.speech_token_ix
            audio_token_prefix = self.speech_token_prefix

        example_ids = self.tokenizer.encode(
            base_prompt + self.answer_template.format(target)
        )

        if not audio_token_prefix:
            del example_ids[audio_token_ix]

        # AUDIO TOKENS
        audio_tokens_ids = [self.AUDIO_TOKEN_ID] * audio_length

        example_ids = (
            example_ids[:audio_token_ix]
            + audio_tokens_ids
            + example_ids[audio_token_ix:]
            + [self.tokenizer.eos_token_id]
        )

        # INSERT CONTEXT TOKENS
        if self.context_token and avail_ctx >= 0:
            ctx_ix = (
                context_token_idx
                if context_token_idx is not None
                else self.context_token_ix
            )

            if ctx_ix > audio_token_ix:
                ctx_ix += audio_length

            del example_ids[ctx_ix]
            example_ids = (
                example_ids[:ctx_ix]
                + ([self.CONTEXT_TOKEN_ID] * avail_ctx)
                + example_ids[ctx_ix:]
            )

        example_ids = torch.tensor(example_ids, dtype=torch.int64)

        labels_ids = copy.deepcopy(example_ids)
        labels_ids[: avail_ctx + audio_length + prompt_length] = -1

        example_mask = example_ids.ge(-2)
        label_mask = labels_ids.ge(0)

        example_ids[~example_mask] = 0
        labels_ids[~label_mask] = self.IGNORE_INDEX

        return {
            "input_ids": example_ids,
            "labels": labels_ids,
            "attention_mask": example_mask,
            "audio": audio_raw if self.input_type == "raw" else None,
            "audio_mel": audio_mel if self.input_type == "mel" else None,
            "audio_length": audio_length,
            "prompt_length": prompt_length,
            "context_length": avail_ctx if self.dataset_config.enable_context_projector else -1,
            "context": context if self.dataset_config.enable_context_projector else None,
            "target": target,
        }


    def pad(self, sequence, max_length, padding_idx=0):
        if isinstance(sequence, (int, list, tuple)):
            if len(sequence) < max_length:
                sequence = sequence + [padding_idx] * (max_length - len(sequence))
            else:
                sequence = sequence[:max_length]
        elif isinstance(sequence, torch.Tensor):
            if len(sequence) < max_length:
                sequence = torch.cat(
                    (
                        sequence,
                        torch.full(
                            ([max_length - len(sequence)] + list(sequence.size())[1:]),
                            padding_idx,
                        ),
                    )
                )
            else:
                sequence = sequence[:max_length]
        elif isinstance(sequence, np.ndarray):
            if len(sequence) < max_length:
                sequence = np.concatenate(
                    (
                        sequence,
                        np.full(
                            (max_length - len(sequence),) + sequence.shape[1:],
                            padding_idx,
                        ),
                    )
                )
            else:
                sequence = sequence[:max_length]
        else:
            raise Exception("Type mismatch during padding!")
        return sequence

    @classmethod
    def padding(cls, sequence, padding_length, padding_idx=0, padding_side="right"):
        if isinstance(sequence, (int, list, tuple)):
            if padding_length >= 0:
                sequence = sequence + [padding_idx] * padding_length
            else:
                sequence = sequence[:padding_length]
        elif isinstance(sequence, torch.Tensor):
            if sequence.ndimension() == 2:
                if padding_length >= 0:
                    sequence = torch.nn.functional.pad(sequence, (0, padding_length))
                else:
                    sequence = sequence[:, :padding_length]
            else:
                if padding_length >= 0:
                    if padding_side == "left":
                        sequence = torch.cat(
                            (
                                torch.full(
                                    ([padding_length] + list(sequence.size())[1:]),
                                    padding_idx,
                                ),
                                sequence,
                            )
                        )
                    else:
                        sequence = torch.cat(
                            (
                                sequence,
                                torch.full(
                                    ([padding_length] + list(sequence.size())[1:]),
                                    padding_idx,
                                ),
                            )
                        )
                else:
                    sequence = sequence[:padding_length]
        elif isinstance(sequence, np.ndarray):
            if padding_length >= 0:
                sequence = np.concatenate(
                    (
                        sequence,
                        np.full((padding_length,) + sequence.shape[1:], padding_idx),
                    )
                )
            else:
                sequence = sequence[:padding_length]
        else:
            raise Exception("Type mismatch during padding!")
        return sequence
    def collator(self, samples):
        assert samples is not None

        # =========================
        # LENGTH ACCOUNTING
        # =========================
        input_context_lengths = [s.get("context_length", 0) for s in samples]

        input_prompt_lengths = [
            s["audio_length"] + s["prompt_length"] + input_context_lengths[i]
            for i, s in enumerate(samples)
        ]

        input_answer_lengths = [
            len(s["input_ids"])
            - s["audio_length"]
            - s["prompt_length"]
            - input_context_lengths[i]
            for i, s in enumerate(samples)
        ]

        input_prompt_max_length = max(input_prompt_lengths)
        input_answer_max_length = max(input_answer_lengths)
        input_context_max_length = max(input_context_lengths) if input_context_lengths else 0

        # =========================
        # TOKEN PADDING
        # =========================
        input_ids = torch.stack([
            self.padding(
                self.padding(
                    samples[i]["input_ids"],
                    input_prompt_max_length - input_prompt_lengths[i],
                    self.tokenizer.pad_token_id,
                    padding_side="left",
                ),
                input_answer_max_length - input_answer_lengths[i],
                self.tokenizer.pad_token_id,
            )
            for i in range(len(samples))
        ])

        attention_mask = torch.stack([
            self.padding(
                self.padding(
                    samples[i]["attention_mask"],
                    input_prompt_max_length - input_prompt_lengths[i],
                    False,
                    padding_side="left",
                ),
                input_answer_max_length - input_answer_lengths[i],
                False,
            )
            for i in range(len(samples))
        ])

        # =========================
        # AUDIO
        # =========================
        audio_raw = None
        audio_mask = None
        audio_mel = None
        audio_mel_post_mask = None

        if self.input_type == "raw":
            audio_raw_max_length = max([s["audio"].shape[0] for s in samples])
            audio_raw = torch.stack(
                [self.pad(s["audio"], audio_raw_max_length, 0) for s in samples]
            )
            audio_mask = torch.zeros(len(samples), audio_raw_max_length)
            for line, sample in enumerate(samples):
                audio_mask[line, : sample["audio"].shape[0]] = 1
        elif self.input_type == "mel":
            audio_mel_max_length = max([s["audio_mel"].shape[0] for s in samples])
            audio_mel = torch.stack(
                [self.pad(s["audio_mel"], audio_mel_max_length, 0) for s in samples]
            )
            audio_mel_post_mask = torch.zeros(
                len(samples), (audio_mel_max_length + 1) // 2
            )  # ad-hoc for whisper for 2x downsample from mel to feats
            for line, sample in enumerate(samples):
                audio_mel_post_mask[line, : (sample["audio_mel"].shape[0] + 1) // 2] = 1

        # =========================
        # CONTEXT EMBEDDINGS
        # =========================
        input_context_embeddings = None
        if self.dataset_config.enable_context_projector and input_context_max_length > 0:
            input_context_embeddings = torch.zeros(
                len(samples), input_context_max_length, self.context_dim
            )
            for i, s in enumerate(samples):
                if s.get("context") is not None:
                    ctx = torch.tensor(s["context"], dtype=torch.float32)
                    seq_len = min(ctx.shape[0], input_context_max_length)
                    input_context_embeddings[i, :seq_len] = ctx[:seq_len]

        # =========================
        # MASKS
        # =========================
        modality_mask = input_ids == self.AUDIO_TOKEN_ID
        context_mask = input_ids == self.CONTEXT_TOKEN_ID

        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)

        # =========================
        # INFERENCE RETURN
        # =========================
        if self.inference_mode:
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "audio": audio_raw if self.input_type == "raw" else None,
                "audio_mask": audio_mask if self.input_type == "raw" else None,
                "audio_mel": audio_mel if self.input_type == "mel" else None,
                "audio_mel_post_mask": audio_mel_post_mask if self.input_type == "mel" else None,
                "modality_mask": modality_mask,
                "context_embeddings": input_context_embeddings if self.dataset_config.enable_context_projector else None,
                "context_mask": context_mask if self.dataset_config.enable_context_projector else None,
                "keys": [s["key"] for s in samples],
                "targets": [s["target"] for s in samples],
            }

        # =========================
        # TRAINING RETURN
        # =========================
        labels = torch.stack([
            self.padding(
                self.padding(
                    samples[i]["labels"],
                    input_prompt_max_length - input_prompt_lengths[i],
                    self.IGNORE_INDEX,
                    padding_side="left",
                ),
                input_answer_max_length - input_answer_lengths[i],
                self.IGNORE_INDEX,
            )
            for i in range(len(samples))
        ])

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "audio": audio_raw if self.input_type == "raw" else None,
            "audio_mask": audio_mask if self.input_type == "raw" else None,
            "audio_mel": audio_mel if self.input_type == "mel" else None,
            "audio_mel_post_mask": audio_mel_post_mask if self.input_type == "mel" else None,
            "modality_mask": modality_mask,
            "context_embeddings": input_context_embeddings if self.dataset_config.enable_context_projector else None,
            "context_mask": context_mask if self.dataset_config.enable_context_projector else None,
            "targets": [s["target"] for s in samples],
        }


def get_speech_dataset(dataset_config, tokenizer, processor, split, llm_name="vicuna-7b-v1.5", context_dim=768):
    dataset = SpeechDatasetJsonl(dataset_config, tokenizer, processor, split, context_dim=context_dim, llm_name=llm_name)

    return dataset
