import logging
import math
import os
import sys
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Union, Tuple, Callable
from scipy.stats import hmean
import torch

from datasets import load_dataset, Dataset

import transformers
import transformers.adapters.composition as ac
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_MASKED_LM_MAPPING,
    AutoConfig,
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    HfArgumentParser,
    MultiLingAdapterArguments,
    Trainer,
    TrainingArguments,
    set_seed,
)

from torch.utils.data.dataset import IterableDataset
from transformers.adapters.configuration import AdapterConfig
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version
from transformers.utils.versions import require_version
from transformers.modeling_utils import PreTrainedModel
from transformers.data.data_collator import DataCollator
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.trainer_utils import EvalPrediction
from transformers.trainer_callback import TrainerCallback
# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.8.0")

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")

logger = logging.getLogger(__name__)
MODEL_CONFIG_CLASSES = list(MODEL_FOR_MASKED_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


class _MultilingualDatasetIterator:

    def __init__(self, ml_dataset, k):
        self.datasets = list(ml_dataset.datasets.items())
        self.k = k

        if ml_dataset.weighted_sampling:
            lengths = np.array([len(dataset) for _, dataset in self.datasets], dtype=np.float32)
            smoothed_lengths = lengths ** ml_dataset.smoothing
            self.sample_prob = smoothed_lengths / np.sum(smoothed_lengths)
        else:
            self.sample_prob = np.ones([len(self.datasets)], dtype=np.float32) / len(self.datasets)
        logging.info('Languages will be sampled in the following proportions:')
        for i, (language, _) in enumerate(self.datasets):
            logging.info('%s: %.7f' % (language, self.sample_prob[i]))
        self.monolingual_generators = [iter(dataset) for _, dataset in self.datasets]
#        print(next(self.monolingual_generators[0]))
        self.batch_size = ml_dataset.batch_size
        self.training = ml_dataset.training
        self.n_steps = ml_dataset.n_steps
        self.step_count = 0
        self.current = 0
        self.current_gen = 1
        self.count_en_batches = 0

    def __next__(self):
        if self.training:
            if self.k == 1:
                if len(self.monolingual_generators) == 2:
                    if self.step_count % self.batch_size == 0:
                        self.current_gen = 0 if self.current_gen == 1 else 1
                else:
                    if self.step_count % self.batch_size == 0:
                        self.current_gen = 0 if self.step_count == 0 else (self.current_gen + 1) % len(self.monolingual_generators)
            else:
                if self.step_count % self.batch_size == 0:
                    if self.count_en_batches == self.k - 1:
                        assert self.current_gen == 0
                        self.current_gen = 1
                        self.count_en_batches = 0
                    elif self.current_gen == 1:
                        self.current_gen = 0
                    else:
                        self.count_en_batches += 1
            # print(self.current_gen)

            dataset = self.monolingual_generators[self.current_gen]
#            dataset = np.random.choice(self.monolingual_generators, p=self.sample_prob)
            try:
                batch = next(dataset)
            except StopIteration:
                self.monolingual_generators[self.current_gen] = [iter(dataset) for _, dataset in self.datasets][self.current_gen]
                dataset = self.monolingual_generators[self.current_gen]
                batch = next(dataset)
            self.step_count += 1
            return batch

        else:
            while True:
                if self.current >= len(self.monolingual_generators):
                    raise StopIteration()

                try:
                    batch = next(self.monolingual_generators[self.current])
                except StopIteration:
                    self.current += 1
                    continue
                return batch


class MultilingualDataset(IterableDataset):

    def __init__(
            self, tokenizer, files_by_language, batch_size, cache_dir,
            overwrite_cache=False, training=False, weighted_sampling=True, smoothing=0.7, n_steps=250000, max_seq_length=512,
            validation_split_percentage=None, max_samples=None, k=1):
        self.training = training
        self.k = k
        self.split = 'train' if self.training else 'validation'
        self.weighted_sampling = weighted_sampling
        self.smoothing = smoothing
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length
        self.extension = 'text'
        self.text_column_name = 'text'
        languages = ', '.join(sorted(list(files_by_language.keys())))
        logging.info('Initialising multilingual dataset with languages ' + languages)
        if validation_split_percentage is not None:
            if self.training:
                self.datasets = {
                    language: load_dataset(self.extension, data_files = {self.split : file_path}, split=f"train[{validation_split_percentage}%:]", cache_dir=cache_dir)#[self.split]
                    for language, file_path in files_by_language.items()
                }
            else:
                self.datasets = {
                    language: load_dataset(self.extension, data_files = {self.split : file_path}, split=f"validation[:{validation_split_percentage}%]", cache_dir=cache_dir)#[self.split]
                    for language, file_path in files_by_language.items()
                }
        else:
            self.datasets = {
                language: load_dataset(self.extension, data_files = {self.split : file_path}, cache_dir=cache_dir)[self.split]
                for language, file_path in files_by_language.items()
            }
        column_names = { language : ds.column_names for language, ds in self.datasets.items()}
        # text_column_names = { language : "text" if "text" in names else names[0] for language, names in column_names.items()}

        def tokenize_function(examples):
            return tokenizer(examples[self.text_column_name])
                             #,return_special_tokens_mask=True)  # , max_length=max_seq_length, truncation=True)

        self.datasets = { language : ds.map(
            tokenize_function,
            batched=True,
            remove_columns=column_names[language],
            load_from_cache_file=True,
            desc="Running tokenizer on every text in dataset",
        ) for language, ds in self.datasets.items()}

        def group_texts(examples):
            # Concatenate all texts.
            concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
            total_length = len(concatenated_examples[list(examples.keys())[0]])
            # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
            # customize this part to your needs.
            total_length = (total_length // self.max_seq_length) * self.max_seq_length
            # Split by chunks of max_len.
            result = {
                k: [t[i : i + self.max_seq_length] for i in range(0, total_length, self.max_seq_length)]
                for k, t in concatenated_examples.items()
            }
            return result

        # Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws away a
        # remainder for each of those groups of 1,000 texts. You can adjust that batch_size here but a higher value
        # might be slower to preprocess.
        #
        # To speed up this part, we use multiprocessing. See the documentation of the map method for more information:
        # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.map

        self.datasets = { language : ds.map(
            group_texts,
            batched=True,
            load_from_cache_file=True,
            desc=f"Grouping texts in chunks of {self.max_seq_length}",
        ) for language, ds in self.datasets.items()}
        
        if max_samples is not None:
            self.datasets = { language : ds.select(range(max_samples)) if len(ds) > max_samples else ds
                              for language, ds in self.datasets.items()}

        self.length = sum(
            len(dataset) for dataset in self.datasets.values())

    def __len__(self):
        return self.length

    def __iter__(self):
        return _MultilingualDatasetIterator(self, self.k)

    
@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization."
            "Don't set if you want to train a model from scratch."
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    config_overrides: Optional[str] = field(
        default=None,
        metadata={
            "help": "Override some existing default config settings when a model is trained from scratch. Example: "
            "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index"
        },
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )

    def __post_init__(self):
        if self.config_overrides is not None and (self.config_name is not None or self.model_name_or_path is not None):
            raise ValueError(
                "--config_overrides can't be used in combination with --config_name or --model_name_or_path"
            )

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    k: Optional[int] = field(
        default=1, metadata={"help": "The sampling ratio k to 1."}
    )
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    max_seq_length: Optional[int] = field(
        default=None,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated."
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    mlm_probability: float = field(
        default=0.15, metadata={"help": "Ratio of tokens to mask for masked language modeling loss"}
    )
    line_by_line: bool = field(
        default=False,
        metadata={"help": "Whether distinct lines of text in the dataset are to be handled as distinct sequences."},
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )

    def __post_init__(self):
        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
#        else:
#            if self.train_file is not None:
#                extension = self.train_file.split(".")[-1]
#                assert extension in ["csv", "json", "txt"], "`train_file` should be a csv, a json or a txt file."
#            if self.validation_file is not None:
#                extension = self.validation_file.split(".")[-1]
#                assert extension in ["csv", "json", "txt"], "`validation_file` should be a csv, a json or a txt file."

def get_training_files(file_path):
    files_by_language = {}
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            language, path = line.split()
            if not os.path.isabs(path):
                path = os.path.join(os.path.dirname(file_path), path)
            files_by_language[language] = path
    return files_by_language

def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments, MultiLingAdapterArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args, adapter_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, data_args, training_args, adapter_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO if training_args.should_log else logging.WARN)

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if training_args.should_log:
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column. You can easily tweak this
    # behavior (see below)
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.

    if data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        datasets = load_dataset(data_args.dataset_name, data_args.dataset_config_name, cache_dir=model_args.cache_dir)
        if "validation" not in datasets.keys():
            datasets["validation"] = load_dataset(
                data_args.dataset_name,
                data_args.dataset_config_name,
                split=f"train[:{data_args.validation_split_percentage}%]",
                cache_dir=model_args.cache_dir,
            )
            datasets["train"] = load_dataset(
                data_args.dataset_name,
                data_args.dataset_config_name,
                split=f"train[{data_args.validation_split_percentage}%:]",
                cache_dir=model_args.cache_dir,
            )
#    else:
        
    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    if model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name, **config_kwargs)
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)
    else:
        config = CONFIG_MAPPING[model_args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")
        if model_args.config_overrides is not None:
            logger.info(f"Overriding config: {model_args.config_overrides}")
            config.update_from_string(model_args.config_overrides)

    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, **tokenizer_kwargs)
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, **tokenizer_kwargs)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    if model_args.model_name_or_path:
        model = AutoModelForMaskedLM.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
    else:
        logger.info("Training new model from scratch")
        model = AutoModelForMaskedLM.from_config(config)

    model.resize_token_embeddings(len(tokenizer))

    # Setup adapters
    if adapter_args.train_adapter:
        task_name = data_args.dataset_name or "mlm"
        # check if adapter already exists, otherwise add it
        if task_name not in model.config.adapters:
            # resolve the adapter config
            adapter_config = AdapterConfig.load(
                adapter_args.adapter_config,
                non_linearity=adapter_args.adapter_non_linearity,
                reduction_factor=adapter_args.adapter_reduction_factor,
            )
            # load a pre-trained from Hub if specified
            if adapter_args.load_adapter:
                model.load_adapter(
                    adapter_args.load_adapter,
                    config=adapter_config,
                    load_as=task_name,
                )
            # otherwise, add a fresh adapter
            else:
                model.add_adapter(task_name, config=adapter_config)
        # optionally load a pre-trained language adapter
        if adapter_args.load_lang_adapter:
            # resolve the language adapter config
            lang_adapter_config = AdapterConfig.load(
                adapter_args.lang_adapter_config,
                non_linearity=adapter_args.lang_adapter_non_linearity,
                reduction_factor=adapter_args.lang_adapter_reduction_factor,
            )
            # load the language adapter from Hub
            lang_adapter_name = model.load_adapter(
                adapter_args.load_lang_adapter,
                config=lang_adapter_config,
                load_as=adapter_args.language,
            )
        else:
            lang_adapter_name = None
        # Freeze all model weights except of those of this adapter
        model.train_adapter([task_name])
        # Set the adapters to be used in every forward pass
        if lang_adapter_name:
            model.set_active_adapters(ac.Stack(lang_adapter_name, task_name))
        else:
            model.set_active_adapters(task_name)
    else:
        if adapter_args.load_adapter or adapter_args.load_lang_adapter:
            raise ValueError(
                "Adapters can only be loaded in adapters training mode."
                "Use --train_adapter to enable adapter training"
            )


    train_dataset=None
    eval_dataset=None
    if training_args.do_train:
        train_files_by_language = get_training_files(data_args.train_file)
        train_dataset = MultilingualDataset(tokenizer=tokenizer,
                                            files_by_language=train_files_by_language,
                                            batch_size=training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps,
                                            cache_dir=model_args.cache_dir,
                                            overwrite_cache=data_args.overwrite_cache,
                                            training=True,
                                            weighted_sampling=False,
                                            smoothing=0.5,
                                            n_steps=training_args.max_steps,
                                            max_seq_length=data_args.max_seq_length,
                                            validation_split_percentage=data_args.validation_split_percentage,
                                            max_samples=data_args.max_train_samples,
                                            k=data_args.k
                                            )
    if training_args.do_eval:
        if data_args.validation_split_percentage is not None:
            eval_files_by_language = get_training_files(data_args.train_file)
            eval_files_by_language.pop('en', None)
            file_path = list(eval_files_by_language.values())[0]
            # file_path = eval_files_by_language['en']
            eval_dataset = load_dataset('text', data_files = {"validation" : file_path}, split=f"validation[:{data_args.validation_split_percentage}%]", cache_dir=model_args.cache_dir)
        else:
            # validation split is None, should have eval files
            eval_files_by_language = get_training_files(data_args.validation_file)
            eval_files_by_language.pop('en', None)
            file_path = list(eval_files_by_language.values())[0]
            # file_path = eval_files_by_language['en']
            eval_dataset = load_dataset('text', data_files = {"validation" : file_path}, cache_dir=model_args.cache_dir)["validation"]
        column_names = eval_dataset.column_names
        # text_column_names = { language : "text" if "text" in names else names[0] for language, names in column_names.items()}

        def tokenize_function(examples):
            return tokenizer(examples["text"])
                             #,return_special_tokens_mask=True)  # , max_length=max_seq_length, truncation=True)

        eval_dataset = eval_dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=column_names,
            load_from_cache_file=data_args.overwrite_cache,
            desc="Running tokenizer on every text in dataset",
        )

        def group_texts(examples):
            # Concatenate all texts.
            concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
            total_length = len(concatenated_examples[list(examples.keys())[0]])
            # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
            # customize this part to your needs.
            total_length = (total_length // data_args.max_seq_length) * data_args.max_seq_length
            # Split by chunks of max_len.
            result = {
                k: [t[i : i + data_args.max_seq_length] for i in range(0, total_length, data_args.max_seq_length)]
                for k, t in concatenated_examples.items()
            }
            return result

        # Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws away a
        # remainder for each of those groups of 1,000 texts. You can adjust that batch_size here but a higher value
        # might be slower to preprocess.
        #
        # To speed up this part, we use multiprocessing. See the documentation of the map method for more information:
        # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.map

        eval_dataset = eval_dataset.map(
            group_texts,
            batched=True,
            load_from_cache_file=data_args.overwrite_cache,
            desc=f"Grouping texts in chunks of {data_args.max_seq_length}",
        )
        
        if data_args.max_eval_samples is not None:
            eval_dataset = eval_dataset.select(range(data_args.max_eval_samples)) if len(eval_dataset) > data_args.max_eval_samples else eval_dataset

    # if training_args.do_train:
    #     if "train" not in tokenized_datasets:
    #         raise ValueError("--do_train requires a train dataset")
    #     train_dataset = tokenized_datasets["train"]
    #     if data_args.max_train_samples is not None:
    #         train_dataset = train_dataset.select(range(data_args.max_train_samples))
    #
    # if training_args.do_eval:
    #     if "validation" not in tokenized_datasets:
    #         raise ValueError("--do_eval requires a validation dataset")
    #     eval_dataset = tokenized_datasets["validation"]
    #     if data_args.max_eval_samples is not None:
    #         eval_dataset = eval_dataset.select(range(data_args.max_eval_samples))
#    if data_args.max_eval_samples is not None:
#        eval_dataset = eval_dataset.select(range(data_args.max_eval_samples))
    # Data collator
    # This one will take care of randomly masking the tokens.
    pad_to_multiple_of_8 = data_args.line_by_line and training_args.fp16 and not data_args.pad_to_max_length
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm_probability=data_args.mlm_probability
    )

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        do_save_full_model=not adapter_args.train_adapter,
        do_save_adapters=adapter_args.train_adapter,
    )

    # Training
    if training_args.do_train:
        model_path = (
            model_args.model_name_or_path
            if model_args.model_name_or_path is not None and os.path.isdir(model_args.model_name_or_path)
            else None
        )
        trainer.train(model_path=model_path)
        trainer.save_model()
        # For convenience, we also re-save the tokenizer to the same directory,
        # so that you can share your model easily on huggingface.co/models =)
        #if trainer.is_world_master():
        tokenizer.save_pretrained(training_args.output_dir)

    # Evaluation
    results = {}
    if training_args.do_eval:
         logger.info("*** Evaluate ***")
         
         if not isinstance(eval_dataset, dict):
             language = list(eval_files_by_language.keys())[0]
             eval_dataset = {language: eval_dataset}
         for language, dataset in eval_dataset.items():
             logging.info('Evaluating on %s dataset' % language)
             eval_output = trainer.evaluate(dataset)
    
             perplexity = math.exp(eval_output["eval_loss"])
             result = {language + "_perplexity": perplexity}
             logging.info(result)
             results.update(result)
         output_eval_file = os.path.join(training_args.output_dir, "eval_results_lm.txt")
         with open(output_eval_file, "w") as writer:
             logger.info("***** Eval results *****")
             for key in sorted(results.keys()):
                 logger.info("  %s = %s", key, str(results[key]))
                 writer.write("%s = %s\n" % (key, str(results[key])))
         print(results)
    return results


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
