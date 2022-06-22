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

"""
class BiTrainer(Trainer):
    def __init__(
            self,
            model: Union[PreTrainedModel, torch.nn.Module] = None,
            args: TrainingArguments = None,
            data_collator: Optional[DataCollator] = None,
            train_dataset: Optional[Dataset] = None,
            eval_dataset: Optional[List[Dataset]] = None,
            tokenizer: Optional[PreTrainedTokenizerBase] = None,
            model_init: Callable[[], PreTrainedModel] = None,
            compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
            callbacks: Optional[List[TrainerCallback]] = None,
            do_save_full_model: bool = True,
            do_save_adapters: bool = False,
            do_save_adapter_fusion: bool = False,
            adapter_names: Optional[List[List[str]]] = None,
            optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
        ):
        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            tokenizer=tokenizer,
            model_init=model_init,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            do_save_full_model=do_save_full_model,
            do_save_adapters=do_save_adapters,
            do_save_adapter_fusion=do_save_adapter_fusion,
            adapter_names=adapter_names,
            optimizers=optimizers
        )
        self.eval_dataset = eval_dataset
    def evaluate(
        self,
        eval_dataset: Optional[List[Dataset]] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:
        print("In evaluate bitrainer")
        print(eval_dataset)
        # memory metrics - must set up as early as possible
        self._memory_tracker.start()

        eval_dataloader1 = self.get_eval_dataloader(eval_dataset[0])
        eval_dataloader2 = self.get_eval_dataloader(eval_dataset[1])
        start_time = time.time()

        eval_loop = self.prediction_loop if self.args.use_legacy_prediction_loop else self.evaluation_loop
        output1 = eval_loop(
            eval_dataloader1,
            description="Evaluation",
            # No point gathering the predictions if there are no metrics, otherwise we defer to
            # self.args.prediction_loss_only
            prediction_loss_only=True if self.compute_metrics is None else None,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )
        output2 = eval_loop(
            eval_dataloader2,
            description="Evaluation",
            # No point gathering the predictions if there are no metrics, otherwise we defer to
            # self.args.prediction_loss_only
            prediction_loss_only=True if self.compute_metrics is None else None,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )

        print(output1)
        exit(1)

        total_batch_size = self.args.eval_batch_size * self.args.world_size
        output1.metrics.update(
            speed_metrics(
                metric_key_prefix,
                start_time,
                num_samples=output1.num_samples,
                num_steps=math.ceil(output1.num_samples / total_batch_size),
            )
        )
        
        output2.metrics.update(
            speed_metrics(
                metric_key_prefix,
                start_time,
                num_samples=output2.num_samples,
                num_steps=math.ceil(output2.num_samples / total_batch_size),
            )
        )

        self.log(output1.metrics)
        self.log(output2.metrics)
        
        if DebugOption.TPU_METRICS_DEBUG in self.args.debug:
            # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
            xm.master_print(met.metrics_report())

        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, output.metrics)

        self._memory_tracker.stop_and_update_metrics(output.metrics)

        return output.metrics
    
#    def evaluate(self, eval_dataset, ignore_keys=None, metric_key_prefix='eval'):
#        metrics1 = super().trainer.evaluate(eval_dataset[0], ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix)
#        try:
#            perplexity = math.exp(metrics1["eval_loss"])
#        except OverflowError:
#            perplexity = float("inf")
#        metrics1["perplexity"] = perplexity

#        metrics2 = super().trainer.evaluate(eval_dataset[1], ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix)
#        try:
#            perplexity = math.exp(metrics2["eval_loss"])
#        except OverflowError:
#            perplexity = float("inf")
#        metrics2["perplexity"] = perplexity
#        output.metrics.update(
#            "harmonic_perplexity" : hmean(metrics1["perplexity"], metrics2["perplexity"]),
#            speed_metrics(
#                metric_key_prefix,
#                metrics1.start_time,
#                "harmonic_perplexity" : hmean(metrics1["perplexity"], metrics2["perplexity"])
#                num_samples=metrics1.num_samples + metrics2.num.samples,
#            )
#        )
#        return output.metrics
"""


class _MultilingualDatasetIterator:

    def __init__(self, ml_dataset):
        self.datasets = sorted(list(ml_dataset.datasets.items()))
        # self.monolingual_generators = [
        #     dataset.generator(repeat=ml_dataset.training,
        #                       shuffle=ml_dataset.training)
        #     for _, dataset in datasets
        # ]

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

    def __next__(self):
        if self.training:
#            if self.step_count >= self.n_steps:
#                raise StopIteration()
#            print(self.current_gen)
            if self.step_count % self.batch_size == 0:
                self.current_gen = 0 if self.current_gen == 1 else 1
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
            validation_split_percentage=None, max_samples=None):
        self.training = training
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
            load_from_cache_file=overwrite_cache,
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
            load_from_cache_file=overwrite_cache,
            desc=f"Grouping texts in chunks of {self.max_seq_length}",
        ) for language, ds in self.datasets.items()}
        
        if max_samples is not None:
            self.datasets = { language : ds.select(range(max_samples)) if len(ds) > max_samples else ds
                              for language, ds in self.datasets.items()}

#        for i in self.datasets.values():
#            print(len(i))
        self.length = sum(
            len(dataset) for dataset in self.datasets.values())

    def __len__(self):
#        print(self.length)
        return self.length
#        if self.training:
#            return self.n_steps
#        else:
#            return self.length

    def __iter__(self):
        return _MultilingualDatasetIterator(self)

    
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
    os.environ["WANDB_DISABLED"] = "true"
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
    else:
        data_files = {}
        data_paths = get_training_files(data_args.train_file)
        data_paths = list(data_paths.values())
        data_files["train"] = data_paths[0]
        
        train_dataset1=None
        eval_dataset1=None
        extension = "text"
#        if data_args.train_file1 is not None:
#            data_files["train"] = data_args.train_file1
#        if data_args.validation_file1 is not None:
#            data_files["validation"] = data_args.validation_file1
        datasets1 = load_dataset(extension,
                                data_files=data_files,
                                cache_dir=model_args.cache_dir
        )
        if training_args.do_eval and "validation" not in datasets1.keys():
            datasets1["validation"] = load_dataset(extension,
                                                  data_files=data_files,
                                                  split=f"train[:{data_args.validation_split_percentage}%]",
                                                  cache_dir=model_args.cache_dir,
            )

        train_dataset2=None
        eval_dataset2=None
        data_files["train"] = data_paths[1]
#        if data_args.train_file2 is not None:
#            data_files["train"] = data_args.train_file2
#        if data_args.validation_file1 is not None:
#            data_files["validation"] = data_args.validation_file2
        datasets2 = load_dataset(extension,
                                data_files=data_files,
                                cache_dir=model_args.cache_dir
        )
        if training_args.do_eval and "validation" not in datasets2.keys():
            datasets2["validation"] = load_dataset(extension,
                                                  data_files=data_files,
                                                  split=f"train[:{data_args.validation_split_percentage}%]",
                                                  cache_dir=model_args.cache_dir,
            )

    datasets1.pop("train", None)
    datasets2.pop("train", None)
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

    if training_args.do_train:
        column_names = datasets1["train"].column_names
    else:
        column_names = datasets1["validation"].column_names
    text_column_name = "text" if "text" in column_names else column_names[0]

    if data_args.max_seq_length is None:
        max_seq_length = tokenizer.model_max_length
        if max_seq_length > 1024:
            logger.warning(
                f"The tokenizer picked seems to have a very large `model_max_length` ({tokenizer.model_max_length}). "
                "Picking 1024 instead. You can change that default value by passing --max_seq_length xxx."
            )
            max_seq_length = 1024
    else:
        if data_args.max_seq_length > tokenizer.model_max_length:
            logger.warning(
                f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
                f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
            )
        max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)
        
    def tokenize_function(examples):
        return tokenizer(examples[text_column_name], return_special_tokens_mask=True)

    tokenized_datasets1 = datasets1.map(
        tokenize_function,
        batched=True,
        num_proc=data_args.preprocessing_num_workers,
        remove_columns=column_names,
        load_from_cache_file=not data_args.overwrite_cache,
        desc="Running tokenizer on every text in dataset",
    )

    tokenized_datasets2 = datasets2.map(
        tokenize_function,
        batched=True,
        num_proc=data_args.preprocessing_num_workers,
        remove_columns=column_names,
        load_from_cache_file=not data_args.overwrite_cache,
        desc="Running tokenizer on every text in dataset",
    )

    # Main data processing function that will concatenate all texts from our dataset and generate chunks of
    # max_seq_length.
    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        total_length = (total_length // max_seq_length) * max_seq_length
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + max_seq_length] for i in range(0, total_length, max_seq_length)]
            for k, t in concatenated_examples.items()
        }
        return result

    # Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws away a
    # remainder for each of those groups of 1,000 texts. You can adjust that batch_size here but a higher value
    # might be slower to preprocess.
    #
    # To speed up this part, we use multiprocessing. See the documentation of the map method for more information:
    # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.map

    tokenized_datasets1 = tokenized_datasets1.map(
        group_texts,
        batched=True,
        num_proc=data_args.preprocessing_num_workers,
        load_from_cache_file=not data_args.overwrite_cache,
        desc=f"Grouping texts in chunks of {max_seq_length}",
    )

    tokenized_datasets2 = tokenized_datasets2.map(
        group_texts,
        batched=True,
        num_proc=data_args.preprocessing_num_workers,
        load_from_cache_file=not data_args.overwrite_cache,
        desc=f"Grouping texts in chunks of {max_seq_length}",
    )
    """
    if training_args.do_train:
        if "train" not in tokenized_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = tokenized_datasets["train"]
        if data_args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(data_args.max_train_samples))
    """
    if training_args.do_eval:
#        if "validation" not in tokenized_datasets1:
#            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset1 = tokenized_datasets1["validation"]
        eval_dataset2 = tokenized_datasets2["validation"]
        if data_args.max_eval_samples is not None:
            if len(eval_dataset1) > data_args.max_eval_samples:
                eval_dataset1 = eval_dataset1.select(range(data_args.max_eval_samples))
            if len(eval_dataset2) > data_args.max_eval_samples:
                eval_dataset2 = eval_dataset2.select(range(data_args.max_eval_samples))
            
    # Data collator
    # This one will take care of randomly masking the tokens.
    pad_to_multiple_of_8 = data_args.line_by_line and training_args.fp16 and not data_args.pad_to_max_length
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm_probability=data_args.mlm_probability,
        pad_to_multiple_of=8 if pad_to_multiple_of_8 else None,
    )

    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    """
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
    """
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
#    adapter_args.load_adapter = os.path.join(training_args.output_dir, "checkpoint-" + str(i), "mlm")
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

    # Evaluation
    last_dash_idx = adapter_args.load_adapter.rfind("-")
    last_slash_idx = adapter_args.load_adapter.rfind("/")
    checkpoint_num = adapter_args.load_adapter[last_dash_idx+1:last_slash_idx]
    results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        perplexities = []
        data_paths = get_training_files(data_args.train_file)
        eval_datasets = [eval_dataset1, eval_dataset2]
        for language, dataset in zip(data_paths.keys(), eval_datasets):
            logging.info('Evaluating on %s dataset' % language)
            eval_output = trainer.evaluate(dataset)
            perplexity = math.exp(eval_output["eval_loss"])
            perplexities.append(perplexity)
            result = {language + "_perplexity": perplexity}
            logging.info(result)
            results.update(result)
        harmonic_perplexity = hmean([perplexities[0], perplexities[1]])
        output_eval_file = os.path.join(training_args.output_dir, "eval_all", "eval_results_lm_" + checkpoint_num + ".txt")
        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results *****")
            for key in sorted(results.keys()):
                logger.info("  %s = %s", key, str(results[key]))
                writer.write("%s = %s\n" % (key, str(results[key])))
            logger.info("  %s = %s", "harmonic_perplexity", str(harmonic_perplexity))
            writer.write("%s = %s\n" % ("harmonic_perplexity", str(harmonic_perplexity)))


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
