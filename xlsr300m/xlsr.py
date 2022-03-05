#Fine-Tune XLS-R on Common Voice
#- [**Wav2Vec2-XLS-R-300M**](https://huggingface.co/facebook/wav2vec2-xls-r-300m)
#- [**Wav2Vec2-XLS-R-1B**](https://huggingface.co/facebook/wav2vec2-xls-r-1b)
#- [**Wav2Vec2-XLS-R-2B**](https://huggingface.co/facebook/wav2vec2-xls-r-2b)

# !pip install datasets==1.18.3
# !pip install transformers==4.11.3
# !pip install huggingface_hub==0.1
# !pip install torchaudio==0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
# !pip install jiwer
# !apt install git-lfs
from huggingface_hub import notebook_login
from datasets import load_dataset, load_metric, Audio
from datasets import ClassLabel
import random
import pandas as pd
import re
import json
from transformers import Wav2Vec2CTCTokenizer
from transformers import Wav2Vec2FeatureExtractor
from transformers import Wav2Vec2Processor
import torch
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from transformers import Wav2Vec2ForCTC
from transformers import TrainingArguments
from transformers import Trainer


online = True
try:
	request = requests.get("http://www.google.com", timeout=5)
	print("Connected to the Internet")
except (requests.ConnectionError, requests.Timeout) as exception:
    online = False
	print("No internet connection.")


common_voice_dataset = "/mnt/matylda3/xskura01/datasets/cv_8.0_cs"
model_path = "/mnt/matylda3/xskura01/gym/xlsr/wav2vec2-xls-r-300m"
if online:
    notebook_login()
    model_path = "facebook/wav2vec2-xls-r-300m"
    common_voice_dataset = "common_voice"

 
#def show_random_elements(dataset, num_examples=10):
#    assert num_examples <= len(dataset), "Can't pick more elements than there are in the dataset."
#    picks = []
#    for _ in range(num_examples):
#        pick = random.randint(0, len(dataset)-1)
#        while pick in picks:
#            pick = random.randint(0, len(dataset)-1)
#        picks.append(pick)
#    df = pd.DataFrame(dataset[picks])
#    display(HTML(df.to_html()))
#show_random_elements(common_voice_train.remove_columns(["path","audio"]))

def replace_hatted_characters(batch):
    batch["sentence"] = re.sub('[â]', 'a', batch["sentence"])
    batch["sentence"] = re.sub('[î]', 'i', batch["sentence"])
    batch["sentence"] = re.sub('[ô]', 'o', batch["sentence"])
    batch["sentence"] = re.sub('[û]', 'u', batch["sentence"])
    batch["sentence"] = re.sub('[ä]', 'a', batch["sentence"])
    batch["sentence"] = re.sub('[è]', 'e', batch["sentence"])
    batch["sentence"] = re.sub('[ï]', 'i', batch["sentence"])
    batch["sentence"] = re.sub('[ö]', 'o', batch["sentence"])
    return batch

def remove_special_characters(batch):
    chars_to_remove_regex = '[\–\—\/\„\…\,\?\.\!\-\;\:\"\“\%\‘\”\�\']'
    batch["sentence"] = re.sub(chars_to_remove_regex, '', batch["sentence"]).lower()
    return batch

def extract_all_chars(batch):
  all_text = " ".join(batch["sentence"])
  vocab = list(set(all_text))
  return {"vocab": [vocab], "all_text": [all_text]}

def prepare_dataset(batch):
    audio = batch["audio"]
    # batched output is "un-batched"
    batch["input_values"] = processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_values[0]
    batch["input_length"] = len(batch["input_values"])
    with processor.as_target_processor():
        batch["labels"] = processor(batch["sentence"]).input_ids
    return batch



columns_to_remove = ["accent", "age", "client_id", "down_votes", "gender", "locale", "segment", "up_votes"]
common_voice_train = load_dataset(common_voice, "cs", split="train+validation")
common_voice_test = load_dataset(common_voice, "cs", split="test")
common_voice_train = common_voice_train.remove_columns(columns_to_remove)
common_voice_test = common_voice_test.remove_columns(columns_to_remove)
common_voice_train = common_voice_train.map(remove_special_characters)
common_voice_test = common_voice_test.map(remove_special_characters)
common_voice_train = common_voice_train.map(replace_hatted_characters)
common_voice_test = common_voice_test.map(replace_hatted_characters)

vocab_train = common_voice_train.map(
        extract_all_chars, 
        batched=True,
        batch_size=-1,
        keep_in_memory=True,
        remove_columns=common_voice_train.column_names
)
vocab_test = common_voice_test.map(
        extract_all_chars,
        batched=True,
        batch_size=-1,
        keep_in_memory=True,
        remove_columns=common_voice_test.column_names
)

# Create the union of all distinct letters in the training dataset and test dataset and 
# convert the resulting list into an enumerated dictionary.
vocab_list = list(set(vocab_train["vocab"][0]) | set(vocab_test["vocab"][0]))
vocab_dict = {v: k for k, v in enumerate(sorted(vocab_list))}
print(vocab_dict)

# To make it clearer that `" "` has its own token class, we give it a more visible character `|`. 
# In addition, we also add an "unknown" token so that the model can later deal with characters 
# not encountered in Common Voice's training set.
vocab_dict["|"] = vocab_dict[" "]
del vocab_dict[" "]

# Add a padding token that corresponds to CTC's "*blank token*". 
# The "blank token" is a core component of the CTC algorithm. 
# For more information, take a look at the "Alignment" section [here](https://distill.pub/2017/ctc/).
vocab_dict["[PAD]"] = len(vocab_dict)
with open('vocab.json', 'w') as vocab_file:
    json.dump(vocab_dict, vocab_file)

# Use the json file to load the vocabulary into an instance of the `Wav2Vec2CTCTokenizer` class.
tokenizer = Wav2Vec2CTCTokenizer.from_pretrained("./", unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|")

repo_name = "wav2vec2-large-xls-r-300m-czech-colab"
if online:
    tokenizer.push_to_hub(repo_name)

# Create `Wav2Vec2FeatureExtractor`
#
#A `Wav2Vec2FeatureExtractor` object requires the following parameters to be instantiated:
#
# **feature_size**:          Speech models take a sequence of feature vectors as an input. 
#                            While the length of this sequence obviously varies, the feature size should not. 
#                            In the case of Wav2Vec2, the feature size is 1 because the model was trained on the raw speech signal.
# 
# **sampling_rate**:         The sampling rate at which the model is trained on.
#
# **padding_value**:         For batched inference, shorter inputs need to be padded with a specific value
#
# **do_normalize**:          Whether the input should be *zero-mean-unit-variance* normalized or not. 
#                            Usually, speech models perform better when normalizing the input
#
# **return_attention_mask**: Whether the model should make use of an **attention_mask** for 
#                            batched inference. 
#                            In general, XLS-R models checkpoints should **always** use the **attention_mask**.
feature_extractor = Wav2Vec2FeatureExtractor(
    feature_size=1, 
    sampling_rate=16000, 
    padding_value=0.0,
    do_normalize=True,
    return_attention_mask=True
)
processor = Wav2Vec2Processor(
    feature_extractor=feature_extractor,
    tokenizer=tokenizer
)

# Next, we can prepare the dataset.
# Thankfully, `datasets` does this automatically by calling the other column `audio`. Let try it out.
common_voice_train = common_voice_train.cast_column("audio", Audio(sampling_rate=16_000))
common_voice_test = common_voice_test.cast_column("audio", Audio(sampling_rate=16_000))

# Finally, we can use `Wav2Vec2Processor` to process the data 
# to the format expected by `Wav2Vec2ForCTC` for training.
# To do so let's make use of Dataset's 
# - First, we load and resample the audio data, simply by calling `batch["audio"]`.
# - Second, we extract the `input_values` from the loaded audio file. 
#   In our case, the `Wav2Vec2Processor` only normalizes the data. 
#   For other speech models, however, this step can include more complex feature extraction, 
#   such as [Log-Mel feature extraction](https://en.wikipedia.org/wiki/Mel-frequency_cepstrum). 
# - Third, we encode the transcriptions to label ids.
#
# For more information: 
#[docs](https://huggingface.co/transformers/master/model_doc/wav2vec2.html#transformers.Wav2Vec2Processor.__call__).
common_voice_train = common_voice_train.map(prepare_dataset, remove_columns=common_voice_train.column_names)
common_voice_test = common_voice_test.map(prepare_dataset, remove_columns=common_voice_test.column_names)

#max_input_length_in_sec = 5.0
#common_voice_train = common_voice_train.filter(lambda x: x < max_input_length_in_sec * processor.feature_extractor.sampling_rate, input_columns=["input_length"])

# ## Training
# The data is processed so that we are ready to start setting up the training pipeline. 
# https://huggingface.co/transformers/master/main_classes/trainer.html?highlight=trainer
# - Define a data collator. It is much more efficient to pad the training batches dynamically 
#   meaning that all training samples should only be padded to the longest sample in their batch 
#   and not the overall longest sample. 
#   Therefore, fine-tuning XLS-R requires a special padding data collator, which we will define below
# - Evaluation metric. During training, the model should be evaluated on the word error rate. 
#   We should define a `compute_metrics` function accordingly
# - Load a pretrained checkpoint. We need to load a pretrained checkpoint and configure it correctly for training.
# - Define the training configuration.
@dataclass
class DataCollatorCTCWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        processor (:class:`~transformers.Wav2Vec2Processor`)
            The processor used for proccessing the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
    """
    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lenghts and need
        # different padding methods
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            return_tensors="pt",
        )
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=self.padding,
                return_tensors="pt",
            )

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        batch["labels"] = labels
        return batch


data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)

# Next, the evaluation metric is defined. As mentioned earlier, the 
# predominant metric in ASR is the word error rate (WER), hence we will use it in this notebook as well.
wer_metric = load_metric("wer")

def compute_metrics(pred):
    """The model will return a sequence of logit vectors:
    A logit vector $\mathbf{y}_1$ contains the log-odds for each word in the vocabulary we defined earlier,
    thus $\text{len}(\mathbf{y}_i) =$ `config.vocab_size`. 
    We are interested in the most likely prediction of the model and thus take the `argmax(...)`
    of the logits. Also, we transform the encoded labels back to the original string by replacing 
    `-100` with the `pad_token_id` and decoding the ids while making sure that consecutive 
    tokens are **not** grouped to the same token in CTC style.
    """
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)
    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id
    pred_str = processor.batch_decode(pred_ids)
    # we do not want to group tokens when computing the metrics
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)
    wer = wer_metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer}

model = Wav2Vec2ForCTC.from_pretrained(
    model_path, 
    attention_dropout=0.0,
    hidden_dropout=0.0,
    feat_proj_dropout=0.0,
    mask_time_prob=0.05,
    layerdrop=0.0,
    ctc_loss_reduction="mean", 
    pad_token_id=processor.tokenizer.pad_token_id,
    vocab_size=len(processor.tokenizer),
)

# The first component of XLS-R consists of a stack of CNN layers that are used to extract 
# acoustically meaningful - but contextually independent - features from the raw speech signal.
# This part of the model has already been sufficiently trained during pretraining and as stated 
# in the [paper](https://arxiv.org/pdf/2006.13979.pdf) does not need to be fine-tuned anymore. 
# Thus, we can set the `requires_grad` to `False` for all parameters of the *feature extraction* part.
model.freeze_feature_extractor()

# In a final step, we define all parameters related to training. 
# To give more explanation on some of the parameters:
# - group_by_length makes training more efficient by grouping training samples of similar input 
#   length into one batch. This can significantly speed up training time by heavily reducing the 
#   overall number of useless padding tokens that are passed through the model
# - learning_rate and weight_decay were heuristically tuned until fine-tuning has become stable. 
#   Note that those parameters strongly depend on the Common Voice 
#   dataset and might be suboptimal for other speech datasets.
#   
# For more info on other parameters, 
# one can take a look at the [docs](https://huggingface.co/transformers/master/main_classes/trainer.html?highlight=trainer#trainingarguments).
# During training, a checkpoint will be uploaded asynchronously to the hub every 400 training steps. ------- NOOOOOO
# It allows you to also play around with the demo widget even while your model is still training.
# 
# **Note**: If one does not want to upload the model checkpoints to the hub, simply set `push_to_hub=False`.
# TODO: fix?
training_args = TrainingArguments(
  output_dir=repo_name,
  group_by_length=True,
  per_device_train_batch_size=16,
  gradient_accumulation_steps=2,
  evaluation_strategy="steps",
  num_train_epochs=30,
  gradient_checkpointing=True,
  fp16=True,
  save_steps=400,
  eval_steps=400,
  logging_steps=100000, # dont want to log
  learning_rate=3e-4,
  warmup_steps=500,
  save_total_limit=2,
  push_to_hub=False,
)
trainer = Trainer(
    model=model,
    data_collator=data_collator,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=common_voice_train,
    eval_dataset=common_voice_test,
    tokenizer=processor.feature_extractor,
)
trainer.train()

if online:
    trainer.push_to_hub()
    model = Wav2Vec2ForCTC.from_pretrained(repo_name).to("cuda")
    processor = Wav2Vec2Processor.from_pretrained(repo_name)

input_dict = processor(common_voice_test[0]["input_values"], return_tensors="pt", padding=True)
logits = model(input_dict.input_values.to("cuda")).logits
pred_ids = torch.argmax(logits, dim=-1)[0]

# We adapted `common_voice_test` quite a bit so that the dataset instance does not
# contain the original sentence label anymore. Thus, we re-use the original dataset 
# to get the label of the first example.
common_voice_test_transcription = load_dataset("common_voice", "tr", data_dir="./cv-corpus-6.1-2020-12-11", split="test")

print("Prediction:")
print(processor.decode(pred_ids))

print("\nReference:")
print(common_voice_test_transcription[0]["sentence"].lower())

