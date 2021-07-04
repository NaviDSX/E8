# Assignment 8 
# Sequence-to-Sequence (seq2seq) models using Torchtext.

## * Tasks to be completed:

In this [repo](https://github.com/bentrevett/pytorch-seq2seq) refactor the 2 and 3 Notebooks (optional 4,for 500 additional points) such that:
  - is uses none of the legacy stuff
  - It MUST use Multi30k dataset from torchtext
  - uses yield_token, and other code that we wrote


## * Refactored Notebooks and Links / Comments:
Sl. |   Refactored Code -  Filename                                                                     |  Colab Link   | Comment
--- | ------------------------------------------------------------------------------------------------- | ------------- | -------------------
1   | [Sequence to Sequence Learning with Neural Networks](https://github.com/NaviDSX/E8/blob/main/1.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NaviDSX/E8/blob/main/1.ipynb) | In-Class
2   | [Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation](https://github.com/NaviDSX/E8/blob/main/2.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NaviDSX/E8/blob/main/2.ipynb) | Completed
3   | [Neural Machine Translation by Jointly Learning to Align and Translate](https://github.com/NaviDSX/E8/blob/main/3.ipynb)                         | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NaviDSX/E8/blob/main/3.ipynb) | Completed
4   | [Packed Padded Sequences, Masking, Inference and BLEU](https://github.com/NaviDSX/E8/blob/main/4.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NaviDSX/E8/blob/main/4.ipynb)  | WIP 


## * Notes:
Torchtext can preprocess the text input and prepare the data to train/validate a model with the following steps:

 * Train/validate/test split: generate train/validate/test data set if they are available
 * Tokenization: break a raw text string sentence into a list of words
 * Vocab: define a "contract" from tokens to indexes
 * Numericalize: convert a list of tokens to the corresponding indexes
 * Batch: generate batches of data samples and add padding if necessary

## * Steps Taken:

### For 2 and 3:
 - The template used in the classroom for refactoring is used as-is.
 - The encoder, decoder, seq2seq classes along with attention and kept unchanged.
 - This is run for one epcoh to validate refactoring.
 - No special comments are added.

### For 4:
I referred to the following [Notebook](https://github.com/pytorch/text/blob/master/examples/legacy_tutorial/migration_tutorial.ipynb) and made changes to use the new torchtext library.

The following things need to be taken care of:

LOAD DATA:
>When using packed padded sequences, we need to tell PyTorch how long the actual (non-padded) sequences are. Luckily for us, TorchText's Field objects allow us to use the >include_lengths argument, this will cause our batch.src to be a tuple. The first element of the tuple is the same as before, a batch of numericalized source sentence as a >tensor, and the second element is the non-padded lengths of each source sentence within the batch.

ITERATOR:
>One quirk about packed padded sequences is that all elements in the batch need to be sorted by their non-padded lengths in descending order, i.e. the first sentence in the >batch needs to be the longest. We use two arguments of the iterator to handle this, sort_within_batch which tells the iterator that the contents of the batch need to be sorted, >and sort_key a function which tells the iterator how to sort the elements in the batch. Here, we sort by the length of the src sentence.

ENCODER:
>Next up, we define the encoder.
>The changes here all within the forward method. It now accepts the lengths of the source sentences as well as the sentences themselves.
>After the source sentence (padded automatically within the iterator) has been embedded, we can then use pack_padded_sequence on it with the lengths of the sentences. Note that >the tensor containing the lengths of the sequences must be a CPU tensor as of the latest version of PyTorch, which we explicitly do so with to('cpu'). packed_embedded will then >be our packed padded sequence. This can be then fed to our RNN as normal which will return packed_outputs, a packed tensor containing all of the hidden states from the >sequence, and hidden which is simply the final hidden state from our sequence. hidden is a standard tensor and not packed in any way, the only difference is that as the input >was a packed sequence, this tensor is from the final non-padded element in the sequence.
>We then unpack our packed_outputs using pad_packed_sequence which returns the outputs and the lengths of each, which we don't need.
>The first dimension of outputs is the padded sequence lengths however due to using a packed padded sequence the values of tensors when a padding token was the input will be all >zeros.

ATTENTION:
>The attention module is where we calculate the attention values over the source sentence.
>Previously, we allowed this module to "pay attention" to padding tokens within the source sentence. However, using masking, we can force the attention to only be over non->padding elements.
>The forward method now takes a mask input. This is a [batch size, source sentence length] tensor that is 1 when the source sentence token is not a padding token, and 0 when it >is a padding token. For example, if the source sentence is: ["hello", "how", "are", "you", "?", <pad>, <pad>], then the mask would be [1, 1, 1, 1, 1, 0, 0].
>We apply the mask after the attention has been calculated, but before it has been normalized by the softmax function. It is applied using masked_fill. This fills the tensor at >each element where the first argument (mask == 0) is true, with the value given by the second argument (-1e10). In other words, it will take the un-normalized attention values, >and change the attention values over padded elements to be -1e10. As these numbers will be miniscule compared to the other values they will become zero when passed through the >softmax layer, ensuring no attention is payed to padding tokens in the source sentence.
  
DECODER:
>The decoder only needs a few small changes. It needs to accept a mask over the source sentence and pass this to the attention module. As we want to view the values of attention >during inference, we also return the attention tensor.
  
SEQ2SEQ:
>The overarching seq2seq model also needs a few changes for packed padded sequences, masking and inference.
>We need to tell it what the indexes are for the pad token and also pass the source sentence lengths as input to the forward method.
>We use the pad token index to create the masks, by creating a mask tensor that is 1 wherever the source sentence is not equal to the pad token. This is all done within the >create_mask function.
>The sequence lengths as needed to pass to the encoder to use packed padded sequences.
>The attention at each time-step is stored in the attentions

INFERENCE AND BLEU SCORES:
  
>
>
>
