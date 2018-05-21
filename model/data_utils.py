import re
import os
import numpy as np
import pandas as pd

# shared global variables to be imported from model also
UNK = "#UNK#"
NUM = "#NUM#"
NONE = "O"
tag_r = re.compile("\[(?:@|\$)(.*)#(.*)\*")


# special error message
class MyIOError(Exception):
    def __init__(self, filename):
        # custom error message
        message = """
        ERROR: Unable to locate file {}.FIX: Have you tried running python build_data.py first?
        This will build vocab file from your train, test and dev sets and
        trimm your word vectors.""".format(filename)
        super(MyIOError, self).__init__(message)


class CoNLLDataset(object):
    """Class that iterates over CoNLL Dataset

    __iter__ method yields a tuple (words, tags)
        words: list of raw words
        tags: list of raw tags

    If processing_word and processing_tag are not None,
    optional preprocessing is appplied

    Example:
        ```python
        data = CoNLLDataset(filename)
        for sentence, tags in data:
            pass
        ```

    """

    def __init__(self, filename, processing_word=None, processing_tag=None,
                 max_iter=None):
        """
        Args:
            filename: path to the file
            processing_words: (optional) function that takes a word as input
            processing_tags: (optional) function that takes a tag as input
            max_iter: (optional) max number of sentences to yield

        """
        self.filename = filename
        self.processing_word = processing_word
        self.processing_tag = processing_tag
        self.max_iter = max_iter
        self.length = None

    def __iter__(self):
        n_iter = 0
        with open(self.filename) as f:
            words, tags = [], []
            for line in f:
                line = line.strip()
                if len(line) == 0:
                    if len(words) != 0:
                        n_iter += 1
                        if self.max_iter and n_iter > self.max_iter:
                            break
                        yield words, tags
                        words, tags = [], []
                else:
                    ls = re.split("\s+|\t+", line)
                    word, tag = ls[0], ls[-1]
                    if self.processing_word:
                        word = self.processing_word(word)
                    if self.processing_tag:
                        tag = self.processing_tag(tag)
                    words.append(word)
                    tags.append(tag)

    def __len__(self):
        """Iterates once over the corpus to set and store length"""
        if self.length is None:
            self.length = 0
            for _ in self:
                self.length += 1
        return self.length


def get_vocabs(datasets):
    """Build vocabulary from an iterable of datasets objects

    Args:
        datasets: a list of dataset objects

    Returns:
        a set of all the words in the dataset

    """
    print("Building vocab...")
    vocab_words = set()
    vocab_tags = set()
    for dataset in datasets:
        for words, tags in dataset:
            vocab_words.update(words)
            vocab_tags.update(tags)
    print("- done. {} tokens".format(len(vocab_words)))
    return vocab_words, vocab_tags


def get_char_vocab(dataset):
    """Build char vocabulary from an iterable of datasets objects

    Args:
        dataset: a iterator yielding tuples (sentence, tags)

    Returns:
        a set of all the characters in the dataset

    """
    vocab_char = set()
    for words, _ in dataset:
        for word in words:
            vocab_char.update(word)

    return vocab_char


def get_glove_vocab(filename):
    """Load vocab from file

    Args:
        filename: path to the glove vectors

    Returns:
        vocab: set() of strings
    """
    print("Building vocab...")
    vocab = set()
    with open(filename) as f:
        for line in f:
            word = line.strip().split(' ')[0]
            vocab.add(word)
    print("- done. {} tokens".format(len(vocab)))
    return vocab


def write_vocab(vocab, filename):
    """Writes a vocab to a file

    Writes one word per line.

    Args:
        vocab: iterable that yields word
        filename: path to vocab file

    Returns:
        write a word per line

    """
    print("Writing vocab...")
    with open(filename, "w") as f:
        for i, word in enumerate(vocab):
            if i != len(vocab) - 1:
                f.write("{}\n".format(word))
            else:
                f.write(word)
    print("- {} done. {} tokens".format(filename, len(vocab)))


def load_vocab(filename):
    """Loads vocab from a file

    Args:
        filename: (string) the format of the file must be one word per line.

    Returns:
        d: dict[word] = index

    """
    try:
        d = dict()
        with open(filename) as f:
            for idx, word in enumerate(f):
                word = word.strip()
                d[word] = idx
    except IOError:
        raise MyIOError(filename)
    return d


def export_trimmed_glove_vectors(
        vocab, glove_filename, trimmed_filename, dim):
    """Saves glove vectors in numpy array

    Args:
        vocab: dictionary vocab[word] = index
        glove_filename: a path to a glove file
        trimmed_filename: a path where to store a matrix in npy
        dim: (int) dimension of embeddings

    """
    print("Export glove vectors...")
    embeddings = np.zeros([len(vocab), dim])
    with open(glove_filename) as f:
        for line in f:
            line = line.strip().split(' ')
            word = line[0]
            embedding = [float(x) for x in line[1:]]
            if word in vocab:
                word_idx = vocab[word]
                embeddings[word_idx] = np.asarray(embedding)
    np.savez_compressed(trimmed_filename, embeddings=embeddings)
    print("- save trimmed glove vectors done.")


def get_trimmed_glove_vectors(filename):
    """
    Args:
        filename: path to the npz file

    Returns:
        matrix of embeddings (np array)

    """
    try:
        with np.load(filename) as data:
            return data["embeddings"]

    except IOError:
        raise MyIOError(filename)


def get_processing_word(vocab_words=None, vocab_chars=None,
                        lowercase=False, chars=False, allow_unk=True):
    """Return lambda function that transform a word (string) into list,
    or tuple of (list, id) of int corresponding to the ids of the word and
    its corresponding characters.

    Args:
        vocab: dict[word] = idx

    Returns:
        f("cat") = ([12, 4, 32], 12345)
                 = (list of char ids, word id)

    """
    def f(word):
        # 0. get chars of words
        if vocab_chars and chars:
            char_ids = []
            for char in word:
                # ignore chars out of vocabulary
                if char in vocab_chars:
                    char_ids.extend([vocab_chars[char]])
        # 1. preprocess word
        if lowercase:
            word = word.lower()
        if word.isdigit():
            word = NUM
        # 2. get id of word
        if vocab_words:
            if word in vocab_words:
                word = vocab_words[word]
            else:
                if allow_unk:
                    word = vocab_words[UNK]
                else:
                    raise Exception(
                        "Unknow key is not allowed. Check that your vocab (tags?) is correct")
        # 3. return tuple char ids, word id
        if vocab_chars and chars:
            return char_ids, word
        else:
            return word

    return f


def _pad_sequences(sequences, pad_tok, max_length):
    """
    Args:
        sequences: a generator of list or tuple
        pad_tok: the char to pad with

    Returns:
        a list of list where each sublist has same length
    """
    sequence_padded, sequence_length = [], []

    for seq in sequences:
        seq = list(seq)
        seq_ = seq[:max_length] + [pad_tok] * max(max_length - len(seq), 0)
        sequence_padded.append(seq_)
        sequence_length.append(min(len(seq), max_length))

    return sequence_padded, sequence_length


def pad_sequences(sequences, pad_tok, nlevels=1):
    """
    Args:
        sequences: a generator of list or tuple
        pad_tok: the char to pad with
        nlevels: "depth" of padding, for the case where we have characters ids

    Returns:
        a list of list where each sublist has same length

    """
    sequence_padded, sequence_length = [], []
    if nlevels == 1:
        max_length = max(map(lambda x: len(x), sequences))
        sequence_padded, sequence_length = _pad_sequences(sequences,
                                                          pad_tok, max_length)

    elif nlevels == 2:
        max_length_word = max([max(map(lambda x: len(x), seq))
                               for seq in sequences])
        for seq in sequences:
            # all words are same length now
            sp, sl = _pad_sequences(seq, pad_tok, max_length_word)
            sequence_padded.append(sp)
            sequence_length.append(sl)

        max_length_sentence = max(map(lambda x: len(x), sequences))
        sequence_padded, _ = _pad_sequences(sequence_padded,
                                            [pad_tok] * max_length_word, max_length_sentence)
        sequence_length, _ = _pad_sequences(sequence_length, 0,
                                            max_length_sentence)

    return sequence_padded, sequence_length


def minibatches(data, minibatch_size):
    """
    Args:
        data: generator of (sentence, tags) tuples
        minibatch_size: (int)

    Yields:
        list of tuples

    """
    x_batch, y_batch = [], []
    for (x, y) in data:
        if len(x_batch) == minibatch_size:
            yield x_batch, y_batch
            x_batch, y_batch = [], []

        if type(x[0]) == tuple:
            x = zip(*x)
        x_batch.append(x)
        y_batch.append(y)

    if len(x_batch) != 0:
        yield x_batch, y_batch


def get_chunk_type(tok, idx_to_tag):
    """
    Args:
        tok: id of token, ex 4
        idx_to_tag: dictionary {4: "B-PER", ...}

    Returns:
        tuple: "B", "PER"

    """
    tag_name = idx_to_tag[tok]
    tag_class = tag_name.split('-')[0]
    tag_type = tag_name.split('-')[-1]
    return tag_class, tag_type


def get_chunks(seq, tags):
    """Given a sequence of tags, group entities and their position

    Args:
        seq: [4, 4, 0, 0, ...] sequence of labels
        tags: dict["O"] = 4

    Returns:
        list of (chunk_type, chunk_start, chunk_end)

    Example:
        seq = [4, 5, 0, 3]
        tags = {"B-PER": 4, "I-PER": 5, "B-LOC": 3}
        result = [("PER", 0, 2), ("LOC", 3, 4)]

    """
    default = tags[NONE]
    idx_to_tag = {idx: tag for tag, idx in tags.items()}
    chunks = []
    chunk_type, chunk_start = None, None
    for i, tok in enumerate(seq):
        # End of a chunk 1
        if tok == default and chunk_type:
            # Add a chunk.
            chunk = (chunk_type, chunk_start, i)
            chunks.append(chunk)
            chunk_type, chunk_start = None, None

        # End of a chunk + start of a chunk!
        elif tok != default:
            tok_chunk_class, tok_chunk_type = get_chunk_type(tok, idx_to_tag)
            if chunk_type is None:
                chunk_type, chunk_start = tok_chunk_type, i
            elif tok_chunk_type != chunk_type or tok_chunk_class == "B":
                chunk = (chunk_type, chunk_start, i)
                chunks.append(chunk)
                chunk_type, chunk_start = tok_chunk_type, i
        else:
            pass

    # end condition
    if chunk_type is not None:
        chunk = (chunk_type, chunk_start, len(seq))
        chunks.append(chunk)

    return chunks


def reform_training_data(path, suffix, save_name,
                         h_label="c", m_label="m", sep=" "):
    result = []
    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        if os.path.splitext(file_path)[-1] == suffix:
            records = []
            with open(file_path, "r") as fin:
                for row in fin:
                    records = []
                    tokens = _segment(row)
                    for ti in tokens:
                        matcher = tag_r.search(ti)
                        if matcher:
                            ti = matcher.group(1).strip()
                            label = h_label
                        else:
                            label = m_label
                        for tii in ti.split(" "):
                            records.append((tii, label))
                    result.append(records)
            if records:
                result.append(records)
    with open(save_name, "w", encoding="utf-8") as fout:
        for records1 in result:
            for row in records1:
                fout.write(sep.join(row))
                fout.write("\n")
            fout.write("\n")


def _segment(string):
    tokens = []
    string = re.sub("]\[", "] \[", string)
    string = re.sub("\s+", " ", string).strip()
    chars = list(string)
    tmp_chars = []
    left_b = False
    for ci in chars:
        if ci == "[":
            left_b = True
        if ci == "]":
            left_b = False
        if ci == " " and not left_b:
            tokens.append("".join(tmp_chars).strip())
            tmp_chars.clear()
        else:
            tmp_chars.append(ci)
    if tmp_chars:
        tokens.append("".join(tmp_chars))
    return tokens


def test_outcome(out_file="data/out.txt",
                 save_file="data/test_out.csv",
                 h_label="c"):
    records = []
    count = 0
    with open(out_file, "r", encoding="utf8") as fin:
        tmp = ""
        true_head = ""
        predict_head = ""
        for row in fin:
            if row == "\n":
                if true_head == predict_head:
                    flag = 1
                    count += 1
                else:
                    flag = 0
                records.append((tmp.strip(), true_head.strip(), predict_head, flag))
                tmp = ""
                true_head = ""
                predict_head = ""
            else:
                line = str(row).strip().split()
                word = line[0]
                true_label = line[-2]
                predict_label = line[-1]
                tmp += " " + word
                if true_label == h_label:
                    true_head += " " + word
                if predict_label == h_label:
                    predict_head += " " + word
    print("准确率: {}".format(count/len(records)))
    df = pd.DataFrame(records)
    df.columns = ["query", "true_head", "predict_head", "if_true"]
    df.to_csv(save_file)
