import torch
from transformers import BertTokenizer
from transformers.optimization import AdamW

from .modelling_highway_bert import DeeBertForSequenceClassification

def load_pretrained_model_tokenizer(base_model=None, deebert=True, base_tokenizer=None, device='cuda'):
    if device == 'cuda':
        assert torch.cuda.is_available()

    # Load pre-trained model (weights)
    if base_model is None:
        # Download from huggingface
        base_model = 'bert-base-uncased'
#    model = BertForNextSentencePrediction.from_pretrained(base_model)
    model = DeeBertForSequenceClassification.from_pretrained(base_model, cache_dir='./.cache') if deebert else BertForNextSentencePrediction.from_pretrained(base_model, cache_dir='./.cache')

    if base_tokenizer is None:
        # Download from huggingface
        tokenizer = BertTokenizer.from_pretrained(base_model, cache_dir='./.cache')
    else:
        # Load local vocab file
        tokenizer = BertTokenizer.from_pretrained(base_tokenizer, cache_dir='./.cache')
    model.to(device)
    return model, tokenizer


def init_optimizer(model, learning_rate, warmup_proportion, num_train_epochs,
                   batch_size):
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    # size of MB:2014 - 4179
    num_train_steps = 41579 / batch_size * num_train_epochs
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if n not in no_decay],
         'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if n in no_decay],
         'weight_decay_rate': 0.0}
    ]

    optimizer = AdamW(
        optimizer_grouped_parameters,
        lr=learning_rate
    )
                         # warmup=warmup_proportion,
                         # t_total=num_train_steps)

    return optimizer


def print_scores(scores, mode='test'):
    print('[{}] '.format(mode), end='')
    for sn, score in zip(scores[0], scores[1]):
        print('{}: {}'.format(sn, score), end=' ')
    print()


def save_checkpoint(epoch, model, tokenizer, scores, filename):
    print('Save PyTorch model to {}'.format(filename))
    state = {
        'epoch': epoch,
        'model': model,
        'tokenizer': tokenizer,
        'scores': scores,
    }
    torch.save(state, filename)


def load_checkpoint(filename, device='cpu'):
    print('Load PyTorch model from {}'.format(filename))
    state = torch.load(filename, map_location='cpu') if device == 'cpu' else torch.load(filename)
    return state['epoch'], state['model'], state['tokenizer'], state['scores']
