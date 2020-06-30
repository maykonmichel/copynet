import itertools

import torch.optim as optim
from allennlp.data.dataset_readers.seq2seq import Seq2SeqDatasetReader
from allennlp.data.iterators import BucketIterator
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.data.tokenizers.character_tokenizer import CharacterTokenizer
from allennlp.data.tokenizers.word_tokenizer import WordTokenizer
from allennlp.data.vocabulary import Vocabulary
from allennlp.models.encoder_decoders.simple_seq2seq import SimpleSeq2Seq
from allennlp.modules.attention import DotProductAttention
from allennlp.modules.seq2seq_encoders import StackedSelfAttentionEncoder
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders import Embedding
from allennlp.predictors import Seq2SeqPredictor
from allennlp.training.trainer import Trainer

EN_EMBEDDING_DIM = 256
POR_EMBEDDING_DIM = 256
HIDDEN_DIM = 256
CUDA_DEVICE = -1


def main():
    reader = Seq2SeqDatasetReader(
        source_tokenizer=WordTokenizer(),
        target_tokenizer=CharacterTokenizer(),
        source_token_indexers={'tokens': SingleIdTokenIndexer()},
        target_token_indexers={'tokens': SingleIdTokenIndexer(namespace='target_tokens')})
    train_dataset = reader.read('tatoeba.eng_por.train.tsv')
    validation_dataset = reader.read('tatoeba.eng_por.dev.tsv')

    vocab = Vocabulary.from_instances(train_dataset + validation_dataset,
                                      min_count={'tokens': 3, 'target_tokens': 3})

    en_embedding = Embedding(num_embeddings=vocab.get_vocab_size('tokens'),
                             embedding_dim=EN_EMBEDDING_DIM)
    encoder = StackedSelfAttentionEncoder(input_dim=EN_EMBEDDING_DIM, hidden_dim=HIDDEN_DIM, projection_dim=128,
                                          feedforward_hidden_dim=128, num_layers=1, num_attention_heads=8)

    source_embedder = BasicTextFieldEmbedder({"tokens": en_embedding})

    attention = DotProductAttention()

    max_decoding_steps = 20
    model = SimpleSeq2Seq(vocab, source_embedder, encoder, max_decoding_steps,
                          target_embedding_dim=POR_EMBEDDING_DIM,
                          target_namespace='target_tokens',
                          attention=attention,
                          beam_size=8,
                          use_bleu=True)
    optimizer = optim.Adam(model.parameters())
    iterator = BucketIterator(batch_size=32, sorting_keys=[("source_tokens", "num_tokens")])

    iterator.index_with(vocab)

    trainer = Trainer(model=model,
                      optimizer=optimizer,
                      iterator=iterator,
                      train_dataset=train_dataset,
                      validation_dataset=validation_dataset,
                      num_epochs=1,
                      cuda_device=CUDA_DEVICE)

    for i in range(50):
        print('Epoch: {}'.format(i))
        trainer.train()

        predictor = Seq2SeqPredictor(model, reader)

        for instance in itertools.islice(validation_dataset, 10):
            print('SOURCE:', instance.fields['source_tokens'].tokens)
            print('GOLD:', instance.fields['target_tokens'].tokens)
            print('PRED:', predictor.predict_instance(instance)['predicted_tokens'])


if __name__ == '__main__':
    main()
