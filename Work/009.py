import spacy
from collections import Counter
from gensim.models import Word2Vec
from gensim.models.word2vec import Text8Corpus
import logging

# 加载英文模型
nlp = spacy.load('en_core_web_sm')

# 待处理的文本
text = "I love machine learning and natural language processing."

# 使用SpaCy的tokenization功能提取单词
words = [token.text for doc in nlp.pipe(text) for token in doc]
print(words)
# 统计词频
word_counts = Counter(words)
print(word_counts)
print(word_counts.most_common())
print([(idx, word, count) for idx, (word, count) in enumerate(word_counts.most_common())])
# 构建词汇表，并为每个单词分配一个索引
vocab = {word: idx for idx, (word, count) in enumerate(word_counts.most_common())}

print(vocab)

# 设置日志级别，用于观察训练过程
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


# 假设我们有一个大型的文本语料库 corpus，可以使用 Text8Corpus 来获取
# corpus = Text8Corpus('text8.zip')

# 训练词向量模型
# model = Word2Vec(corpus, vector_size=100, window=5, min_count=1, workers=4)

# 使用词汇表训练词向量模型
sentences = [[word for word in vocab.keys()]]
print(sentences)
model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)

# 保存词向量模型
model.save('word2vec.model')

# 查看词向量
print(model.wv['l'])

# 计算两个词向量的余弦相似度
print(model.wv.similarity('l', 'm'))

# 找出最相似的词
print(model.wv.most_similar('l'))





