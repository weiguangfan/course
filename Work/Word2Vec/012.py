import jieba
import thulac
import gensim
from gensim import corpora
import re


# import jieba
#
#
# # 读取停用词词典文件
# with open('stopwords-zh.txt', 'r', encoding='utf-8') as f:
#     stopwords = f.read().splitlines()
#
# # 加载停用词词典到jieba分词器
# jieba.load_userdict(stopwords)
#
# # 示例文本
# text = "这是一个中文文本示例，用于演示文本处理过程。"
#
# # 使用jieba分词器进行分词，并去除停用词
# seg_list = jieba.cut(text, cut_all=False)
# filtered_words = [word for word in seg_list if word not in stopwords]
#
# # 输出结果
# print(filtered_words)







# 初始化THUOCL关键词提取器
thulac_ins = thulac.thulac(seg_only=True)
# 读取停用词词典文件
with open('../stopwords-zh.txt', 'r', encoding='utf-8') as f:
    stopwords = f.read().splitlines()

# 加载停用词词典到jieba分词器
jieba.load_userdict(stopwords)

# # 初始化jieba分词器
# jieba.load_userdict("stopwords-zh.txt")  # 可加载自定义词典

# 初始化LDA模型，需要预先准备好语料库
texts = ["中文文本1", "中文文本2", "中文文本3"]  # 示例文本
dictionary = corpora.Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]
ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=5, id2word=dictionary, passes=10)

# 文本清洗函数
def clean_text(text):
    text = re.sub(r'\s+', '', text)  # 删除空白字符
    text = re.sub(r'\t+', '', text)  # 删除制表符
    text = re.sub(r'\n+', '', text)  # 删除换行符
    text = re.sub(r'[a-zA-Z]', '', text)  # 删除英文字符
    text = re.sub(r'[^0-9a-zA-Z\u4e00-\u9fff]', '', text)  # 删除非法字符
    return text

# 示例文本
text = "这是一个中文文本示例，用于演示文本处理过程。"

# 文本清洗
cleaned_text = clean_text(text)

# 中文分词
seg_list = jieba.cut(cleaned_text, cut_all=False)
seg_result = " ".join(seg_list)

# 提取关键词
key_words = thulac_ins.get_keywords(seg_result, num=5)  # 提取前5个关键词

# 提取主题词
topic_words = ldamodel.print_topics()

# 输出结果
print("清洗后的文本：", cleaned_text)
print("分词结果：", seg_result)
print("关键词：", key_words)
print("主题词：", topic_words)
