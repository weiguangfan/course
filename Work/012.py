from snownlp import SnowNLP
import snownlp

print(snownlp.__version__)

# # 示例文本
# text = "SnowNLP是一个中文分词、词性标注、命名实体识别等功能的自然语言处理库。"
#
# # 初始化SnowNLP对象
# snownlp = SnowNLP(text)
#
# # 分词
# segments = snownlp.cut()
# print("分词结果：", segments)
#
# # 命名实体识别
# ners = snownlp.ner()
# print("命名实体识别结果：", ners)
#
# # 词性标注
# postags = snownlp.pos_tag()
# print("词性标注结果：", postags)
