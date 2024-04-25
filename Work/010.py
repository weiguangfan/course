from transformers import BertTokenizer

# 初始化分词器
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# 输入文本，包含两个人各说3句话的对话
text = "Person 1: Hello, how are you? I hope you are doing well. What's the weather like in your city?\nPerson 2: I'm good, thanks! The weather here is sunny. How about in your city?"

# 分词
tokens = tokenizer.tokenize(text)
print(f"分词结果：{tokens}")

# 转换为ID
input_ids = tokenizer.convert_tokens_to_ids(tokens)
print(f"转换为ID后的结果：{input_ids}")

# 添加特殊符号
tokens = tokenizer.build_inputs_with_special_tokens(tokens)
print(f"添加特殊符号后的结果：{tokens}")

# 生成类型ID，这里区分句子A和句子B
token_type_ids = tokenizer.create_token_type_ids_from_sequences(tokens)
print(f"生成类型ID后的结果：{token_type_ids}")

# # 生成注意力掩码
# attention_mask = tokenizer.get_special_tokens_mask(tokens)
# print(f"生成注意力掩码后的结果：{attention_mask}")
