from transformers import pipeline


ner = pipeline('ner', grouped_entities=True)
# print(ner('My name is Sylvain and I work at Hugging Face in Brooklyn.'))
print(ner('My name is Sylvain and my girl-friend is Sunny.I work at Hugging Face in Brooklyn. She work at HuaWei in Shanghai'))
