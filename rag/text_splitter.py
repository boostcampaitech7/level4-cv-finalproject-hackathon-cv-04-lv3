import tiktoken

tokenizer = tiktoken.get_encoding("cl100k_base")

def calculate_token(text):
    tokens = tokenizer.encode(text)
    return len(tokens)