import re

def extract_curse_words(text):
    # [숫자, 숫자, '문자열'] 형식을 찾는 패턴
    pattern = r'\[(\d+),\s*(\d+),\s*\'([^\']+)\'\]'
    matches = re.findall(pattern, text)
    
    # 찾은 패턴을 리스트로 변환
    curse_words = []
    for match in matches:
        start, end, word = match
        curse_words.append([int(start), int(end), word])
    
    return curse_words