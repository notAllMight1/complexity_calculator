# ml/feature_engineering.py

import re

# Define simple keyword lists for each language.
KEYWORDS = {
    "python": ["def", "for", "while", "if", "elif", "else", "import", "return"],
    "java": ["public", "class", "for", "while", "if", "else", "import", "return", "void"],
    "c++": ["int", "for", "while", "if", "else", "include", "return", "void", "std"]
}

def remove_comments(code, language):
    """
    Removes comments from code based on the programming language.
    """
    if language.lower() == "python":
        # Remove single-line comments (starting with #)
        code = re.sub(r'#.*', '', code)
        # Remove multi-line comments (between triple quotes)
        code = re.sub(r'""".*?"""', '', code, flags=re.DOTALL)
        code = re.sub(r"'''.*?'''", '', code, flags=re.DOTALL)
    elif language.lower() == "java" or language.lower() == "c++":
        # Remove single-line comments (starting with //)
        code = re.sub(r'//.*', '', code)
        # Remove multi-line comments (starting with /* and ending with */)
        code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)
    
    return code

def tokenize_code(code):
    """
    Tokenize the code snippet by extracting word characters.
    This works decently for Python, Java, and C++.
    """
    tokens = re.findall(r'\w+', code)
    return tokens

def count_keywords(code, keywords):
    """
    Count the occurrences of each keyword from the provided list in the code snippet.
    """
    count = 0
    for kw in keywords:
        count += len(re.findall(r'\b' + re.escape(kw.lower()) + r'\b', code.lower()))
    return count

def extract_features(code, language):
    """
    Extract features from the code snippet:
    - Token count and code length (in terms of lines)
    - Keyword counts specific to the given language
    - Common loop and conditional constructs counts
    """
    # Remove comments before processing
    code = remove_comments(code, language)
    
    # Normalize code to lowercase for keyword matching.
    code_lower = code.lower()
    tokens = tokenize_code(code_lower)
    
    # Basic features.
    features = {
        "token_count": len(tokens),
        "code_length": len(code.splitlines())
    }
    
    # Add language-specific keyword count.
    lang = language.lower()
    if lang in KEYWORDS:
        features[f"{lang}_keyword_count"] = count_keywords(code_lower, KEYWORDS[lang])
    else:
        features["default_keyword_count"] = 0

    # Count common constructs across languages.
    features["for_count"] = len(re.findall(r'\bfor\b', code_lower))
    features["while_count"] = len(re.findall(r'\bwhile\b', code_lower))
    features["if_count"] = len(re.findall(r'\bif\b', code_lower))
    
    return features
