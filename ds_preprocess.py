import re

def clean_code(code: str, language: str) -> str:
    # 1. usuń komentarze jednoliniowe (//, #)
    if language == "python":
        code = re.sub(r"#.*?$", " ", code, flags=re.MULTILINE)
    elif language in ["c++", "c", "go", "java", "c#", "verilog"]:
        code = re.sub(r"//.*?$", " ", code, flags=re.MULTILINE)
    elif language == "prolog":
        code = re.sub(r"%.*?$", " ", code, flags=re.MULTILINE)
    elif language == "assembly":
        code = re.sub(r";.*?$", " ", code, flags=re.MULTILINE)

    # 2. usuń komentarze blokowe /* ... */
    code = re.sub(r"/\*.*?\*/", " ", code, flags=re.DOTALL)
    
    if language == "python":
        code = re.sub(r'"""(.*?)"""', " ", code, flags=re.DOTALL)
        code = re.sub(r"'''(.*?)'''", " ", code, flags=re.DOTALL)

    # 3. usuń wielokrotne spacje
    code = re.sub(r"\s+", " ", code).strip()


    return code