import ast
import hashlib
import io
import re
import tokenize
from pathlib import Path


def source_audit(path):
    source = Path(path).read_text()
    tree = ast.parse(source)
    tokens = list(tokenize.generate_tokens(io.StringIO(source).readline))
    prefixes = re.compile('^[rubf]*', re.I)
    string_tokens = {tokenize.STRING, getattr(tokenize, 'FSTRING_START', -1), getattr(tokenize, 'TSTRING_START', -2)}
    triple = sum(t.type in string_tokens and prefixes.sub('', t.string).startswith((chr(34) * 3, chr(39) * 3)) for t in tokens)
    docs = sum(ast.get_docstring(node) is not None for node in ast.walk(tree) if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)))
    comments = sum(t.type == tokenize.COMMENT for t in tokens)
    inventory = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            value = node.value
            words = re.findall(r'[A-Za-z]+', value)
            narrative = len(words) >= 10 and bool(re.search(r'[.!?](?:\s|$)', value)) and not any(marker in value for marker in ['://', '=>', 'document.', '\\', 'import ', 'function('])
            inventory.append({'line': node.lineno, 'kind': 'string_component', 'value': value, 'narrative_candidate': narrative})
        elif isinstance(node, ast.JoinedStr):
            value = ''.join(part.value if isinstance(part, ast.Constant) and isinstance(part.value, str) else '{VALUE}' for part in node.values)
            words = re.findall(r'[A-Za-z]+', value)
            narrative = len(words) >= 10 and bool(re.search(r'[.!?](?:\s|$)', value)) and not any(marker in value for marker in ['://', '=>', 'document.', '\\', 'import ', 'function('])
            inventory.append({'line': node.lineno, 'kind': 'formatted_string', 'value': value, 'narrative_candidate': narrative})
    findings = [item for item in inventory if item['narrative_candidate']]
    result = {'path': str(path), 'comments': comments, 'docstrings': docs, 'triple_quoted_strings': triple, 'embedded_narrative_strings': len(findings), 'string_literal_inventory': inventory, 'sha256': hashlib.sha256(source.encode()).hexdigest()}
    assert comments == docs == triple == len(findings) == 0, 'SOURCE_AUDIT_FAILED'
    return result

