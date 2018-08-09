import json
import traceback


def make_dict(*expr):
    (filename, line_number, function_name, text) = traceback.extract_stack()[-2]
    begin = text.find('make_dict(') + len('make_dict(')
    end = text.find(')', begin)
    text = [name.strip() for name in text[begin:end].split(',')]
    return dict(zip(text, expr))


def json_load(path):
    with open(path, "r", encoding="utf8") as f:
        content = json.load(f)
    return content


def write(path, content):
    with open(path, "w", encoding="utf8") as f:
        f.write(content)


def write_lines(path, lines):
    with open(path, "w", encoding="utf8") as f:
        f.write("\n".join(lines))
