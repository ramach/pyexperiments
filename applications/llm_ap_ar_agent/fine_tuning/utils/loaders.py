import json

def load_invoice(file):
    return json.load(file)

def load_rules(file):
    return file.read()
