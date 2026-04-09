import ast
p='main27_train_final.py'
s=open(p,'r').read()
try:
    ast.parse(s)
    print('OK')
except SyntaxError as e:
    print('SyntaxError', e.lineno, e.offset, e.msg)
    lines=s.splitlines()
    for i in range(max(0,e.lineno-6), min(len(lines), e.lineno+5)):
        print(i+1, lines[i])

# report Try nodes
try:
    tree=ast.parse(s)
    for node in ast.walk(tree):
        if isinstance(node, ast.Try):
            print('Try at', node.lineno, 'handlers', len(node.handlers), 'orelse', len(node.orelse), 'final', len(node.finalbody))
except Exception:
    pass
