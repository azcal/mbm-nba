#!/usr/bin/env python3
"""
Flag names a function reads but never binds.

Written because a patch applied the same text to two runners whose `run()` signatures
differed, so NBA ended up calling `rosters_and_outs(datestr)` in a function whose argument
is `today_str`. `ast.parse` and `py_compile` both pass that: it is valid Python that raises
NameError only when the line executes, which for a seasonal board can be months later.

Deliberately conservative. Every name bound anywhere in the function counts as bound,
including inside comprehensions that really have their own scope, so this under-reports
rather than crying wolf. It exists to catch a name that appears nowhere at all, which is
what a bad find-and-replace produces.

Usage: python check_names.py <file.py> [file.py ...]
"""
import ast
import builtins
import sys


def bound_names(node):
    out = set()
    for n in ast.walk(node):
        if isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Store, ast.Del)):
            out.add(n.id)
        elif isinstance(n, ast.arg):
            out.add(n.arg)
        elif isinstance(n, (ast.Import, ast.ImportFrom)):
            for a in n.names:
                out.add((a.asname or a.name).split(".")[0])
        elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            out.add(n.name)
        elif isinstance(n, ast.ExceptHandler) and n.name:
            out.add(n.name)
        elif isinstance(n, ast.Global) or isinstance(n, ast.Nonlocal):
            out.update(n.names)
    return out


def check(path):
    tree = ast.parse(open(path).read(), path)
    module = bound_names(tree) | set(dir(builtins))
    bad = []
    for fn in [n for n in ast.walk(tree)
               if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]:
        scope = module | bound_names(fn)
        for n in ast.walk(fn):
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load) \
                    and n.id not in scope:
                bad.append((fn.name, n.lineno, n.id))
    return bad


if __name__ == "__main__":
    fails = 0
    for path in sys.argv[1:]:
        bad = check(path)
        if bad:
            fails += 1
            print(f"{path}")
            for fn, line, name in bad:
                print(f"   line {line}: {fn}() reads '{name}', which is never bound in it")
        else:
            print(f"{path}  ok")
    sys.exit(1 if fails else 0)
