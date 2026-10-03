import inspect
import tools_reduction


def test_diagnose_import_source():
    print(f"\n[DIAGNOSE] tools_reduction.__file__: {tools_reduction.__file__}")
    print(f"[DIAGNOSE] tools_reduction.__cached__: {getattr(tools_reduction, '__cached__', None)}")
    
    src = inspect.getsource(tools_reduction.make_tools)
    has_init_random = 'init="random"' in src or "init='random'" in src
    has_lr_200 = 'learning_rate=200.0' in src or 'learning_rate=200' in src
    
    print(f"[DIAGNOSE] source has init='random': {has_init_random}")
    print(f"[DIAGNOSE] source has learning_rate=200.0: {has_lr_200}")
    print("\n[DIAGNOSE] inspect.getsource(tools_reduction.make_tools):\n" + src)
