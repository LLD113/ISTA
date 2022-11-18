import importlib, pkgutil
# 工具文件，帮助更好的import相应包

# https://packaging.python.org/guides/creating-and-discovering-plugins/
# 用于下面的load_submodules函数，用于声明命名空间，命名文件
def iter_namespace(pkg_or_path, prefix='', **kwds):
    # Specifying the second argument (prefix) to iter_modules makes the
    # returned name an absolute name instead of a relative one. This
    # allows import_module to work without having to do additional
    # modification to the name.
    is_path = pkg_or_path is None or isinstance(pkg_or_path, list)
    path, prefix = \
        (pkg_or_path, prefix) if is_path else \
            (pkg_or_path.__path__, prefix if prefix is not '' else pkg_or_path.__name__ + ".")
    return pkgutil.walk_packages(path, prefix, **kwds)

# 加载子组件，在plugin.py文件中用到
def load_submodules(root, modules=None, prefix='', **kwds):
    return {
        name: importlib.import_module(name)
        for _finder, name, _ispkg in iter_namespace(root, prefix=prefix, **kwds)
        if (modules is None or name in modules) and name.startswith(prefix)
    }
