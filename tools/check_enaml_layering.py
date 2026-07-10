'''
Enforce the psi package layering for .enaml files.

import-linter (configured in pyproject.toml) enforces the layer order for
Python modules, but it cannot parse .enaml files. This script applies the
same contract to the `from psi.X import ...` statements found in .enaml
sources.

Layer order (low to high): util, core, context, token, controller, data,
experiment, application. A module may only import from layers at or below
its own.

Exceptions (feature packages that sit above the data layer even though they
live inside psi.controller):
- psi.controller.calibration: orchestrates outputs *and* stores/plots
  results, so it may import psi.data.
- psi.controller.engines: optional per-engine plot add-ons may import
  psi.data.
- psi.paradigms and psi.templates: user-facing composition layers; excluded
  entirely.
'''
import re
import sys
from pathlib import Path

LAYERS = ['util', 'core', 'context', 'token', 'controller', 'data',
          'experiment', 'application']

# Packages excluded from checking entirely.
EXCLUDED = ('psi/paradigms/', 'psi/templates/')

# Source prefixes whose effective layer is raised above their location.
FEATURE_LAYERS = {
    'psi/controller/calibration/': LAYERS.index('data'),
    'psi/controller/engines/': LAYERS.index('data'),
}

IMPORT_RE = re.compile(r'^\s*from\s+(psi(?:\.[a-zA-Z_][a-zA-Z0-9_]*)*)\s+import|^\s*import\s+(psi(?:\.[a-zA-Z_][a-zA-Z0-9_]*)*)')


def module_layer(module):
    parts = module.split('.')
    if len(parts) < 2:
        # Bare `psi` (get_config etc.) sits at the bottom.
        return 0
    try:
        return LAYERS.index(parts[1])
    except ValueError:
        # psi.util is a module; psi.version etc. -> bottom.
        return 0


def source_layer(path):
    posix = path.as_posix()
    for prefix, layer in FEATURE_LAYERS.items():
        if posix.startswith(prefix):
            return layer
    parts = path.parts
    try:
        return LAYERS.index(parts[1])
    except (IndexError, ValueError):
        return len(LAYERS) - 1


def main():
    root = Path(__file__).parent.parent
    errors = []
    for path in sorted((root / 'psi').rglob('*.enaml')):
        rel = path.relative_to(root)
        posix = rel.as_posix()
        if posix.startswith(EXCLUDED):
            continue
        src_layer = source_layer(rel)
        for lineno, line in enumerate(path.read_text(encoding='utf-8').splitlines(), 1):
            m = IMPORT_RE.match(line)
            if not m:
                continue
            module = m.group(1) or m.group(2)
            if module_layer(module) > src_layer:
                errors.append(
                    f'{posix}:{lineno}: {LAYERS[src_layer]}-layer module '
                    f'imports {module}'
                )

    if errors:
        print('Layering violations in .enaml files:')
        for error in errors:
            print(f'  {error}')
        return 1
    print('No layering violations in .enaml files.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
