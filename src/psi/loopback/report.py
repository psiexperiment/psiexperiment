'''
Results of a run, written as an HTML page of checks and plots for viewing by
hand, plus an index of every run in the session.
'''
from dataclasses import dataclass
import html
import json
from pathlib import Path


@dataclass
class Check:
    name: str
    passed: bool
    #: What was measured, formatted for reading.
    value: str
    #: What it had to be.
    limit: str = ''
    detail: str = ''


class Checks(dict):
    '''
    Checks by name, so each test can assert the ones it covers.
    '''
    def add(self, name, passed, value, limit='', detail=''):
        self[name] = Check(name, bool(passed), value, limit, detail)

    def assert_passed(self, *names):
        failed = [self[n] for n in names if not self[n].passed]
        if failed:
            lines = [f'{c.name}: {c.value} (expected {c.limit}) {c.detail}'
                     for c in failed]
            raise AssertionError('\n'.join(lines))


STYLE = '''
body { font-family: system-ui, sans-serif; margin: 16px; color: #222; }
table { border-collapse: collapse; margin-bottom: 16px; }
td, th { border: 1px solid #ccc; padding: 4px 8px; text-align: left;
         vertical-align: top; }
.pass { color: #1a7f37; } .fail { color: #cf222e; font-weight: bold; }
img { max-width: 100%; border: 1px solid #ddd; margin: 8px 0; }
'''


def write_report(run_dir, title, checks, figures, info=None):
    '''
    Write `report.html` (and `checks.json`) into `run_dir`.

    `figures` are paths of images, relative to `run_dir`.
    '''
    run_dir = Path(run_dir)
    rows = []
    for c in checks.values():
        cls = 'pass' if c.passed else 'fail'
        rows.append(
            f'<tr><td>{html.escape(c.name)}</td>'
            f'<td class="{cls}">{"pass" if c.passed else "FAIL"}</td>'
            f'<td>{html.escape(c.value)}</td><td>{html.escape(c.limit)}</td>'
            f'<td>{html.escape(c.detail)}</td></tr>')
    info_rows = ''.join(f'<tr><th>{html.escape(str(k))}</th>'
                        f'<td>{html.escape(str(v))}</td></tr>'
                        for k, v in (info or {}).items())
    images = ''.join(f'<img src="{html.escape(str(f))}">' for f in figures)
    page = f'''<!doctype html><html><head><meta charset="utf-8">
<title>{html.escape(title)}</title><style>{STYLE}</style></head><body>
<p><a href="../index.html">All runs</a></p>
<h1>{html.escape(title)}</h1>
<table>{info_rows}</table>
<table><tr><th>Check</th><th></th><th>Measured</th><th>Expected</th>
<th>Detail</th></tr>{''.join(rows)}</table>
{images}
</body></html>'''
    (run_dir / 'report.html').write_text(page, encoding='utf-8')
    summary = {'title': title,
               'passed': all(c.passed for c in checks.values()),
               'n_failed': sum(not c.passed for c in checks.values()),
               'checks': [vars(c) for c in checks.values()]}
    (run_dir / 'checks.json').write_text(json.dumps(summary, indent=2))
    write_index(run_dir.parent)


def write_index(results_dir):
    '''
    List every run under `results_dir` that has a report.
    '''
    results_dir = Path(results_dir)
    rows = []
    for summary_file in sorted(results_dir.glob('*/checks.json')):
        s = json.loads(summary_file.read_text())
        name = summary_file.parent.name
        status = '<span class="pass">pass</span>' if s['passed'] else \
            f'<span class="fail">{s["n_failed"]} failed</span>'
        rows.append(f'<tr><td><a href="{html.escape(name)}/report.html">'
                    f'{html.escape(s["title"])}</a></td><td>{status}</td></tr>')
    extra = ''
    if (results_dir / 'routing.png').exists():
        extra = '<h2>Routing</h2><img src="routing.png">'
    page = f'''<!doctype html><html><head><meta charset="utf-8">
<title>Loopback tests</title><style>{STYLE}</style></head><body>
<h1>Loopback tests</h1>
<p>{html.escape(str(results_dir))}</p>
<table><tr><th>Run</th><th>Result</th></tr>{''.join(rows)}</table>
{extra}
</body></html>'''
    (results_dir / 'index.html').write_text(page, encoding='utf-8')
