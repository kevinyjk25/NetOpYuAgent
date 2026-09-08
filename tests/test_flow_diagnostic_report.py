import copy
import json

import pytest

from evaluation.flow_diagnostic_report import cell, main, render
from evaluation.flow_diagnostics import diagnose, signed
from tests.test_flow_diagnostics import inputs


def test_readable_report_keeps_chinese_first_and_does_not_claim_accuracy():
    source, tree, mapping = inputs()
    mapping['selections']['clause-0001'].pop(1)
    report = diagnose(source, tree, mapping)
    text = render(report)
    assert text.index('## 中文') < text.index('## English')
    assert '/steps/1' in text and '/nodes/1' in text
    assert '已有节点缺来源证据' in text and '未评估 ≠ 0%' in text


def test_source_content_is_escaped_not_executed_html():
    assert '<script>' not in cell('<script>alert(1)</script>')
    assert '\\|' in cell('a|b')
    assert '\\[x\\]' in cell('[x](https://example.test)')


@pytest.mark.parametrize('nested', [False, True])
def test_invalid_digest_is_rejected_even_inside_signed_batch(nested):
    r = diagnose(*inputs())
    r['semanticAccuracy'] = 1.0
    if nested:
        r = signed(dict(protocol=r['protocol'], cases=[dict(case='one', diagnosis=r)]))
    with pytest.raises(ValueError, match='digest'):
        render(r)


def test_render_reproducible_read_only_and_no_overwrite(tmp_path):
    r = diagnose(*inputs())
    before = copy.deepcopy(r)
    src = tmp_path / 'report.json'
    src.write_text(json.dumps(r))
    out = tmp_path / 'report.md'
    assert main([str(src), '--output', str(out)]) == 0
    assert out.read_text() == render(r) and r == before
    with pytest.raises(FileExistsError):
        main([str(src), '--output', str(out)])
