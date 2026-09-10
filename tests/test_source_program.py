"""Parse-only source planning: no execution, inferred success or policy bypass."""
import pytest

from evaluation.source_program import parse

END = 'end("read_path_completed", "b0001", "Source-defined inspection completed")'
READ = 'observed = read("tool", "b0001", "Original observation required")'


@pytest.mark.parametrize("prefix", [
    "import os", "os.system('echo should-not-run')", "__import__('os').system('true')",
    "for item in []:\n    pass", "while True:\n    pass", "def f():\n    pass",
    "x = lambda: 1", "x = [v for v in []]", "with open('x') as f:\n    pass",
    "read = 1", "a = b = read('tool', 'b0001', 'some reason')", "x = read(*[])" ,
    "x = read('tool', 'b0001', 'some reason', **{})", "input = read('tool', 'b0001', 'some reason')",
    "x = read('tool', 'b0001', f'some reason')", "x = read.__call__('tool', 'b0001', 'some reason')",
])
def test_python_surface_outside_whitelist_is_rejected(prefix):
    with pytest.raises(ValueError):
        parse(prefix + "\n" + END)


@pytest.mark.parametrize("test", ["field(observed, '/x', 'b0001') > 0", "observed.x == 0",
    "observed['x'] == 0", "field(observed, '/x', 'b0001') == 1e999",
    "field(observed, '/x', 'b0001') == 0 == 0", "True or False"])
def test_general_predicates_are_not_silently_interpreted(test):
    with pytest.raises(ValueError):
        parse(READ + f"\nif {test}:\n    {END}\nelse:\n    {END}")


def test_branch_fallthrough_is_not_a_success_or_a_scope_merge():
    result = parse(READ + f"\nif field(observed, '/x', 'b0001') == True:\n    {END}\n{END}")
    assert result["steps"][1]["otherwise"] == {"steps": [], "exit": {"kind": "continue"}}
    assert result["exit"]["outcome"] == "read_path_completed"
    with pytest.raises(ValueError, match="all root paths"):
        parse(READ + f"\nif field(observed, '/x', 'b0001') == True:\n    {END}")
    with pytest.raises(ValueError, match="unreachable"):
        parse(END + "\n" + READ)


def test_no_empty_handoff_or_completed_duty_and_no_implicit_end():
    for program in [READ, 'end("needs_l1", "b0001", "Human reasoning remains")',
                    'end("read_path_completed", "b0001", "A description", [("b0001", "After observation", "Still needs explanation")])']:
        with pytest.raises(ValueError):
            parse(program)


def test_explicit_modes_and_source_operation_are_preserved_not_inferred():
    result = parse(READ[:-1] + ', operation_mode="selected", source_operation="original")\n' + END)
    assert result["steps"][0]["operationMode"] == "selected"
    assert result["steps"][0]["sourceOperation"] == "original"


@pytest.mark.parametrize("literal,expected", [("-1", -1), ("+2.5", 2.5), ("None", None), ("False", False)])
def test_scalar_literals_preserve_value_and_type(literal, expected):
    result = parse(READ + f"\nif field(observed, '/x', 'b0001') == {literal}:\n    {END}\nelse:\n    {END}")
    assert result["steps"][1]["equals"] == expected
    assert type(result["steps"][1]["equals"]) is type(expected)


def test_bounded_statements_and_sources_remain_literal():
    with pytest.raises(ValueError, match="budget"):
        parse("\n".join(READ.replace("observed", f"obs_{i}") for i in range(33)) + "\n" + END)
    with pytest.raises(ValueError):
        parse('obs = read("tool", dynamic_source, "Some read reason")\n' + END)


def test_field_aliases_are_composed_references_not_duplicate_reads():
    from evaluation.structured_flow_demo import object_schema
    catalog = {"tools": [{"name": "tool", "outputSchema": object_schema({"nested": object_schema({"flag": {"type": "boolean"}})})}]}
    program = (READ.replace("observed", "OBS") + '\nPART = field(OBS, "/nested", "b0001")'
               + '\nFLAG = field(PART, "/flag", "b0001")' + f'\nif FLAG == True:\n    {END}\nelse:\n    {END}')
    result = parse(program, input_schema=object_schema({}), catalog=catalog)
    assert len(result["steps"]) == 2
    assert result["steps"][1]["left"] == {"kind": "reference", "source": "OBS", "pointer": "/nested/flag"}
    assert len(result["fieldBindings"]) == 2
    assert result["fieldBindings"][1]["expression"] == result["steps"][1]["left"]
    # Even an unused alias must reference a real declared field; relative path
    # fragments cannot be concatenated into a different, accidentally valid key.
    for bad in [program.replace('"/flag"', '"flag"'), program.replace('"/flag"', '"/unknown"')]:
        with pytest.raises(ValueError):
            parse(bad, input_schema=object_schema({}), catalog=catalog)


def test_field_alias_cannot_leak_from_a_sibling_or_be_overwritten():
    program = READ + f'\nif field(observed, "/flag", "b0001") == True:\n    local = field(observed, "/flag", "b0001")\n    {END}\nelse:\n    if local == True:\n        {END}\n    else:\n        {END}'
    with pytest.raises(ValueError, match="preceding field"):
        parse(program)
    with pytest.raises(ValueError, match="unique"):
        parse(READ + '\nobserved = field(observed, "/flag", "b0001")\n' + END)
