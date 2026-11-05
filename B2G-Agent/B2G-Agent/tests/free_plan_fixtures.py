"""Controlled textual plans for offline tests; never supplied to real runs."""
import json


def render_plan(plan, *, multiline=False):
    names = {a.action_id: a.action_type.value for a in plan.actions}
    separator = "\n" if multiline else "; "
    return "\n\n".join(separator.join([
        f"{i}. {a.action_type.value} using {a.tool}",
        "Depends on: " + (", ".join(names.get(d, d) for d in a.depends_on) or "none"),
        "Inputs: " + json.dumps(a.inputs), "Outputs: " + json.dumps(a.outputs),
    ]) for i, a in enumerate(plan.actions, 1))
