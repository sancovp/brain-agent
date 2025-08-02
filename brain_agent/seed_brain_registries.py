#!/usr/bin/env python3
"""Seed brain_personas_registry and brain_modes_registry with default MVP values."""

from heaven_base.tools.registry_tool import registry_util_func

# ---------------- Personas -----------------
PERSONAS = [
    {
        "id": "logical_philosopher",
        "name": "Logical Philosopher",
        "description": "Responds with rigorous logical analysis, like a seasoned analytic philosopher.",
        "prompt_block": (
            "You are a seasoned analytic philosopher. Evaluate every claim with meticulous logic, "
            "explicitly state premises, derive conclusions carefully, and avoid rhetorical flourish."
        ),
    },
    {
        "id": "senior_scientist",
        "name": "Senior Scientist",
        "description": "Speaks as an experienced research scientist – methodical, evidence-driven, cautious with claims.",
        "prompt_block": (
            "You are a senior research scientist. Approach problems with experimental rigor, reference data when "
            "possible, articulate hypotheses and limitations clearly."
        ),
    },
    {
        "id": "senior_engineer",
        "name": "Senior Engineer",
        "description": "Acts as a pragmatic senior software engineer focused on real-world implementation.",
        "prompt_block": (
            "You are a senior software engineer. Provide concrete implementation guidance, highlight edge cases, "
            "and balance trade-offs pragmatically."
        ),
    },
]

# ---------------- Modes -----------------
MODES = [
    {
        "id": "summarize",
        "name": "Summarize",
        "description": "Comprehensively summarize in granular detail how the neuron content relates to the query.",
        "prompt_block": (
            "Task: Provide a clear, structured summary. Separate ‘Neuron Content’ from ‘Query’ and then explain, "
            "point-by-point, how the former relates to the latter."
        ),
    },
    {
        "id": "imagine",
        "name": "Imagine",
        "description": "The query describes an imaginary idea. Imagine how the neuron content could relate to it.",
        "prompt_block": (
            "Task: Think creatively. Describe potential connections, synergies, or inspirations between the neuron "
            "content and the imagined scenario in the query."
        ),
    },
    {
        "id": "reify",
        "name": "Reify",
        "description": "The query proposes a concrete idea. Detail how the neuron content can be used to make it real.",
        "prompt_block": (
            "Task: Provide actionable steps and considerations to turn the query’s idea into reality using insights "
            "from the neuron content."
        ),
    },
]


def safe_add(registry_name: str, key: str, value_dict: dict):
    """Add entry if key not already present; otherwise update."""
    existing = registry_util_func(operation="get", registry_name=registry_name, key=key)
    if "not found" not in existing:
        # update
        registry_util_func(
            operation="update",
            registry_name=registry_name,
            key=key,
            value_dict=value_dict,
        )
        print(f"Updated {key} in {registry_name}")
    else:
        registry_util_func(
            operation="add",
            registry_name=registry_name,
            key=key,
            value_dict=value_dict,
        )
        print(f"Added {key} to {registry_name}")


def main():
    for persona in PERSONAS:
        safe_add("brain_personas_registry", persona["id"], persona)

    for mode in MODES:
        safe_add("brain_modes_registry", mode["id"], mode)

    print("\nSeeding complete. Current registry keys:")
    persons = registry_util_func(operation="list_keys", registry_name="brain_personas_registry")
    modes = registry_util_func(operation="list_keys", registry_name="brain_modes_registry")
    print(f"Personas: {persons}")
    print(f"Modes: {modes}")


if __name__ == "__main__":
    main()
