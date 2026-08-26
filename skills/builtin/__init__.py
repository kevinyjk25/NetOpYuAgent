"""
skills/builtin/ — common built-in skills.

As of the SKILL.md standardization (2026-06), built-in skills live as
Anthropic-standard SKILL.md folders directly under this package:

    skills/builtin/<name>/SKILL.md

They are loaded by skills.loader.SkillLoader (NOT via a Python dict). This
package no longer exports a SKILLS mapping.
"""
