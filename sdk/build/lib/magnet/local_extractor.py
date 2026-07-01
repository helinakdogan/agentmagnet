"""
LocalExtractor
--------------
Rule-based category detector: assigns project memory categories without an LLM.
Used when MAGNET_OPENAI_KEY is not set.

Categories (priority order):
  tried_failed  — something that was attempted and didn't work
  watch_out     — warnings, things to be careful about
  decision      — explicit decisions or choices made
  convention    — team/project conventions and patterns
  goal          — what the project is working toward
  preference    — personal likes/dislikes (fallback)
"""

from __future__ import annotations

import re

# Each list is (pattern, category). First match wins.
_RULES: list[tuple[re.Pattern, str]] = []

def _r(pattern: str, cat: str) -> None:
    _RULES.append((re.compile(pattern, re.IGNORECASE), cat))

# tried_failed — check before decision (overlapping words like "tried")
_r(r"\btried\b.{0,50}\b(and|but)\b.{0,50}\b(fail|broke|broken|didn'?t|doesn'?t|not work)", "tried_failed")
_r(r"\b(we tried|we attempted|we used)\b.{0,80}\b(fail|broke|broken|didn'?t work|not work)", "tried_failed")
_r(r"\b(gave up on|stopped using|abandoned|dropped)\b", "tried_failed")
_r(r"\b(fail|broke|broken|didn'?t work|doesn'?t work|not working)\b.{0,40}\b(on|with|for|when)\b", "tried_failed")
_r(r"\b(css grid|flexbox|approach|solution).{0,50}\b(broke|broken|fail|issue|problem)\b", "tried_failed")

# watch_out
_r(r"\b(be careful|careful about|watch out|don'?t forget|never (do|change|modify|touch)|important[!:])", "watch_out")
_r(r"\b(warning|critical|dangerous|breaks if|breaks when|will break)\b", "watch_out")
_r(r"\balways remember\b", "watch_out")
_r(r"\b(auth|authentication|token|permission|security).{0,40}\b(break|issue|careful|problem)\b", "watch_out")

# decision
_r(r"\bdecided?\b", "decision")
_r(r"\b(chose|choosing|going with|went with|we('re| are| will) use)\b", "decision")
_r(r"\bswitch(ed|ing)?\b.{0,30}\bto\b", "decision")
_r(r"\b(using|adopted|picked)\b.{0,20}\binstead\b", "decision")
_r(r"\b(will use|are using|now using)\b.{0,30}\b(for|as|in)\b", "decision")

# convention
_r(r"\b(always use|we always|convention|pattern we|standard(ize)?|our approach)\b", "convention")
_r(r"\b(code style|naming convention|folder structure|project structure)\b", "convention")
_r(r"\bevery(thing| file| component)\b.{0,30}\b(must|should|needs to)\b", "convention")

# goal
_r(r"\b(working on|we('re| are) building|we('re| are) making|trying to)\b", "goal")
_r(r"\b(goal|objective|aim|purpose|mission)\b.{0,20}\b(is|to)\b", "goal")
_r(r"\bproject\b.{0,20}\b(is|to|about|for)\b", "goal")

# preference (fallback)
_r(r"\bprefer\b", "preference")
_r(r"\b(like|love|enjoy|want|wish)\b.{0,20}\b(to |the |it )", "preference")
_r(r"\b(don'?t like|dislike|hate|avoid|don'?t want)\b", "preference")


def detect_category(text: str) -> str:
    """
    Classify text into a project memory category.
    Returns one of: decision | watch_out | tried_failed | convention | goal | preference
    """
    for pattern, category in _RULES:
        if pattern.search(text):
            return category
    return "preference"
