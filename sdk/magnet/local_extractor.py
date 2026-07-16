"""
LocalExtractor
--------------
Rule-based category detector: assigns project memory categories without an LLM.
Used when MAGNET_OPENAI_KEY is not set.

Categories (priority order):
  action        — concrete work that was actually completed (not proposed)
  tried_failed  — something that was attempted and didn't work
  watch_out     — warnings, things to be careful about
  decision      — explicit decisions or choices made
  convention    — team/project conventions and patterns
  goal          — what the project is working toward
  preference    — personal likes/dislikes (fallback)

Also provides compress_essence(), a mechanical (non-LLM) approximation of
telegraphic essence-compression: strips filler/hedging, drops code bodies,
and hard-caps length. This is the safety net for the rule-based extraction
path (checkpoint/save_now); the LLM-guided `remember` tool relies primarily
on its own tool-description instructions for true compression, but callers
should still run text through here as a backstop regardless of source.
"""

from __future__ import annotations

import re

# Each list is (pattern, category). First match wins.
_RULES: list[tuple[re.Pattern, str]] = []

def _r(pattern: str, cat: str) -> None:
    _RULES.append((re.compile(pattern, re.IGNORECASE), cat))

# Hedge/proposal language — if present, a text that would otherwise match
# the action rules is NOT treated as a completed action (it's a proposal or
# suggestion). Checked only against action-rule matches, not other
# categories, since e.g. "let's decide" is still meaningfully a decision.
_ACTION_HEDGES = re.compile(
    r"\b(let'?s|let us|should we|shouldn'?t we|we could|could we|"
    r"going to|gonna|plan(?:ning)? to|will |maybe|perhaps|might|"
    r"consider(?:ing)?|propos(?:e[sd]?|ing)|suggest(?:ed|ing|s)?|"
    r"thinking about|what if we|we should|why don'?t we)\b",
    re.IGNORECASE,
)

# action — check first: concrete completed work reads distinctly from
# proposals/decisions once hedge language is excluded (see _ACTION_HEDGES).
_r(r"\b(renamed|switched|removed|deleted|refactored|migrated|merged|deployed|"
   r"implemented|replaced|extracted|consolidated|rewrote|upgraded|downgraded|"
   r"reverted|wired up|integrated|updated|configured|built|wrote|created|"
   r"fixed|patched|shipped)\b", "action")
_r(r"\badded\b.{0,60}\b(to|for|in)\b", "action")
_r(r"\bjust (finished|shipped|pushed|merged|deployed)\b", "action")

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

# decision — "switch(ing)" (present/in-progress) stays a decision; the
# completed past tense "switched" is claimed by the action rules above.
_r(r"\bdecided?\b", "decision")
_r(r"\b(chose|choosing|going with|went with|we('re| are| will) use)\b", "decision")
_r(r"\bswitch(ing)?\b.{0,30}\bto\b", "decision")
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


def detect_category(text: str) -> str | None:
    """
    Classify text into a project memory category.
    Returns one of: action | decision | watch_out | tried_failed | convention | goal | preference,
    or None if nothing matched.

    None is a real answer, not a placeholder to paper over with a keyword
    whitelist downstream — text that doesn't match any category rule
    (including the explicit preference rules above: prefer/like/love/
    enjoy/want/wish/dislike/hate/avoid) isn't worth saving, full stop.
    Callers must treat None as "skip this," never coerce it to a default
    category — that's exactly the bug this replaced (see git history:
    detect_category used to default to "preference" for anything
    unmatched, which then needed a separate hardcoded phrase list downstream
    just to filter the resulting noise back out).
    """
    for pattern, category in _RULES:
        if pattern.search(text):
            if category == "action" and _ACTION_HEDGES.search(text):
                # Reads like a proposal, not completed work — keep checking
                # the remaining rules instead of mis-filing it as an action.
                continue
            return category
    return None


# ── Essence compression ────────────────────────────────────────────────────
#
# Memory items store the essence, telegraphically — not prose, not
# transcripts, never code bodies. This is a mechanical approximation (no
# LLM): strip filler/hedging, drop fenced code blocks, hard-cap length and
# truncate at a sentence-ish boundary. Applied as a backstop in both the
# local rule-based path (checkpoint/save_now) and after LLM-guided writes
# (remember) — the LLM is instructed to already write telegraphically, but
# this guarantees the cap holds regardless of whether it complies.

_CODE_FENCE = re.compile(r"```.*?```", re.DOTALL)
_INLINE_CODE = re.compile(r"`[^`]{20,}`")  # long inline code spans only — short `x` identifiers are fine to keep

_FILLER_PATTERNS = [
    re.compile(r"\bthe user (mentioned|said|noted|stated|explained) that\b", re.IGNORECASE),
    re.compile(r"\bi think (that )?\b", re.IGNORECASE),
    re.compile(r"\bit (might|could|would) be (better|good|nice|worth it) (to|if)\b", re.IGNORECASE),
    re.compile(r"\b(just|really|actually|basically|essentially|honestly)\b\s*", re.IGNORECASE),
    re.compile(r"\bplease\b\s*", re.IGNORECASE),
    re.compile(r"\b(maybe|perhaps|i guess|kind of|sort of)\b\s*", re.IGNORECASE),
    re.compile(r"^(so|well|ok(ay)?|alright)[,.]?\s+", re.IGNORECASE),
]

_MAX_ITEM_CHARS = 140  # roughly the spec's "~15 words" target, as a hard cap


def compress_essence(text: str) -> str:
    """Strip filler/hedging/code and hard-cap length, truncating at a
    sentence boundary where possible. Never returns raw code — fenced or
    long inline code spans are replaced with a short placeholder so the
    caller keeps whatever surrounding decision/gotcha text there was."""
    if not text:
        return text

    text = _CODE_FENCE.sub(" [code omitted] ", text)
    text = _INLINE_CODE.sub(" [code omitted] ", text)
    for pattern in _FILLER_PATTERNS:
        text = pattern.sub("", text)
    text = re.sub(r"\s+", " ", text).strip()

    if len(text) <= _MAX_ITEM_CHARS:
        return text

    truncated = text[:_MAX_ITEM_CHARS]
    for boundary in (". ", "; ", ", "):
        idx = truncated.rfind(boundary)
        if idx > _MAX_ITEM_CHARS * 0.4:
            return truncated[:idx].rstrip() + "."
    return truncated.rstrip() + "…"
