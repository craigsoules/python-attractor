"""Human-in-the-loop interviewer implementations."""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class QuestionType(str, Enum):
    YES_NO = "yes_no"
    MULTIPLE_CHOICE = "multiple_choice"
    FREEFORM = "freeform"
    CONFIRMATION = "confirmation"


@dataclass
class Option:
    key: str
    label: str


@dataclass
class Answer:
    value: str = ""
    selected_option: Option | None = None
    text: str = ""


@dataclass
class Question:
    text: str
    type: QuestionType
    options: list[Option] = field(default_factory=list)
    default: Answer | None = None
    timeout_seconds: float | None = None
    stage: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


class Interviewer(ABC):
    @abstractmethod
    def ask(self, question: Question) -> Answer:
        ...


class AutoApproveInterviewer(Interviewer):
    """Always approves – for automation and testing."""

    def ask(self, question: Question) -> Answer:
        if question.type in (QuestionType.YES_NO, QuestionType.CONFIRMATION):
            return Answer(value="YES")
        if question.type == QuestionType.MULTIPLE_CHOICE and question.options:
            opt = question.options[0]
            return Answer(value=opt.key, selected_option=opt, text=opt.label)
        return Answer(value="auto-approved")


class ConsoleInterviewer(Interviewer):
    """Interactive CLI interviewer."""

    def ask(self, question: Question) -> Answer:
        print(f"\n[?] {question.text}")

        if question.type == QuestionType.MULTIPLE_CHOICE:
            for opt in question.options:
                print(f"  [{opt.key}] {opt.label}")
            response = input("Select: ").strip()
            for opt in question.options:
                if opt.key.lower() == response.lower():
                    return Answer(value=opt.key, selected_option=opt)
            if question.options:
                return Answer(value=question.options[0].key, selected_option=question.options[0])
            return Answer(value=response)

        if question.type in (QuestionType.YES_NO, QuestionType.CONFIRMATION):
            response = input("[Y/N]: ").strip().lower()
            return Answer(value="YES" if response in ("y", "yes") else "NO")

        if question.type == QuestionType.FREEFORM:
            text = input("> ").strip()
            return Answer(value=text, text=text)

        return Answer(value="")


class QueueInterviewer(Interviewer):
    """Pre-filled answer queue – for testing."""

    def __init__(self, answers: list[Answer]) -> None:
        self.answers = list(answers)
        self.index = 0

    def ask(self, question: Question) -> Answer:
        if self.index < len(self.answers):
            answer = self.answers[self.index]
            self.index += 1
            return answer
        return Answer(value="SKIPPED")
