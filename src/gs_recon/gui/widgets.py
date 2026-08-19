"""Small reusable pieces of the window."""

from __future__ import annotations

from typing import Optional

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QVBoxLayout,
    QWidget,
)

# Status glyphs for the project tree. Text rather than icons so the tool stays
# a single dependency-free package and still reads at a glance.
GLYPH = {
    "pending": "○",
    "running": "◐",
    "done": "●",
    "failed": "✕",
    "skipped": "–",
}


def hint(text: str) -> QLabel:
    label = QLabel(text)
    label.setObjectName("sectionHint")
    label.setWordWrap(True)
    return label


def form(parent: Optional[QWidget] = None) -> QFormLayout:
    layout = QFormLayout(parent) if parent is not None else QFormLayout()
    layout.setLabelAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
    layout.setFormAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
    layout.setHorizontalSpacing(14)
    layout.setVerticalSpacing(9)
    layout.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
    return layout


def row(*widgets: QWidget, stretch_last: bool = False) -> QWidget:
    container = QWidget()
    layout = QHBoxLayout(container)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(8)
    for index, widget in enumerate(widgets):
        layout.addWidget(widget, 1 if (stretch_last and index == len(widgets) - 1) else 0)
    if not stretch_last:
        layout.addStretch()
    return container


class TextDialog(QDialog):
    """Read-only monospace dump -- used for the plan preview and doctor report."""

    def __init__(self, title: str, body: str, parent: Optional[QWidget] = None,
                 *, subtitle: str = "", width: int = 1000, height: int = 620):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(width, height)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(10)

        if subtitle:
            layout.addWidget(hint(subtitle))

        self.view = QPlainTextEdit()
        self.view.setObjectName("log")
        self.view.setReadOnly(True)
        self.view.setPlainText(body)
        self.view.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        layout.addWidget(self.view, 1)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        copy_button = buttons.addButton("Copy to clipboard", QDialogButtonBox.ButtonRole.ActionRole)
        copy_button.clicked.connect(self._copy)
        buttons.rejected.connect(self.reject)
        buttons.accepted.connect(self.accept)
        layout.addWidget(buttons)

    def _copy(self) -> None:
        from PyQt6.QtWidgets import QApplication

        clipboard = QApplication.clipboard()
        if clipboard is not None:
            clipboard.setText(self.view.toPlainText())
