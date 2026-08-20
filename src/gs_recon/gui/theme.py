"""A single stylesheet for the whole window.

The original tool scattered inline setStyleSheet calls across a dozen widgets,
which is why nothing lined up. Everything lives here now, and the palette flips
with the desktop theme instead of hard-coding light colours that turn into
white-on-white for dark-mode users.
"""

from __future__ import annotations

LIGHT = {
    "bg": "#f4f5f7",
    "surface": "#ffffff",
    "surface_alt": "#eceef1",
    "border": "#d3d7de",
    "text": "#1c1f24",
    "text_dim": "#5c636e",
    "accent": "#2563eb",
    "accent_text": "#ffffff",
    "ok": "#15803d",
    "warn": "#b45309",
    "fail": "#b91c1c",
    "running": "#2563eb",
    "log_bg": "#1b1e23",
    "log_text": "#e6e8ec",
}

DARK = {
    "bg": "#1b1e23",
    "surface": "#24282f",
    "surface_alt": "#2c313a",
    "border": "#3a404a",
    "text": "#e6e8ec",
    "text_dim": "#9aa2af",
    "accent": "#3b82f6",
    "accent_text": "#ffffff",
    "ok": "#4ade80",
    "warn": "#fbbf24",
    "fail": "#f87171",
    "running": "#60a5fa",
    "log_bg": "#12151a",
    "log_text": "#e6e8ec",
}


def palette_for(is_dark: bool) -> dict[str, str]:
    return DARK if is_dark else LIGHT


def apply_palette(app, colors: dict[str, str]) -> None:
    """Set base colours through QPalette rather than a global QWidget rule.

    Styling ``QWidget { background-color: ... }`` in a stylesheet looks like the
    obvious way to set a window background, but it switches every descendant to
    stylesheet painting -- which silently removes the frame Qt draws around a
    checkbox indicator, leaving a bare tick and, when unchecked, nothing at all.
    The palette sets the same colours without disturbing native painting.
    """
    from PyQt6.QtGui import QColor, QPalette

    palette = QPalette()
    window, text = QColor(colors["bg"]), QColor(colors["text"])
    palette.setColor(QPalette.ColorRole.Window, window)
    palette.setColor(QPalette.ColorRole.WindowText, text)
    palette.setColor(QPalette.ColorRole.Base, QColor(colors["surface_alt"]))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor(colors["surface"]))
    palette.setColor(QPalette.ColorRole.Text, text)
    palette.setColor(QPalette.ColorRole.Button, QColor(colors["surface_alt"]))
    palette.setColor(QPalette.ColorRole.ButtonText, text)
    palette.setColor(QPalette.ColorRole.ToolTipBase, QColor(colors["surface"]))
    palette.setColor(QPalette.ColorRole.ToolTipText, text)
    palette.setColor(QPalette.ColorRole.PlaceholderText, QColor(colors["text_dim"]))
    palette.setColor(QPalette.ColorRole.Highlight, QColor(colors["accent"]))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor(colors["accent_text"]))
    palette.setColor(
        QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text, QColor(colors["text_dim"])
    )
    palette.setColor(
        QPalette.ColorGroup.Disabled, QPalette.ColorRole.ButtonText, QColor(colors["text_dim"])
    )
    palette.setColor(
        QPalette.ColorGroup.Disabled, QPalette.ColorRole.WindowText, QColor(colors["text_dim"])
    )
    app.setPalette(palette)


def stylesheet(colors: dict[str, str]) -> str:
    return f"""
    QMainWindow, QDialog {{ background-color: {colors['bg']}; }}
    QSplitter, QTabWidget, QScrollArea > QWidget > QWidget {{ background: transparent; }}
    QGroupBox {{
        background-color: {colors['surface']};
        border: 1px solid {colors['border']};
        border-radius: 10px;
        margin-top: 14px;
        padding: 16px 14px 12px 14px;
        font-weight: 600;
    }}
    QGroupBox::title {{
        subcontrol-origin: margin;
        subcontrol-position: top left;
        left: 14px;
        top: 3px;
        padding: 0 4px;
        color: {colors['text_dim']};
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }}
    QTabWidget::pane {{
        border: 1px solid {colors['border']};
        border-radius: 10px;
        background-color: {colors['surface']};
        top: -1px;
    }}
    QTabBar::tab {{
        background: transparent;
        color: {colors['text_dim']};
        padding: 9px 18px;
        margin-right: 2px;
        border: 1px solid transparent;
        border-top-left-radius: 8px;
        border-top-right-radius: 8px;
        font-weight: 600;
    }}
    QTabBar::tab:selected {{
        background: {colors['surface']};
        color: {colors['text']};
        border-color: {colors['border']};
        border-bottom-color: {colors['surface']};
    }}
    QTabBar::tab:hover:!selected {{ color: {colors['text']}; }}
    QTabBar::tab:disabled {{ color: {colors['border']}; }}

    QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QPlainTextEdit, QTextEdit {{
        background-color: {colors['surface_alt']};
        border: 1px solid {colors['border']};
        border-radius: 6px;
        padding: 5px 8px;
        selection-background-color: {colors['accent']};
        selection-color: {colors['accent_text']};
    }}
    QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {{
        border-color: {colors['accent']};
    }}
    QLineEdit:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled, QComboBox:disabled {{
        color: {colors['text_dim']};
        background-color: {colors['bg']};
    }}
    QComboBox::drop-down {{ border: none; width: 20px; }}
    QComboBox QAbstractItemView {{
        background-color: {colors['surface']};
        border: 1px solid {colors['border']};
        selection-background-color: {colors['accent']};
        selection-color: {colors['accent_text']};
    }}

    QPushButton {{
        background-color: {colors['surface_alt']};
        border: 1px solid {colors['border']};
        border-radius: 7px;
        padding: 7px 14px;
        font-weight: 600;
    }}
    QPushButton:hover {{ border-color: {colors['accent']}; }}
    QPushButton:pressed {{ background-color: {colors['border']}; }}
    QPushButton:disabled {{ color: {colors['text_dim']}; border-color: {colors['border']}; }}
    QPushButton#primary {{
        background-color: {colors['accent']};
        color: {colors['accent_text']};
        border-color: {colors['accent']};
    }}
    QPushButton#primary:disabled {{
        background-color: {colors['surface_alt']};
        color: {colors['text_dim']};
        border-color: {colors['border']};
    }}
    QPushButton#danger:enabled {{ color: {colors['fail']}; border-color: {colors['fail']}; }}

    QTreeWidget {{
        background-color: {colors['surface']};
        border: 1px solid {colors['border']};
        border-radius: 10px;
        padding: 4px;
        outline: none;
    }}
    QTreeWidget::item {{ padding: 4px 2px; border-radius: 4px; }}
    QTreeWidget::item:selected {{
        background-color: {colors['accent']};
        color: {colors['accent_text']};
    }}

    QPlainTextEdit#log {{
        background-color: {colors['log_bg']};
        color: {colors['log_text']};
        border: 1px solid {colors['border']};
        border-radius: 8px;
        font-family: "JetBrains Mono", "DejaVu Sans Mono", "Courier New", monospace;
        font-size: 12px;
    }}

    QProgressBar {{
        background-color: {colors['surface_alt']};
        border: 1px solid {colors['border']};
        border-radius: 7px;
        height: 14px;
        text-align: center;
        font-size: 11px;
        color: {colors['text_dim']};
    }}
    QProgressBar::chunk {{ background-color: {colors['accent']}; border-radius: 6px; }}

    QLabel {{ background: transparent; }}

    QLabel#hint {{ color: {colors['text_dim']}; font-size: 12px; }}
    QLabel#sectionHint {{ color: {colors['text_dim']}; font-size: 12px; padding-bottom: 6px; }}
    QLabel#liveMath {{
        color: {colors['accent']};
        font-size: 12px;
        padding: 6px 8px;
        border-left: 3px solid {colors['accent']};
        background-color: {colors['surface']};
    }}
    QLabel#statusPill {{
        border: 1px solid {colors['border']};
        border-radius: 12px;
        padding: 5px 14px;
        font-weight: 600;
        background-color: {colors['surface']};
    }}

    /* Indicator deliberately unstyled -- see apply_palette() above for why. */
    QCheckBox {{ spacing: 8px; padding: 3px 0; }}
    QSplitter::handle {{ background-color: transparent; }}
    QScrollArea {{ border: none; background: transparent; }}
    QScrollBar:vertical {{
        background: transparent; width: 10px; margin: 2px;
    }}
    QScrollBar::handle:vertical {{
        background: {colors['border']}; border-radius: 5px; min-height: 30px;
    }}
    QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; width: 0; }}
    QToolTip {{
        background-color: {colors['surface']};
        color: {colors['text']};
        border: 1px solid {colors['border']};
        padding: 5px 8px;
        border-radius: 6px;
    }}
    """
