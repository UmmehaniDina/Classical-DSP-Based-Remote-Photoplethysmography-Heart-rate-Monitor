"""Design tokens, colors, fonts, and ttk styling for the rPPG monitor GUI."""

from __future__ import annotations

import platform
import tkinter as tk
from tkinter import ttk

# The palette is intentionally mutated in-place when the user switches themes.
# Widgets import this mapping directly, so keeping its identity stable makes theme
# changes safe for canvases and for code that already holds a reference to it.
DARK_PALETTE = {
    # Surface & background hierarchy
    "bg_root": "#050B14",       # Deep midnight obsidian background
    "bg_panel": "#08101E",      # Sidebar, headers & statusbar
    "bg_card": "#0A1324",       # Elevated card surfaces
    "bg_card_inner": "#060B16", # Recessed display viewports and plot areas
    "bg_hover": "#131F35",      # Interactive hover state
    "border": "#152238",        # Subtle navy borders
    "border_subtle": "#0F1829", # Fine grid lines and dividers
    "border_focus": "#0066FF",  # Focused element outline

    # Pure, legible typography
    "text_primary": "#FFFFFF",  # Crisp white for headings and values
    "text_secondary": "#8E9CAE",# Refined slate for labels and descriptions
    "text_muted": "#506176",    # Understated caption gray

    # Clinical accents
    "accent_blue": "#0066FF",   # Electric blue for primary action and active tab
    "accent_blue_hover": "#0052CC",
    "accent_cyan": "#00E5FF",   # Electric cyan for pulse waveforms and BPM history
    "accent_heart": "#FF3366",  # Coral red for systolic peaks and heart accents
    "accent_emerald": "#10B981",# Medical emerald for normal badge and active state
    "accent_emerald_bg": "#064E3B", # Badge fill
    "accent_purple": "#A855F7", # Vibrant purple for Welch PSD curve and Ground Truth
    "accent_purple_fill": "#20123A", # Shaded area under PSD curve
    "accent_indigo": "#A855F7", # Alias for backward compatibility
    "accent_amber": "#F59E0B",  # Warning amber
    "accent_sky": "#38BDF8",

    # Theme-aware utility colors
    "text_on_accent": "#FFFFFF",
    "accent_blue_pressed": "#0040A0",
    "accent_heart_hover": "#E11D48",
    "accent_heart_pressed": "#BE123C",
    "accent_warning_bg": "#451A03",
    "badge_idle_bg": "#152238",
    "badge_icon_bg": "#071224",
    "plot_marker_outline": "#FFFFFF",
    "accent_purple_label": "#E9D5FF",
    "progress_track": "#121D30",

    # Channels
    "channel_r": "#EF4444",
    "channel_g": "#10B981",
    "channel_b": "#3B82F6",
}


LIGHT_PALETTE = {
    # A cool clinical light mode: lower glare than pure white, with the same
    # blue/cyan monitoring identity as the dark interface.
    "bg_root": "#EEF5FC",
    "bg_panel": "#F8FBFF",
    "bg_card": "#FFFFFF",
    "bg_card_inner": "#F3F8FD",
    "bg_hover": "#E2EEF9",
    "border": "#C8D8E8",
    "border_subtle": "#DCE7F1",
    "border_focus": "#0066FF",
    "text_primary": "#10233A",
    "text_secondary": "#4B6279",
    "text_muted": "#71859A",
    "accent_blue": "#006DE8",
    "accent_blue_hover": "#005BC4",
    "accent_cyan": "#008EAE",
    "accent_heart": "#E13B62",
    "accent_emerald": "#078765",
    "accent_emerald_bg": "#D9F5E9",
    "accent_purple": "#7652D4",
    "accent_purple_fill": "#E9E2FA",
    "accent_indigo": "#7652D4",
    "accent_amber": "#B76A00",
    "accent_sky": "#167EBC",
    "text_on_accent": "#FFFFFF",
    "accent_blue_pressed": "#004A9D",
    "accent_heart_hover": "#C92F52",
    "accent_heart_pressed": "#A92744",
    "accent_warning_bg": "#FFF1D6",
    "badge_idle_bg": "#E6EEF6",
    "badge_icon_bg": "#E2F1FB",
    "plot_marker_outline": "#FFFFFF",
    "accent_purple_label": "#5F3CB4",
    "progress_track": "#D8E6F2",
    "channel_r": "#D9364F",
    "channel_g": "#078765",
    "channel_b": "#167EBC",
}


PALETTE = dict(DARK_PALETTE)
_active_theme = "dark"


def set_theme(theme: str) -> str:
    """Select ``dark`` or ``light`` and return the normalized theme name."""
    global _active_theme
    normalized = theme.strip().lower()
    if normalized not in {"dark", "light"}:
        raise ValueError("theme must be 'dark' or 'light'")
    PALETTE.clear()
    PALETTE.update(DARK_PALETTE if normalized == "dark" else LIGHT_PALETTE)
    _active_theme = normalized
    return _active_theme


def get_theme() -> str:
    """Return the selected interface appearance."""
    return _active_theme


def get_font_family() -> str:
    """Return the cleanest available modern sans-serif font."""
    system = platform.system().lower()
    if system == "windows":
        return "Segoe UI"
    if system == "darwin":
        return "SF Pro Display"
    return "Helvetica"


FAMILY = get_font_family()

FONTS = {
    "title": (FAMILY, 18, "bold"),
    "subtitle": (FAMILY, 10),
    "heading": (FAMILY, 11, "bold"),
    "subheading": (FAMILY, 10, "bold"),
    "body": (FAMILY, 10),
    "body_bold": (FAMILY, 10, "bold"),
    "caption": (FAMILY, 9),
    "caption_bold": (FAMILY, 9, "bold"),
    "bpm_large": (FAMILY, 38, "bold"),
    "bpm_unit": (FAMILY, 12, "bold"),
    "metric_value": (FAMILY, 20, "bold"),
    "hero_sqi": (FAMILY, 24, "bold"),
    "mono": ("Consolas" if FAMILY == "Segoe UI" else "Courier", 9),
}


def apply_custom_theme(root: tk.Tk) -> ttk.Style:
    """Apply the active clinical color theme across ttk widgets."""
    style = ttk.Style(root)

    available = style.theme_names()
    if "clam" in available:
        style.theme_use("clam")

    # General Defaults
    style.configure(
        ".",
        background=PALETTE["bg_root"],
        foreground=PALETTE["text_primary"],
        font=FONTS["body"],
        borderwidth=0,
    )

    # Frames
    style.configure("Root.TFrame", background=PALETTE["bg_root"])
    style.configure("Panel.TFrame", background=PALETTE["bg_panel"])
    style.configure("Card.TFrame", background=PALETTE["bg_card"])
    style.configure("InnerCard.TFrame", background=PALETTE["bg_card_inner"])

    # Labels
    style.configure("TLabel", background=PALETTE["bg_root"], foreground=PALETTE["text_primary"])
    style.configure("Panel.TLabel", background=PALETTE["bg_panel"], foreground=PALETTE["text_primary"])
    style.configure("Card.TLabel", background=PALETTE["bg_card"], foreground=PALETTE["text_primary"])
    style.configure("CardMuted.TLabel", background=PALETTE["bg_card"], foreground=PALETTE["text_secondary"], font=FONTS["caption"])
    style.configure("CardHeading.TLabel", background=PALETTE["bg_card"], foreground=PALETTE["text_primary"], font=FONTS["heading"])
    style.configure("Title.TLabel", background=PALETTE["bg_panel"], foreground=PALETTE["text_primary"], font=FONTS["title"])
    style.configure("Subtitle.TLabel", background=PALETTE["bg_panel"], foreground=PALETTE["text_secondary"], font=FONTS["subtitle"])

    # Primary Start Button (Electric Blue)
    style.configure(
        "Start.TButton",
        background=PALETTE["accent_blue"],
        foreground=PALETTE["text_on_accent"],
        font=FONTS["body_bold"],
        padding=(14, 8),
        borderwidth=0,
        relief="flat",
    )
    style.map(
        "Start.TButton",
        background=[("active", PALETTE["accent_blue_hover"]), ("pressed", PALETTE["accent_blue_pressed"]), ("disabled", PALETTE["badge_idle_bg"])],
        foreground=[("disabled", PALETTE["text_muted"])],
    )

    # Stop Button (Coral Red)
    style.configure(
        "Stop.TButton",
        background=PALETTE["accent_heart"],
        foreground=PALETTE["text_on_accent"],
        font=FONTS["body_bold"],
        padding=(14, 8),
        borderwidth=0,
        relief="flat",
    )
    style.map(
        "Stop.TButton",
        background=[("active", PALETTE["accent_heart_hover"]), ("pressed", PALETTE["accent_heart_pressed"]), ("disabled", PALETTE["badge_idle_bg"])],
        foreground=[("disabled", PALETTE["text_muted"])],
    )

    # Secondary Action Button (Dark Navy)
    style.configure(
        "Secondary.TButton",
        background=PALETTE["bg_card"],
        foreground=PALETTE["text_primary"],
        font=FONTS["body"],
        padding=(10, 6),
        borderwidth=1,
        relief="flat",
    )
    style.map(
        "Secondary.TButton",
        background=[("active", PALETTE["bg_hover"]), ("pressed", PALETTE["bg_card_inner"])],
        foreground=[("active", PALETTE["text_primary"])],
    )

    # Reset Button (Dark with subtle border)
    style.configure(
        "Reset.TButton",
        background=PALETTE["bg_card"],
        foreground=PALETTE["text_secondary"],
        font=FONTS["body"],
        padding=(12, 6),
        borderwidth=1,
        relief="flat",
    )
    style.map(
        "Reset.TButton",
        background=[("active", PALETTE["bg_hover"]), ("pressed", PALETTE["bg_card_inner"])],
        foreground=[("active", PALETTE["text_primary"])],
    )

    # Combobox
    style.configure(
        "TCombobox",
        background=PALETTE["bg_card"],
        fieldbackground=PALETTE["bg_card_inner"],
        foreground=PALETTE["text_primary"],
        bordercolor=PALETTE["border"],
        darkcolor=PALETTE["bg_card_inner"],
        lightcolor=PALETTE["border"],
        arrowcolor=PALETTE["text_secondary"],
        font=FONTS["body"],
        padding=(8, 6),
    )
    style.map(
        "TCombobox",
        fieldbackground=[("readonly", PALETTE["bg_card_inner"])],
        foreground=[("readonly", PALETTE["text_primary"])],
    )

    # Header-level appearance toggle. It is intentionally secondary so the
    # monitoring state remains the visual priority.
    style.configure(
        "Theme.TButton",
        background=PALETTE["bg_panel"],
        foreground=PALETTE["text_secondary"],
        font=FONTS["body"],
        padding=(8, 5),
        borderwidth=1,
        relief="flat",
    )
    style.map(
        "Theme.TButton",
        background=[("active", PALETTE["bg_hover"]), ("pressed", PALETTE["bg_card_inner"])],
        foreground=[("active", PALETTE["text_primary"])],
    )

    # Checkbutton
    style.configure(
        "TCheckbutton",
        background=PALETTE["bg_panel"],
        foreground=PALETTE["text_primary"],
        font=FONTS["body"],
        indicatorbackground=PALETTE["bg_card_inner"],
        indicatorcolor=PALETTE["accent_blue"],
    )
    style.map(
        "TCheckbutton",
        background=[("active", PALETTE["bg_panel"])],
        foreground=[("active", PALETTE["text_primary"])],
    )

    # Separator
    style.configure("TSeparator", background=PALETTE["border_subtle"])

    return style
