"""The look of the app, in one place: a Gradio theme (colour variables mapped to our tokens), one CSS file, one small script.

Colours, type and spacing are CSS custom properties in `theme.css` (light, dark). The Gradio theme below only points Gradio's own
variables at them, so nothing in Gradio keeps a colour of its own. No web fonts and no outside requests: the font stacks are
plain strings (a `GoogleFont` object would make the browser download the font from the internet).
"""
from __future__ import annotations

import inspect
from pathlib import Path

import gradio as gr

HERE = Path(__file__).resolve().parent
CSS_PATH = HERE / "theme.css"
SERIF = '"Iowan Old Style", "Palatino Linotype", Palatino, "Book Antiqua", Georgia, serif'
SANS = 'system-ui, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif'

# Gradio variable -> our token. Both the light and the dark variable point at the same token: the tokens change with the theme.
_MAP = {
    "body_background_fill": "var(--bg)",
    "body_text_color": "var(--ink)",
    "body_text_color_subdued": "var(--muted)",
    "background_fill_primary": "var(--surface)",
    "background_fill_secondary": "var(--sunk)",
    "block_background_fill": "var(--surface)",
    "block_border_color": "var(--line)",
    "block_label_background_fill": "transparent",
    "block_label_text_color": "var(--muted)",
    "block_title_text_color": "var(--ink)",
    "block_info_text_color": "var(--muted)",
    "border_color_primary": "var(--line)",
    "border_color_accent": "var(--accent-text)",
    "color_accent_soft": "var(--sunk)",
    "link_text_color": "var(--accent-text)",
    "link_text_color_hover": "var(--accent-fill-hover)",
    "link_text_color_active": "var(--accent-fill-hover)",
    "link_text_color_visited": "var(--accent-text)",
    "panel_background_fill": "var(--surface)",
    "panel_border_color": "var(--line)",
    "input_background_fill": "var(--surface)",
    "input_background_fill_focus": "var(--surface)",
    "input_background_fill_hover": "var(--surface)",
    "input_border_color": "var(--line-strong)",
    "input_border_color_focus": "var(--accent-text)",
    "input_border_color_hover": "var(--line-strong)",
    "input_placeholder_color": "var(--muted)",
    "button_primary_background_fill": "var(--accent-fill)",
    "button_primary_background_fill_hover": "var(--accent-fill-hover)",
    "button_primary_text_color": "#FFFFFF",
    "button_primary_text_color_hover": "#FFFFFF",
    "button_primary_border_color": "var(--accent-fill)",
    "button_primary_border_color_hover": "var(--accent-fill-hover)",
    "button_secondary_background_fill": "var(--surface)",
    "button_secondary_background_fill_hover": "var(--sunk)",
    "button_secondary_text_color": "var(--ink)",
    "button_secondary_text_color_hover": "var(--ink)",
    "button_secondary_border_color": "var(--line-strong)",
    "button_secondary_border_color_hover": "var(--line-strong)",
    "button_cancel_background_fill": "var(--surface)",
    "button_cancel_background_fill_hover": "var(--sunk)",
    "button_cancel_text_color": "var(--danger)",
    "button_cancel_text_color_hover": "var(--danger)",
    "button_cancel_border_color": "var(--line-strong)",
    "button_cancel_border_color_hover": "var(--line-strong)",
    "checkbox_background_color": "var(--surface)",
    "checkbox_background_color_selected": "var(--accent-fill)",
    "checkbox_border_color": "var(--line-strong)",
    "checkbox_border_color_selected": "var(--accent-fill)",
    "checkbox_label_background_fill": "transparent",
    "checkbox_label_background_fill_hover": "transparent",
    "checkbox_label_background_fill_selected": "transparent",
    "checkbox_label_text_color": "var(--ink)",
    "checkbox_label_text_color_selected": "var(--ink)",
    "shadow_drop": "none",
    "shadow_drop_lg": "none",
    "block_shadow": "none",
    "table_even_background_fill": "var(--surface)",
    "table_odd_background_fill": "var(--surface)",
    "table_row_focus": "var(--sunk)",
    "table_border_color": "var(--line)",
    "loader_color": "var(--accent-text)",
    "slider_color": "var(--accent-text)",
}


def build_theme() -> gr.themes.Base:
    theme = gr.themes.Base(
        font=[font.strip().strip('"') for font in SANS.split(",")],
        font_mono=["ui-monospace", "Consolas", "monospace"],
        radius_size=gr.themes.sizes.radius_md,
    )
    accepted = set(inspect.signature(gr.themes.Base.set).parameters)
    settings = {}
    for name, value in _MAP.items():
        for key in (name, f"{name}_dark"):
            if key in accepted:               # not every variable has a dark twin
                settings[key] = value
    return theme.set(**settings)


def css() -> str:
    return CSS_PATH.read_text(encoding="utf-8")


# Runs once when the page loads: apply the saved choice (or the system's) before anything is painted, and wire the toggle button.
# `data-theme` on <html> drives our tokens; Gradio's own `dark` class on <body> is kept in step so its inner widgets agree.
THEME_JS = """
() => {
  const root = document.documentElement;
  const read = () => { try { return localStorage.getItem('sb-theme'); } catch (e) { return null; } };
  const write = (v) => { try { localStorage.setItem('sb-theme', v); } catch (e) { /* private window: just do not remember */ } };
  const media = window.matchMedia('(prefers-color-scheme: dark)');
  const apply = (mode) => {
    const dark = mode === 'dark' || (mode !== 'light' && media.matches);
    root.setAttribute('data-theme', dark ? 'dark' : 'light');
    document.body.classList.toggle('dark', dark);
    const button = document.getElementById('sb-theme');
    if (button) { button.setAttribute('aria-pressed', String(dark)); button.setAttribute('aria-label', dark ? 'Switch to light mode' : 'Switch to dark mode'); }
  };
  apply(read());
  media.addEventListener('change', () => { if (!read()) apply(null); });
  document.addEventListener('click', (event) => {
    if (!event.target.closest('#sb-theme')) return;
    const next = root.getAttribute('data-theme') === 'dark' ? 'light' : 'dark';
    write(next);
    apply(next);
  });
}
"""
