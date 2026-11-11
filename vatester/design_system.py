"""Shared desktop design tokens and native dialog layout helpers."""
import dearpygui.dearpygui as dpg

BG = (16, 21, 29)
SURFACE = (23, 30, 40)
SURFACE_2 = (30, 39, 51)
SURFACE_3 = (40, 52, 66)
BORDER = (53, 66, 81)
ACCENT = (83, 203, 183)
ACCENT_HOVER = (115, 222, 204)
ACCENT_ACTIVE = (58, 178, 159)
TEXT = (232, 239, 245)
TEXT_SECONDARY = (169, 184, 198)
TEXT_MUTED = (137, 154, 173)
ON_ACCENT = (12, 35, 34)
POSITIVE = (117, 211, 161)
WARNING = (238, 191, 109)
NEGATIVE = (241, 145, 148)


def install(app):
    """Component styles layered over the app's shared base theme."""
    app._mk_theme("dialog", colors=[
        ("mvThemeCol_WindowBg", SURFACE),
        ("mvThemeCol_PopupBg", SURFACE),
        ("mvThemeCol_Border", BORDER),
        ("mvThemeCol_TitleBg", SURFACE),
        ("mvThemeCol_TitleBgActive", SURFACE),
    ], styles=[
        ("mvStyleVar_WindowPadding", (24, 20)),
        ("mvStyleVar_WindowRounding", 12),
        ("mvStyleVar_WindowBorderSize", 1),
        ("mvStyleVar_ItemSpacing", (12, 12)),
        ("mvStyleVar_FramePadding", (12, 7)),
    ])
    for name, active in (("nav", False), ("nav_active", True)):
        app._mk_theme(name, colors=[
            ("mvThemeCol_Button", (32, 62, 62) if active else BG),
            ("mvThemeCol_ButtonHovered", (38, 70, 69) if active else SURFACE_2),
            ("mvThemeCol_ButtonActive", (44, 81, 78)),
            ("mvThemeCol_Text", ACCENT if active else TEXT_SECONDARY),
        # Nav buttons have a fixed size (130x28); a large frame padding leaves
        # less inner room than the label needs, and ImGui then pins the text to
        # the top-left of that space instead of centering it.
        ], styles=[("mvStyleVar_FrameRounding", 6), ("mvStyleVar_FramePadding", (6, 0))])
    app._mk_theme("workspace", colors=[("mvThemeCol_ChildBg", BG)],
                  styles=[("mvStyleVar_WindowPadding", (0, 0))])
    app._mk_theme("dialog_body", colors=[("mvThemeCol_ChildBg", (0, 0, 0, 0))],
                  styles=[("mvStyleVar_WindowPadding", (0, 0))])
    app._mk_theme("inspector", colors=[("mvThemeCol_ChildBg", SURFACE)],
                  styles=[("mvStyleVar_WindowPadding", (12, 14)), ("mvStyleVar_ChildRounding", 8)])
    # Disabled controls must remain readable without looking actionable.
    with dpg.theme_component(dpg.mvButton, enabled_state=False, parent=app.themes["global"]):
        dpg.add_theme_color(dpg.mvThemeCol_Button, SURFACE_2)
        dpg.add_theme_color(dpg.mvThemeCol_Text, TEXT_MUTED)


def heading(app, title, description=None):
    item = dpg.add_text(title, color=TEXT)
    if "h2" in app.fonts:
        dpg.bind_item_font(item, app.fonts["h2"])
    if description:
        dpg.add_text(description, color=TEXT_SECONDARY, wrap=490)
    return item


def footer(app, actions):
    """Right-aligned action row: (label, callback, primary, optional tag)."""
    dpg.add_spacer(height=4)
    dpg.add_separator()
    dpg.add_spacer(height=4)
    with dpg.table(header_row=False, policy=dpg.mvTable_SizingStretchProp,
                   borders_innerH=False, borders_outerH=False, borders_innerV=False, borders_outerV=False):
        dpg.add_table_column(init_width_or_weight=1)
        for label, _, _, _ in actions:
            dpg.add_table_column(width_fixed=True, init_width_or_weight=max(100, len(label)*8+26))
        with dpg.table_row():
            dpg.add_text("")
            for label, callback, primary, tag in actions:
                args = dict(label=label, callback=callback, width=-1, height=34)
                if tag:
                    args["tag"] = tag
                button = dpg.add_button(**args)
                if primary:
                    dpg.bind_item_theme(button, app.themes["primary"])


def center_dialog(tag):
    width, height = dpg.get_item_rect_size(tag)
    cfg = dpg.get_item_configuration(tag)
    width = width or cfg.get("width", 560)
    height = height or cfg.get("height", 330) or 330
    dpg.set_item_pos(tag, (max(12, (dpg.get_viewport_client_width()-width)//2),
                           max(24, (dpg.get_viewport_client_height()-height)//2)))
