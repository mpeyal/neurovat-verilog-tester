"""Frame-pumping and layout helpers shared by the desktop GUI test scripts."""
import time

import dearpygui.dearpygui as dpg


def pump(app):
    deadline = time.monotonic()+60
    while app.trainer_running and time.monotonic() < deadline:
        app._process_queue()
        app.desktop.tick()
        dpg.render_dearpygui_frame()
    app._process_queue()
    assert not app.trainer_running


def frames(app, count=15):
    for _ in range(count):
        app.desktop.tick()
        app._process_queue()
        dpg.render_dearpygui_frame()


def activate(app, tag):
    item = dpg.get_alias_id(tag)
    while item:
        info = dpg.get_item_info(item)
        parent = info.get('parent')
        if info.get('type', '').endswith('::mvTab'):
            dpg.set_value(parent, dpg.get_item_alias(item) or item)
        item = parent
    app._on_center_tab()
    frames(app)


def check(app, tags):
    frames(app)
    bottom = dpg.get_viewport_client_height()-12
    for tag in tags:
        pos = dpg.get_item_rect_min(tag)
        size = dpg.get_item_rect_size(tag)
        assert size[1] >= 85, (tag, pos, size)
        assert pos[1]+size[1] <= bottom+2, (tag, pos, size, bottom)
        assert pos[0]+size[0] <= dpg.get_viewport_client_width(), (tag, pos, size)
    assert dpg.get_y_scroll('center_child') == 0
