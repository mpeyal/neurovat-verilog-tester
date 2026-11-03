"""Shared plot navigation, probes, legends and anchored context menus.

App supplies plot state and host callbacks. Keeping the callback methods on a
mixin preserves DearPyGui bindings and control-bridge callers. These methods
run on the render thread; they do not own device or training calculations.
"""
import bisect

import dearpygui.dearpygui as dpg

from .ui_constants import C_ACC, C_GREEN, C_MUTED


class PlotInteractionMixin:
    # ---- plot zoom / fit controls ------------------------------------
    # Wheel zoom is reimplemented here: instead of ImPlot's hard step per
    # notch, targets are animated each frame (exponential easing) and the
    # zoom is centered on the mouse cursor.

    ZOOM_PLOTS = {
        "preview_plot": (("prev_x",), ("prev_y",)),
        "plot_i": (("ax_i_x", "ax_r_x", "ax_g_x"), ("ax_i_y",)),
        "plot_r": (("ax_i_x", "ax_r_x", "ax_g_x"), ("ax_r_y",)),
        "plot_g": (("ax_i_x", "ax_r_x", "ax_g_x"), ("ax_g_y",)),
        "ana_plot": (("ana_x",), ("ana_y",)),
        "stdp_plot": (("stdp_x",), ("stdp_y",)),
    }

    def _axis_view(self, ax):
        a = self._zoom_anim.get(ax)
        return (a[2], a[3]) if a else tuple(dpg.get_axis_limits(ax))

    def _push_zoom(self, ax, factor, center=None):
        lo, hi = self._axis_view(ax)
        span = hi - lo
        if span <= 0 or (factor < 1.0 and span * factor < 1e-12):
            return
        c = 0.5 * (lo + hi) if center is None else center
        t_lo = c - (c - lo) * factor
        t_hi = c + (hi - c) * factor
        cur = self._zoom_anim.get(ax)
        if cur:                       # retarget mid-animation (compounds)
            cur[2], cur[3] = t_lo, t_hi
        else:
            self._zoom_anim[ax] = [lo, hi, t_lo, t_hi]

    def _tick_zoom_anim(self):
        for ax in self._zoom_release:
            if dpg.does_item_exist(ax) and ax not in self._zoom_anim:
                dpg.set_axis_limits_auto(ax)
        self._zoom_release = []
        if not self._zoom_anim:
            return
        done = []
        for ax, st in self._zoom_anim.items():
            if not dpg.does_item_exist(ax):
                done.append(ax)
                continue
            st[0] += (st[2] - st[0]) * 0.28
            st[1] += (st[3] - st[1]) * 0.28
            span = abs(st[3] - st[2]) or 1.0
            if (abs(st[2] - st[0]) < span * 2e-3
                    and abs(st[3] - st[1]) < span * 2e-3):
                dpg.set_axis_limits(ax, st[2], st[3])
                done.append(ax)
            else:
                dpg.set_axis_limits(ax, st[0], st[1])
        for ax in done:
            self._zoom_anim.pop(ax, None)
            self._zoom_release.append(ax)

    def _on_wheel(self, sender, delta):
        if not delta:
            return
        if (dpg.does_item_exist("nt_canvas_scroll")
                and dpg.is_item_hovered("nt_canvas_scroll")):
            self._nt_set_zoom(self._nt_zoom * (1.12 ** float(delta)))
            return
        hovered = next((p for p in self.ZOOM_PLOTS
                        if dpg.does_item_exist(p) and dpg.is_item_hovered(p)),
                       None)
        if not hovered:
            return
        try:
            mx, my = dpg.get_plot_mouse_pos()
        except Exception:
            mx = my = None
        f = 0.85 ** float(delta)        # wheel up -> zoom in, smooth ramp
        xs, ys = self.ZOOM_PLOTS[hovered]
        for ax in xs:
            if dpg.does_item_exist(ax):
                self._push_zoom(ax, f, mx)
        for ay in ys:
            if dpg.does_item_exist(ay):
                self._push_zoom(ay, f, my)

    def _zoom_axes(self, axes, factor):
        for ax in axes:
            if dpg.does_item_exist(ax):
                self._push_zoom(ax, factor)

    def _fit_axes_of(self, axes):
        for ax in axes:
            self._zoom_anim.pop(ax, None)
            if dpg.does_item_exist(ax):
                dpg.set_axis_limits_auto(ax)
                dpg.fit_axis_data(ax)

    # per-plot hover labels: (x_name, x_unit, y_name, y_unit); "{unit}" is
    # resolved from the current signal-unit combo at runtime.
    HOVER_FMT = {
        "preview_plot": ("t", "s", "amp", "{unit}"),
        "plot_i": ("t", "s", "stim", "{unit}"),
        "plot_r": ("t", "s", "R", "ohm"),
        "plot_g": ("t", "s", "G", "uS"),
        "ana_plot": ("pulse", "", "G", "uS"),
        "stdp_plot": ("dt", "ms", "dG", "uS"),
    }

    def _bind_plot_hover_handlers(self):
        """Per-plot hover handlers. get_plot_mouse_pos() is only valid inside
        an input callback (not when polled in the render loop), which is why
        the tooltip must be driven from a hover handler, like the click path.
        """
        for plot in self.HOVER_FMT:
            if not dpg.does_item_exist(plot):
                continue
            reg = f"hh_{plot}"
            if not dpg.does_item_exist(reg):
                with dpg.item_handler_registry(tag=reg):
                    dpg.add_item_hover_handler(callback=self._on_plot_hover,
                                               user_data=plot)
            dpg.bind_item_handler_registry(plot, reg)

    def _nearest_series_point(self, yax, mx, my, xs_span, ys_span):
        """Nearest data point (axis-normalized) across a plot's data series,
        skipping ## helper series. Returns (d2, x, y, label) or None. Robust
        to series that vanish mid-frame."""
        best = None
        for sid in dpg.get_item_children(yax, 1) or []:
            try:
                lbl = dpg.get_item_configuration(sid).get("label", "") or ""
                if lbl.startswith("##"):
                    continue
                val = dpg.get_value(sid)
            except Exception:
                continue
            if not val or len(val) < 2 or not val[0]:
                continue
            for x, y in zip(val[0], val[1]):
                if y != y:                       # NaN gap-break point
                    continue
                d = ((x - mx) / xs_span) ** 2 + ((y - my) / ys_span) ** 2
                if best is None or d < best[0]:
                    best = (d, float(x), float(y), lbl)
        return best

    def _hide_hover_bubble(self):
        for aid in self._hover_ann.values():
            if dpg.does_item_exist(aid):
                dpg.hide_item(aid)

    def _hide_tip_if_stale(self):
        # the hover handler stops firing once the cursor leaves the plot;
        # hide the in-canvas bubble shortly after.
        if dpg.get_frame_count() - self._tip_frame > 1:
            self._hide_hover_bubble()

    def _get_hover_data(self, plot):
        """Cached (label, xs, ys) lists for a plot's data series, so the hover
        handler doesn't copy big arrays every frame. Rebuilt lazily after a
        re-plot (the cache is cleared when data changes)."""
        data = self._hover_cache.get(plot)
        if data is not None:
            return data
        yax = self.PROBE_AXES[plot][1]
        data = []
        for sid in dpg.get_item_children(yax, 1) or []:
            try:
                lbl = dpg.get_item_configuration(sid).get("label", "") or ""
                if lbl.startswith("##"):
                    continue
                val = dpg.get_value(sid)
            except Exception:
                continue
            if not val or len(val) < 2 or not val[0]:
                continue
            data.append((lbl, list(val[0]), list(val[1])))
        self._hover_cache[plot] = data
        return data

    @staticmethod
    def _nearest_on_series(xs, ys, mx, my, xspan, yspan):
        """Nearest point on a single x-sorted series via binary search +
        outward scan with early termination (x-distance is monotonic). Returns
        (d2, x, y) or None. Skips NaN y (gap breaks)."""
        n = len(xs)
        if n == 0:
            return None
        i = bisect.bisect_left(xs, mx)
        best = None
        lo, hi = i - 1, i
        while lo >= 0 or hi < n:
            for j in (lo, hi):
                if 0 <= j < n:
                    dxn = (xs[j] - mx) / xspan
                    if best is not None and dxn * dxn >= best[0]:
                        continue
                    yv = ys[j]
                    if yv != yv:                 # NaN
                        continue
                    d = dxn * dxn + ((yv - my) / yspan) ** 2
                    if best is None or d < best[0]:
                        best = (d, float(xs[j]), float(yv))
            ldx = ((xs[lo] - mx) / xspan) ** 2 if lo >= 0 else float("inf")
            hdx = ((xs[hi] - mx) / xspan) ** 2 if hi < n else float("inf")
            if best is not None and ldx >= best[0] and hdx >= best[0]:
                break
            lo -= 1
            hi += 1
        return best

    def _probe_near(self, plot, mx, my, px_radius=16.0):
        """'A'/'B' if the cursor is within px_radius pixels of that probe."""
        probes = self._probes.get(plot)
        if not probes:
            return None
        xax, yax = self.PROBE_AXES[plot]
        x0, x1 = dpg.get_axis_limits(xax)
        y0, y1 = dpg.get_axis_limits(yax)
        w, h = dpg.get_item_rect_size(plot) or (1, 1)
        pxx = (w or 1) / ((x1 - x0) or 1.0)
        pxy = (h or 1) / ((y1 - y0) or 1.0)
        best = None
        for which in ("A", "B"):
            m = probes.get(which)
            if not m:
                continue
            d2 = ((m["x"] - mx) * pxx) ** 2 + ((m["y"] - my) * pxy) ** 2
            if d2 <= px_radius ** 2 and (best is None or d2 < best[0]):
                best = (d2, which)
        return best[1] if best else None

    def _set_plot_pan_disabled(self, plot, disabled):
        """Park the plot's pan on an unused button while the cursor is on a
        probe, so pressing the mouse grabs the probe instead of panning."""
        if (plot in self._pan_disabled) == bool(disabled):
            return
        try:
            dpg.configure_item(plot, pan_button=(dpg.mvMouseButton_X2
                                                 if disabled else
                                                 dpg.mvMouseButton_Left))
        except Exception:
            return
        if disabled:
            self._pan_disabled.add(plot)
        else:
            self._pan_disabled.discard(plot)

    def _drag_probe_to(self, plot, which, mx, my):
        m = self._probes.get(plot, {}).get(which)
        if not m:
            return
        dpg.set_value(m["pt"], (mx, my))
        if m.get("ann"):
            dpg.configure_item(m["ann"], default_value=(mx, my))
        m["x"], m["y"] = mx, my
        self._update_probe_readout(plot)

    def _units_per_px(self, plot, gx, gy, mx, my):
        """Plot-units per screen pixel at grab time. Baseline: axis span over
        the item rect minus typical axis-label margins (sign always right).
        Refined by a least-squares fit over consecutive hover samples when
        the history is consistent with that estimate."""
        xax, yax = self.PROBE_AXES[plot]
        x0, x1 = dpg.get_axis_limits(xax)
        y0, y1 = dpg.get_axis_limits(yax)
        w, h = dpg.get_item_rect_size(plot) or (1, 1)
        ux = (x1 - x0) / max((w or 1) - 62.0, 1.0)
        uy = -(y1 - y0) / max((h or 1) - 30.0, 1.0)  # screen y grows downward
        hist = self._hover_hist.get(plot, [])
        sxx = sxm = syy = sym = 0.0
        for (g1x, g1y, m1x, m1y), (g2x, g2y, m2x, m2y) in zip(hist, hist[1:]):
            dgx, dgy = g2x - g1x, g2y - g1y
            sxx += dgx * dgx
            sxm += dgx * (m2x - m1x)
            syy += dgy * dgy
            sym += dgy * (m2y - m1y)
        if sxx > 25.0:
            cand = sxm / sxx
            if cand * ux > 0 and 0.4 < abs(cand / ux) < 2.5:
                ux = cand
        if syy > 25.0:
            cand = sym / syy
            if cand * uy > 0 and 0.4 < abs(cand / uy) < 2.5:
                uy = cand
        return ux, uy

    def _tick_probe_drag(self):
        """Render-loop drag driver: while a probe is grabbed, move it with
        the GLOBAL mouse position (which stays live during the hold, unlike
        the plot-space mouse position)."""
        if not self._dragging or not self._drag_anchor:
            return
        if not dpg.is_mouse_button_down(dpg.mvMouseButton_Left):
            return                       # the release handler finishes up
        plot, which = self._dragging
        gx0, gy0, mx0, my0, ux, uy = self._drag_anchor
        gx, gy = dpg.get_mouse_pos(local=False)
        self._drag_probe_to(plot, which,
                            mx0 + (gx - gx0) * ux,
                            my0 + (gy - gy0) * uy)

    def _on_plot_hover(self, sender, app_data, user_data):
        """Draw an in-canvas bubble at the nearest data point, showing its
        x / y values with units. Fires while the plot is hovered. Also drives
        the manual probe drag (independent of ImPlot's native grab)."""
        plot = user_data
        if plot not in self.HOVER_FMT:
            return
        try:
            mx, my = dpg.get_plot_mouse_pos()
        except Exception:
            self._hide_hover_bubble()
            return
        down = dpg.is_mouse_button_down(dpg.mvMouseButton_Left)
        if down:
            # drag motion is driven by _tick_probe_drag (the plot-space mouse
            # pos FREEZES while the button is held, so it is useless here);
            # just keep the bubble out of the way
            self._hide_hover_bubble()
            self._hover_last = None
            return
        # button up: remember px<->plot-unit samples for the drag mapping
        gx, gy = dpg.get_mouse_pos(local=False)
        hist = self._hover_hist.setdefault(plot, [])
        if not hist or (abs(gx - hist[-1][0]) + abs(gy - hist[-1][1])) >= 3:
            hist.append((gx, gy, mx, my))
            if len(hist) > 24:
                del hist[0]
        # when the cursor sits on a probe, park the plot's pan so the next
        # press grabs the probe (not the canvas)
        near = self._probe_near(plot, mx, my)
        self._set_plot_pan_disabled(plot, near is not None)
        if near:
            self._hide_hover_bubble()
            self._hover_last = None
            self._tip_frame = dpg.get_frame_count()
            return
        xax, yax = self.PROBE_AXES[plot]
        x0, x1 = dpg.get_axis_limits(xax)
        y0, y1 = dpg.get_axis_limits(yax)
        xs_span = (x1 - x0) or 1.0
        ys_span = (y1 - y0) or 1.0
        # throttle: if the cursor barely moved since the last computed bubble,
        # keep the bubble alive but skip the (potentially heavy) search
        last = self._hover_last
        if (last and last[0] == plot
                and abs(mx - last[1]) < xs_span * 0.004
                and abs(my - last[2]) < ys_span * 0.004):
            self._tip_frame = dpg.get_frame_count()
            return
        self._hover_last = (plot, mx, my)
        best = None
        labels = set()
        for lbl, xs, ys in self._get_hover_data(plot):
            cl = self._clean_label(lbl)
            if cl:
                labels.add(cl)
            cand = self._nearest_on_series(xs, ys, mx, my, xs_span, ys_span)
            if cand is not None and (best is None or cand[0] < best[0]):
                best = (cand[0], cand[1], cand[2], lbl)
        if best is None or best[0] > 0.05 ** 2:
            self._hide_hover_bubble()
            return
        _, x, y, lbl = best
        xname, xunit, yname, yunit = self.HOVER_FMT[plot]
        # the two transient plots relabel per device profile (R/G vs Vth/P)
        ro = getattr(self, "_result_obs", None)
        if ro and plot in ("plot_r", "plot_g"):
            _, yname, yunit = ro[0 if plot == "plot_r" else 1]
        unit = dpg.get_value("unit_combo") if dpg.does_item_exist(
            "unit_combo") else ""
        xunit = xunit.replace("{unit}", unit)
        yunit = yunit.replace("{unit}", unit)
        # identify the curve (its legend label) when there are several or the
        # legend is hidden - so a hidden legend still tells you which is which
        cl = self._clean_label(lbl)
        head = (cl + "\n") if (cl and (not self.legends_visible
                                       or len(labels) > 1)) else ""
        text = (head + f"{yname} = {y:.4g} {yunit}".rstrip() + "\n"
                + f"{xname} = {x:.4g} {xunit}".rstrip())
        self._show_hover_bubble(plot, x, y, text)
        self._tip_frame = dpg.get_frame_count()

    def _show_hover_bubble(self, plot, x, y, text):
        # only one bubble visible at a time
        for p, aid in self._hover_ann.items():
            if p != plot and dpg.does_item_exist(aid):
                dpg.hide_item(aid)
        aid = self._hover_ann.get(plot)
        if aid and dpg.does_item_exist(aid):
            dpg.configure_item(aid, default_value=(x, y), label=text,
                               show=True)
        else:
            self._hover_ann[plot] = dpg.add_plot_annotation(
                parent=plot, label=text, default_value=(x, y),
                offset=(14, -14), color=(46, 86, 158, 235), clamped=True)

    # ---- A/B probe markers (Virtuoso-style) --------------------------

    PROBE_AXES = {
        "preview_plot": ("prev_x", "prev_y"),
        "plot_i": ("ax_i_x", "ax_i_y"),
        "plot_r": ("ax_r_x", "ax_r_y"),
        "plot_g": ("ax_g_x", "ax_g_y"),
        "ana_plot": ("ana_x", "ana_y"),
        "stdp_plot": ("stdp_x", "stdp_y"),
    }
    PLOT_NAMES = {"plot_i": "stimulus", "plot_r": "R", "plot_g": "G",
                  "preview_plot": "preview", "ana_plot": "G/pulse",
                  "stdp_plot": "STDP"}
    PROBE_READOUT = {"plot_i": "probe_results", "plot_r": "probe_results",
                     "plot_g": "probe_results",
                     "preview_plot": "probe_prev", "ana_plot": "probe_ana",
                     "stdp_plot": "probe_stdp"}
    PROBE_COLORS = {"A": (90, 190, 255, 255), "B": (255, 170, 90, 255)}

    def _nearest_point(self, plot, mx, my):
        """Closest data point on any series of the plot (axis-normalized)."""
        xax, yax = self.PROBE_AXES[plot]
        x0, x1 = dpg.get_axis_limits(xax)
        y0, y1 = dpg.get_axis_limits(yax)
        xs_span = (x1 - x0) or 1.0
        ys_span = (y1 - y0) or 1.0
        best = None
        for sid in dpg.get_item_children(yax, 1) or []:
            val = dpg.get_value(sid)
            if not val or len(val) < 2 or not val[0]:
                continue
            label = dpg.get_item_configuration(sid).get("label", "")
            if label.startswith("##"):       # probe ruler, not data
                continue
            for x, y in zip(val[0], val[1]):
                d = (((x - mx) / xs_span) ** 2
                     + ((y - my) / ys_span) ** 2)
                if best is None or d < best[0]:
                    best = (d, float(x), float(y), label, sid)
        if best is None:
            return None
        return best[1], best[2], best[3], best[4]

    def _set_probe(self, plot, which, x, y, slabel, sid):
        probes = self._probes.setdefault(plot, {})
        color = self.PROBE_COLORS[which]
        m = probes.get(which)
        if m:
            dpg.set_value(m["pt"], (x, y))
            if m.get("ann"):
                dpg.configure_item(m["ann"], default_value=(x, y))
        else:
            # no_inputs: the native ImPlot grab is disabled on purpose - it
            # takes ActiveId on press, which FREEZES the plot's mouse-position
            # updates and breaks the drag. The marker is visual-only; all
            # drag motion is driven manually in _on_plot_hover with live
            # get_plot_mouse_pos coordinates.
            pt = dpg.add_drag_point(parent=plot, default_value=(x, y),
                                    color=color, thickness=4,
                                    no_inputs=True)
            # the A / B label bubble that rides next to the marker
            ann = dpg.add_plot_annotation(parent=plot, label=which,
                                          default_value=(x, y),
                                          offset=(12, -12), color=color,
                                          clamped=True)
            probes[which] = m = {"pt": pt, "ann": ann}
        m.update(x=x, y=y, series=slabel, sid=sid)
        self._update_probe_readout(plot)

    def _on_probe_release(self, sender, app_data):
        """On mouse-up after a drag, snap the moved probe onto the nearest
        curve point."""
        d = self._dragging
        self._dragging = None
        self._drag_anchor = None
        if not d:
            return
        plot, which = d
        self._set_plot_pan_disabled(plot, False)
        m = self._probes.get(plot, {}).get(which)
        if not m:
            return
        xax, yax = self.PROBE_AXES[plot]
        x0, x1 = dpg.get_axis_limits(xax)
        y0, y1 = dpg.get_axis_limits(yax)
        xs_span = (x1 - x0) or 1.0
        ys_span = (y1 - y0) or 1.0
        best, best_lbl = None, None
        for lbl, xs, ys in self._get_hover_data(plot):
            cand = self._nearest_on_series(xs, ys, m["x"], m["y"],
                                           xs_span, ys_span)
            if cand is not None and (best is None or cand[0] < best[0]):
                best, best_lbl = cand, lbl
        if best is not None:
            _, x, y = best
            dpg.set_value(m["pt"], (x, y))
            if m.get("ann"):
                dpg.configure_item(m["ann"], default_value=(x, y))
            m["x"], m["y"], m["series"] = x, y, best_lbl
            self._update_probe_readout(plot)

    def _hovered_probe_plot(self):
        return next((p for p in self.PROBE_AXES
                     if dpg.does_item_exist(p) and dpg.is_item_hovered(p)),
                    None)

    def _arm_probe(self, which):
        self._probe_armed = which
        self._set_status(
            f"● probe {which} armed - click on a plot to place (Esc cancels)",
            self.PROBE_COLORS[which][:3])

    def _disarm_probe(self):
        if self._probe_armed is None:
            return
        self._probe_armed = None
        self._set_status("● ready", C_GREEN)

    def _on_probe_key(self, sender, app_data, user_data):
        # don't react while the user is typing somewhere
        for t in ("chat_input", "custom_text", "va_editor"):
            if dpg.does_item_exist(t) and dpg.is_item_active(t):
                return
        if user_data == "ESC":
            self._disarm_probe()
            return
        if user_data == "C":
            plot = self._hovered_probe_plot()
            if plot:
                self._clear_probes((plot,))
            self._disarm_probe()
            return
        # A / B: arm the probe; it is placed on the next mouse click
        self._arm_probe(user_data)

    def _on_probe_click(self, sender, app_data):
        # click a synapse on the trainer crossbar canvas -> inspect it
        if (dpg.does_item_exist("nt_diagram")
                and dpg.is_item_hovered("nt_diagram") and self._nt_cell_hits):
            self._nt_pick_cell()
            return
        plot = self._hovered_probe_plot()
        if not self._probe_armed:
            if not plot:
                return
            try:
                mx, my = dpg.get_plot_mouse_pos()
            except Exception:
                return
            # click on an existing probe = grab it (manual drag starts)
            which = self._probe_near(plot, mx, my)
            if which:
                gx, gy = dpg.get_mouse_pos(local=False)
                ux, uy = self._units_per_px(plot, gx, gy, mx, my)
                self._dragging = (plot, which)
                self._drag_anchor = (gx, gy, mx, my, ux, uy)
                return
            # un-armed click on the STDP curve = drill into that timing point
            if plot == "stdp_plot":
                self._stdp_drilldown()
            return
        if not plot:
            return                       # stays armed until a plot is clicked
        try:
            mx, my = dpg.get_plot_mouse_pos()
        except Exception:
            return
        hit = self._nearest_point(plot, mx, my)
        if not hit:
            return
        x, y, slabel, sid = hit
        which = self._probe_armed
        self._set_probe(plot, which, x, y, slabel, sid)
        self._disarm_probe()

    def _clear_probes(self, plots):
        for plot in plots:
            probes = self._probes.pop(plot, None)
            if not probes:
                continue
            for m in probes.values():
                for k in ("pt", "ann", "line"):
                    item = m.get(k)
                    if item and dpg.does_item_exist(item):
                        dpg.delete_item(item)
            tag = self.PROBE_READOUT.get(plot)
            if tag and dpg.does_item_exist(tag):
                dpg.set_value(tag, "")
                dpg.hide_item(tag)

    def _update_ruler(self, plot):
        """Straight line through A and B with dX / dY / dy/dx annotation."""
        probes = self._probes.get(plot, {})
        a, b = probes.get("A"), probes.get("B")
        ruler = probes.get("ruler")
        if not (a and b):
            if ruler:
                for k in ("line", "ann"):
                    if dpg.does_item_exist(ruler[k]):
                        dpg.delete_item(ruler[k])
                probes.pop("ruler", None)
            return
        xs, ys = [a["x"], b["x"]], [a["y"], b["y"]]
        dx, dy = b["x"] - a["x"], b["y"] - a["y"]
        slope = "inf" if dx == 0 else f"{dy / dx:.5g}"
        label = f"dX={dx:.5g}\ndY={dy:.5g}\ndy/dx={slope}"
        mid = (0.5 * (a["x"] + b["x"]), 0.5 * (a["y"] + b["y"]))
        if ruler:
            dpg.set_value(ruler["line"], [xs, ys])
            dpg.configure_item(ruler["ann"], default_value=mid, label=label)
        else:
            yax = self.PROBE_AXES[plot][1]
            line = dpg.add_line_series(xs, ys, label="##probe_ruler",
                                       parent=yax)
            dpg.bind_item_theme(line, self.themes["ruler_series"])
            ann = dpg.add_plot_annotation(parent=plot, label=label,
                                          default_value=mid,
                                          offset=(14, 14), clamped=True,
                                          color=(38, 44, 60, 230))
            probes["ruler"] = {"line": line, "ann": ann}

    def _update_probe_readout(self, plot):
        self._update_ruler(plot)
        tag = self.PROBE_READOUT.get(plot)
        if not tag or not dpg.does_item_exist(tag):
            return
        probes = self._probes.get(plot, {})
        a, b = probes.get("A"), probes.get("B")
        parts = []
        for which, m in (("A", a), ("B", b)):
            if m:
                s = f" [{m['series']}]" if m.get("series") else ""
                parts.append(f"{which}: x={m['x']:.6g}  y={m['y']:.6g}{s}")
        if a and b:
            dx, dy = b["x"] - a["x"], b["y"] - a["y"]
            slope = "inf" if dx == 0 else f"{dy / dx:.5g}"
            parts.append(f"dX={dx:.6g}  dY={dy:.6g}  dy/dx={slope}")
        name = self.PLOT_NAMES.get(plot, plot)
        dpg.set_value(tag, (f"[{name}]   " + "    |    ".join(parts))
                      if parts else "")
        dpg.configure_item(tag, show=bool(parts))

    def _on_zoom_btn(self, sender, app_data, user_data):
        axes, factor = user_data
        self._zoom_axes(axes, factor)

    def _on_fit_btn(self, sender, app_data, user_data):
        self._fit_axes_of(user_data)

    def _plot_toolbar(self, xaxes, yaxes, probe_tag=None):
        xaxes, yaxes = tuple(xaxes), tuple(yaxes)
        both = xaxes + yaxes
        with dpg.group(horizontal=True):
            dpg.add_button(label=" Zoom + ", small=True,
                           user_data=(both, 0.7),
                           callback=self._on_zoom_btn)
            dpg.add_button(label=" Zoom - ", small=True,
                           user_data=(both, 1.45),
                           callback=self._on_zoom_btn)
            dpg.add_button(label=" Fit ", small=True, user_data=both,
                           callback=self._on_fit_btn)
            options = dpg.add_button(label="View options", small=True)
            with self.desktop.popup(options):
                dpg.add_menu_item(label="Fit horizontal axis", user_data=xaxes, callback=self._on_fit_btn)
                dpg.add_menu_item(label="Fit vertical axis", user_data=yaxes, callback=self._on_fit_btn)
                dpg.add_separator()
                dpg.add_menu_item(label="Show / hide legends", callback=self._toggle_legends)
                dpg.add_menu_item(label="Move legends inside / outside", callback=self._toggle_legend_outside)
            help_button = dpg.add_button(label="?", small=True)
            with dpg.tooltip(help_button):
                dpg.add_text("Wheel: zoom | Drag: pan | Double click: fit\n"
                             "A / B then click: place probe | C: clear probes")
        if probe_tag:
            self._small("", tag=probe_tag, color=(126, 170, 255))
            dpg.hide_item(probe_tag)

    LEGEND_TAGS = ("preview_plot_leg", "plot_i_leg", "plot_r_leg", "plot_g_leg",
                   "ana_plot_leg", "stdp_plot_leg", "polar_plot_leg")

    def _toggle_legends(self, *_):
        self.legends_visible = not self.legends_visible
        for t in self.LEGEND_TAGS:
            if dpg.does_item_exist(t):
                dpg.configure_item(t, show=self.legends_visible)
        self._sync_leg_checks()                   # keep menu checkboxes in sync
        self.log(f"[plot] legends {'shown' if self.legends_visible else 'hidden'}"
                 + ("" if self.legends_visible else
                    " - hover a curve to see its label"))

    def _toggle_legend_outside(self, *_):
        self.legend_outside = not self.legend_outside
        for t in self.LEGEND_TAGS:
            if dpg.does_item_exist(t):
                dpg.configure_item(t, outside=self.legend_outside)
        self._sync_leg_checks()
        self.log(f"[plot] legend moved {'outside' if self.legend_outside else 'inside'}"
                 " the plot canvas")

    @staticmethod
    def _clean_label(lbl):
        """Strip the '##'/'##pts_' prefixes used to hide helper series."""
        s = (lbl or "").lstrip("#")
        if s.startswith("pts_"):
            s = s[4:]
        return s.strip()


    # short id -> ImPlot legend location constant / readable label
    _LEGEND_GRID = (
        ("NW", "mvPlot_Location_NorthWest"), ("N", "mvPlot_Location_North"),
        ("NE", "mvPlot_Location_NorthEast"),
        ("W", "mvPlot_Location_West"), ("C", "mvPlot_Location_Center"),
        ("E", "mvPlot_Location_East"),
        ("SW", "mvPlot_Location_SouthWest"), ("S", "mvPlot_Location_South"),
        ("SE", "mvPlot_Location_SouthEast"))
    _LOC_NAME = {"NW": "Top-left", "N": "Top", "NE": "Top-right",
                 "W": "Left", "C": "Center", "E": "Right",
                 "SW": "Bottom-left", "S": "Bottom", "SE": "Bottom-right"}

    def _add_plot_menu(self, plot, xaxis, yaxis, copy=False, x_unit=""):
        """Attach a native right-click context menu to `plot`. A plain WINDOW
        (reliable open/close, no auto-dismiss) of nested dpg.menu submenus that
        cascade on hover and keep the parent open. Opened by _on_plot_rclick,
        closed by _tick_menu_dismiss. copy=True adds the STDP copy-points items.
        Registering many plots gives every plot the same Fit/Zoom/Axis/Legend."""
        menu = plot + "_menu"
        with dpg.window(tag=menu, show=False, no_title_bar=True, no_resize=True,
                        no_move=True, no_collapse=True, no_scrollbar=True,
                        autosize=True):
            if copy:
                self._small("COPY POINTS", color=C_ACC)
                dpg.add_menu_item(label="dt  (delta-T) values", callback=lambda:
                                  self._menu_act(menu, lambda: self._copy_stdp("dt")))
                dpg.add_menu_item(label="State-change values (with units)", callback=lambda:
                                  self._menu_act(menu, lambda: self._copy_stdp("dg")))
                dpg.add_menu_item(label="dt, state-change pairs (CSV)", callback=lambda:
                                  self._menu_act(menu, lambda: self._copy_stdp("pairs")))
                dpg.add_separator()
            self._small("VIEW", color=C_ACC)
            dpg.add_menu_item(label="Fit all", callback=lambda: self._menu_act(
                menu, lambda: self._fit_axes_of((xaxis, yaxis))))
            dpg.add_menu_item(label="Fit X", callback=lambda: self._menu_act(
                menu, lambda: self._fit_axes_of((xaxis,))))
            dpg.add_menu_item(label="Fit Y", callback=lambda: self._menu_act(
                menu, lambda: self._fit_axes_of((yaxis,))))
            dpg.add_menu_item(label="Zoom in", callback=lambda: self._menu_act(
                menu, lambda: self._zoom_axes((xaxis, yaxis), 0.7)))
            dpg.add_menu_item(label="Zoom out", callback=lambda: self._menu_act(
                menu, lambda: self._zoom_axes((xaxis, yaxis), 1.45)))
            dpg.add_separator()
            with dpg.menu(label="Axis range"):
                self._build_axis_menu(menu, xaxis, yaxis, x_unit)
            with dpg.menu(label="Legend"):
                self._build_legend_menu(menu)
            dpg.add_separator()
            dpg.add_menu_item(label="Clear A / B probes", callback=lambda:
                              self._menu_act(menu, lambda: self._clear_probes((plot,))))
        if "stdp_menu" in self.themes:
            dpg.bind_item_theme(menu, self.themes["stdp_menu"])
        self._plot_menus.append({"plot": plot, "menu": menu,
                                 "xaxis": xaxis, "yaxis": yaxis})

    def _menu_act(self, menu, fn):
        """Run a one-shot menu action, then close that menu."""
        fn()
        if dpg.does_item_exist(menu):
            dpg.configure_item(menu, show=False)
        if self._open_menu == menu:
            self._open_menu = None

    def _on_plot_rclick(self, sender, app_data):
        """Right-click over any registered plot opens THAT plot's context menu at
        the cursor (ImPlot swallows the plot's own right-click). Only the plot on
        the active tab is hovered, so this naturally targets the right one."""
        for m in self._plot_menus:
            menu, plot = m["menu"], m["plot"]
            if dpg.does_item_exist(menu) and dpg.is_item_hovered(plot):
                self._prefill_axis(menu, m["xaxis"], m["yaxis"])
                mx, my = dpg.get_mouse_pos(local=False)
                vw = dpg.get_viewport_client_width()
                vh = dpg.get_viewport_client_height()
                px = max(0, min(int(mx), vw - 230))
                py = max(0, min(int(my), vh - 340))
                dpg.configure_item(menu, show=True, pos=[px, py])
                dpg.focus_item(menu)
                self._open_menu = menu
                self._menu_unfocused = 0
                return

    @staticmethod
    def _item_visible(tag):
        """is_item_visible that can't raise: a never-rendered item has no
        'visible' state key (is_item_visible would KeyError)."""
        if not dpg.does_item_exist(tag):
            return False
        try:
            return bool(dpg.get_item_state(tag).get("visible"))
        except Exception:
            return False

    def _tick_menu_dismiss(self):
        """Close the open context menu once neither it NOR an open submenu of it
        is focused / hovered (a short grace tolerates focus transitions). A
        submenu is detected by one of its rendered child widgets - the axis-range
        input and the legend checkbox - reporting 'visible'."""
        menu = self._open_menu
        if not menu or not dpg.does_item_exist(menu) \
                or not dpg.is_item_shown(menu):
            return
        submenu_open = (self._item_visible(menu + "_xmin")
                        or self._item_visible(menu + "_leg_show"))
        if (submenu_open or dpg.is_item_focused(menu)
                or dpg.is_item_hovered(menu)):
            self._menu_unfocused = 0
            return
        self._menu_unfocused += 1
        if self._menu_unfocused >= 4:
            dpg.configure_item(menu, show=False)
            self._open_menu = None

    def _prefill_axis(self, menu, xaxis, yaxis):
        """Seed the Axis-range inputs with the plot's CURRENT view, so 'Lock
        view' locks what's on screen (and the user edits from there)."""
        for axis, lo, hi in ((xaxis, "_xmin", "_xmax"), (yaxis, "_ymin", "_ymax")):
            try:
                lim = dpg.get_axis_limits(axis)
            except Exception:
                continue
            if lim and len(lim) == 2 and dpg.does_item_exist(menu + lo):
                dpg.set_value(menu + lo, round(float(lim[0]), 6))
                dpg.set_value(menu + hi, round(float(lim[1]), 6))

    def _build_axis_menu(self, menu, xaxis, yaxis, x_unit=""):
        """'Axis range' submenu: edit a range + Lock view (inputs/buttons don't
        close the submenu, so the user can set X and Y then lock)."""
        self._small("Lock the view to a range; Auto releases it.", color=C_MUTED)
        def _lock(*_):
            self._apply_view(menu, xaxis, yaxis)
        with dpg.group(horizontal=True):
            self._small("X")
            dpg.add_input_float(tag=menu + "_xmin", width=72, step=0,
                                format="%.4g", on_enter=True, callback=_lock)
            self._small("to")
            dpg.add_input_float(tag=menu + "_xmax", width=72, step=0,
                                format="%.4g", on_enter=True, callback=_lock)
            self._small(x_unit)
        with dpg.group(horizontal=True):
            self._small("Y")
            dpg.add_input_float(tag=menu + "_ymin", width=72, step=0,
                                format="%.4g", on_enter=True, callback=_lock)
            self._small("to")
            dpg.add_input_float(tag=menu + "_ymax", width=72, step=0,
                                format="%.4g", on_enter=True, callback=_lock)
            self._small("(0 to 0 = auto Y)", color=C_MUTED)
        with dpg.group(horizontal=True):
            dpg.add_button(label="Lock view", callback=_lock)
            dpg.add_button(label="Auto", callback=lambda *_:
                           self._auto_view(menu, xaxis, yaxis))

    def _apply_view(self, menu, xaxis, yaxis):
        """Lock the plot to the X/Y range in `menu`'s Axis-range inputs."""
        try:
            xlo = float(dpg.get_value(menu + "_xmin"))
            xhi = float(dpg.get_value(menu + "_xmax"))
            ylo = float(dpg.get_value(menu + "_ymin"))
            yhi = float(dpg.get_value(menu + "_ymax"))
        except (TypeError, ValueError):
            return
        if not xhi > xlo:
            self.log("[plot] Axis range: X 'to' must be greater than X 'from'")
            return
        self._zoom_anim.pop(xaxis, None)
        self._zoom_anim.pop(yaxis, None)
        dpg.set_axis_limits(xaxis, xlo, xhi)
        if yhi > ylo:
            dpg.set_axis_limits(yaxis, ylo, yhi)
        else:
            dpg.set_axis_limits_auto(yaxis)

    def _auto_view(self, menu, xaxis, yaxis):
        """Release the lock and fit `menu`'s plot to its data."""
        self._fit_axes_of((xaxis, yaxis))

    def _build_legend_menu(self, menu):
        """'Legend' submenu. Uses checkboxes (not check menu_items) so (a) they
        report a 'visible' state - which is how _tick_menu_dismiss knows this
        submenu is open - and (b) toggling one doesn't close the submenu, so the
        user can flip several at once. Position is a nested anchor submenu."""
        dpg.add_checkbox(label="Show legend", default_value=self.legends_visible,
                         tag=menu + "_leg_show", callback=self._toggle_legends)
        dpg.add_checkbox(label="Outside canvas", default_value=self.legend_outside,
                         tag=menu + "_leg_out", callback=self._toggle_legend_outside)
        dpg.add_checkbox(label="Horizontal layout",
                         default_value=self.legend_horizontal,
                         tag=menu + "_leg_horiz",
                         callback=self._toggle_legend_horizontal)
        with dpg.menu(label="Position"):
            for short, const in self._LEGEND_GRID:
                loc = getattr(dpg, const, 0)
                dpg.add_menu_item(label=self._LOC_NAME[short], user_data=loc,
                                  callback=self._on_legend_pos)

    def _sync_leg_checks(self):
        """Mirror the current legend state into every plot menu's checkboxes."""
        for m in self._plot_menus:
            for suf, val in ((("_leg_show"), self.legends_visible),
                             (("_leg_out"), self.legend_outside),
                             (("_leg_horiz"), self.legend_horizontal)):
                t = m["menu"] + suf
                if dpg.does_item_exist(t):
                    dpg.set_value(t, val)

    def _on_legend_pos(self, sender, app_data, loc):
        """Snap every legend to the picked anchor."""
        self.legend_location = loc
        for t in self.LEGEND_TAGS:
            if dpg.does_item_exist(t):
                dpg.configure_item(t, location=loc)

    def _toggle_legend_horizontal(self, *_):
        self.legend_horizontal = not self.legend_horizontal
        for t in self.LEGEND_TAGS:
            if dpg.does_item_exist(t):
                dpg.configure_item(t, horizontal=self.legend_horizontal)
        self._sync_leg_checks()
        self.log("[plot] legend layout "
                 + ("horizontal" if self.legend_horizontal else "vertical"))
