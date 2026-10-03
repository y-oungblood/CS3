"""The swipe deck: drag the card to "Seen it" (then rate) or "Haven't seen" (then list / pass)."""

from __future__ import annotations

import time

import flet as ft
import pandas as pd

from app import components as ui
from recsys import config

EXIT_MS = 220


class SwipeView:
    def __init__(self, shell):
        self.shell = shell
        self.ctl = shell.ctl
        self.step: str | None = None  # None | "seen" | "unseen"
        self.card_box: ft.Container | None = None
        self.zones: dict[str, ft.Container] = {}
        self.busy = False

    # ------------------------------------------------------------ layout

    def build(self) -> ft.Control:
        narrow = self.shell.is_narrow
        width = self.shell.width(640)
        movie_id = self.ctl.deal()
        if movie_id is None:
            return self._exhausted(width)
        m = self.ctl.movie(movie_id)
        card_w, card_h = (190, 285) if narrow else (250, 375)
        zone_w = max(56, min(130, (width - card_w) // 2 - 12))

        return ft.Container(
            width=width,
            padding=ft.Padding.only(top=16, bottom=8),
            content=ft.Column(
                [
                    self._header(),
                    self._progress(),
                    ft.Container(height=8),
                    ft.Row(
                        [
                            self._zone("unseen", "Haven't seen", ft.Icons.VISIBILITY_OFF_OUTLINED, ui.UNSEEN, zone_w, card_h),
                            self._card(m, card_w, card_h),
                            self._zone("seen", "Seen it", ft.Icons.VISIBILITY_OUTLINED, ui.SEEN, zone_w, card_h),
                        ],
                        alignment=ft.MainAxisAlignment.CENTER,
                        vertical_alignment=ft.CrossAxisAlignment.CENTER,
                        spacing=12,
                    ),  # fmt: skip
                    ft.Text(m["title"], size=20, weight=ft.FontWeight.BOLD, text_align=ft.TextAlign.CENTER),
                    ft.Text(
                        ui.meta_line(None if pd.isna(m["year"]) else int(m["year"]), list(m["genres"])),
                        color=ft.Colors.ON_SURFACE_VARIANT, text_align=ft.TextAlign.CENTER,
                    ),  # fmt: skip
                    ft.Container(height=4),
                    self._step_panel(width),
                    ft.Text(
                        "Drag the card, or use the buttons. Keys: ← haven't seen · → seen it · 1-5 rate · "
                        "A add to list · N not interested · Esc back · U undo",
                        size=11, color=ui.MUTED, text_align=ft.TextAlign.CENTER, visible=not narrow,
                    ),  # fmt: skip
                ],
                horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                spacing=6,
            ),
        )

    def _header(self) -> ft.Control:
        if self.ctl.unlocked:
            recs = ft.FilledTonalButton(
                "My recommendations", icon=ft.Icons.AUTO_AWESOME, on_click=lambda e: self.shell.go("results")
            )
        else:
            n = self.ctl.remaining_to_unlock
            recs = ft.Text(f"{n} more answer{'s' if n != 1 else ''} to unlock recommendations", size=13, color=ui.MUTED)
        return ft.Row(
            [
                ft.Row(
                    [ft.Icon(ft.Icons.ACCOUNT_CIRCLE_OUTLINED, size=18, color=ui.MUTED),
                     ft.Text(self.ctl.user.handle, size=13, color=ft.Colors.ON_SURFACE_VARIANT, selectable=True)],
                    spacing=4, tight=True,
                ),  # fmt: skip
                recs,
            ],
            alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
            wrap=True,
        )

    def _progress(self) -> ft.Control:
        n = self.ctl.n_informative
        return ft.Column(
            [
                ft.Row(
                    [ft.Text("Taste profile", size=12, weight=ft.FontWeight.W_600),
                     ft.Text(f"{min(n, config.PROGRESS_FULL)}/{config.PROGRESS_FULL}", size=12, color=ui.MUTED)],
                    alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                ),  # fmt: skip
                ft.ProgressBar(value=self.ctl.progress, bar_height=8, border_radius=ft.BorderRadius.all(4), color=ui.ACCENT),
            ],
            spacing=4,
        )

    def _card(self, m: pd.Series, w: int, h: int) -> ft.Control:
        url = None if pd.isna(m["poster_url"]) else m["poster_url"]
        self.card_box = ft.Container(
            content=ui.poster(url, m["title"], w, h, radius=16),
            border_radius=ft.BorderRadius.all(16),
            shadow=ft.BoxShadow(blur_radius=18, spread_radius=1, color=ft.Colors.with_opacity(0.25, ft.Colors.BLACK)),
            animate_offset=ft.Animation(EXIT_MS, ft.AnimationCurve.EASE_IN),
            animate_opacity=ft.Animation(EXIT_MS, ft.AnimationCurve.EASE_IN),
            offset=ft.Offset(0, 0),
        )
        ghost = ft.Container(
            width=w, height=h, border_radius=ft.BorderRadius.all(16),
            border=ft.Border.all(2, ft.Colors.OUTLINE_VARIANT),
        )  # fmt: skip
        return ft.Draggable(
            group="card",
            content=self.card_box,
            content_when_dragging=ghost,
            content_feedback=ft.Container(
                content=ui.poster(url, m["title"], w * 0.85, h * 0.85, radius=16),
                opacity=0.9,
                rotate=ft.Rotate(0.05),
            ),
        )

    def _zone(self, key: str, label: str, icon, color: str, w: int, h: int) -> ft.Control:
        box = ft.Container(
            width=w,
            height=h,
            border_radius=ft.BorderRadius.all(16),
            border=ft.Border.all(2, ft.Colors.with_opacity(0.5, color)),
            bgcolor=ft.Colors.with_opacity(0.06, color),
            alignment=ft.Alignment.CENTER,
            content=ft.Column(
                [ft.Icon(icon, color=color, size=28),
                 ft.Text(label, color=color, weight=ft.FontWeight.BOLD, size=13, text_align=ft.TextAlign.CENTER)],
                horizontal_alignment=ft.CrossAxisAlignment.CENTER, alignment=ft.MainAxisAlignment.CENTER, tight=True,
            ),  # fmt: skip
        )
        self.zones[key] = box

        def highlight(on: bool):
            box.bgcolor = ft.Colors.with_opacity(0.22 if on else 0.06, color)
            box.update()

        return ft.DragTarget(
            group="card",
            content=box,
            on_will_accept=lambda e: highlight(True),
            on_leave=lambda e: highlight(False),
            on_accept=lambda e: (highlight(False), self._open_step(key)),
        )

    def _step_panel(self, width: int) -> ft.Control:
        if self.step == "seen":
            body = [
                ft.Text("How was it?", weight=ft.FontWeight.BOLD),
                ui.star_picker(lambda n: self._record("rated", n), size=40),
                ft.TextButton("Back", icon=ft.Icons.ARROW_BACK, on_click=lambda e: self._open_step(None)),
            ]
        elif self.step == "unseen":
            body = [
                ft.Text("Want to watch it?", weight=ft.FontWeight.BOLD),
                ft.Row(
                    [ft.FilledButton("Add to my list", icon=ft.Icons.BOOKMARK_ADD_OUTLINED,
                                     on_click=lambda e: self._record("interested")),
                     ft.OutlinedButton("Not interested", icon=ft.Icons.THUMB_DOWN_OUTLINED,
                                       on_click=lambda e: self._record("not_interested"))],
                    alignment=ft.MainAxisAlignment.CENTER, wrap=True,
                ),  # fmt: skip
                ft.TextButton("Back", icon=ft.Icons.ARROW_BACK, on_click=lambda e: self._open_step(None)),
            ]
        else:
            body = [
                ft.Row(
                    [
                        ft.OutlinedButton("Haven't seen", icon=ft.Icons.ARROW_BACK, on_click=lambda e: self._open_step("unseen")),
                        ft.IconButton(ft.Icons.UNDO, tooltip="Undo last answer", disabled=not self.ctl.can_undo,
                                      on_click=lambda e: self._undo()),
                        ft.FilledButton("Seen it", icon=ft.Icons.ARROW_FORWARD, on_click=lambda e: self._open_step("seen")),
                    ],
                    alignment=ft.MainAxisAlignment.CENTER,
                )  # fmt: skip
            ]
        return ft.Container(
            width=min(width, 420),
            padding=ft.Padding.all(8),
            content=ft.Column(body, horizontal_alignment=ft.CrossAxisAlignment.CENTER, spacing=6, tight=True),
        )

    def _exhausted(self, width: int) -> ft.Control:
        return ft.Container(
            width=width,
            padding=ft.Padding.only(top=48),
            content=ft.Column(
                [ui.section_title("You've been through the whole deck!"),
                 ft.FilledButton("See my recommendations", on_click=lambda e: self.shell.go("results"))],
                horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            ),  # fmt: skip
        )

    # ------------------------------------------------------------ actions

    def _open_step(self, step: str | None) -> None:
        self.step = step
        self.shell.render()

    def _record(self, kind: str, rating: int | None = None) -> None:
        if self.busy:
            return
        self.busy = True
        try:
            if self.card_box is not None:  # fly the card toward the zone it went to
                direction = 1 if kind == "rated" else -1
                self.card_box.offset = ft.Offset(1.6 * direction, -0.05)
                self.card_box.opacity = 0
                self.card_box.update()
                time.sleep(EXIT_MS / 1000)
            trigger = self.ctl.answer_card(kind, rating)
            self.step = None
        finally:
            self.busy = False
        self.shell.render()  # show the updated deck (and progress) behind any check-in dialog
        if trigger:
            self._checkin(trigger)

    def _undo(self) -> None:
        if self.ctl.undo() is not None:
            self.step = None
            self.shell.render()

    def _checkin(self, trigger: str) -> None:
        def go(_):
            self.shell.page.pop_dialog()
            self.shell.go("results", trigger=trigger)

        n = trigger.split("_")[1]
        self.shell.page.show_dialog(
            ft.AlertDialog(
                modal=True,
                icon=ft.Icon(ft.Icons.AUTO_AWESOME, color=ui.ACCENT),
                title=ft.Text("Here's what we think so far"),
                content=ft.Text(
                    f"After {n} answers we have a first read on your taste. How did we do? "
                    "Rate or save a few picks on the next page; it helps us (and the study)."
                ),
                actions=[ft.FilledButton("Show me", on_click=go)],
            )
        )

    def on_key(self, e: ft.KeyboardEvent) -> None:
        key = (e.key or "").lower()
        if self.step is None:
            if key == "arrow left":
                self._open_step("unseen")
            elif key == "arrow right":
                self._open_step("seen")
            elif key == "u":
                self._undo()
        elif self.step == "seen" and key in {"1", "2", "3", "4", "5"}:
            self._record("rated", int(key))
        elif self.step == "unseen" and key in {"a", "n"}:
            self._record("interested" if key == "a" else "not_interested")
        if key == "escape" and self.step is not None:
            self._open_step(None)
