"""Recommendations: ten explained picks, the adventurousness dial, and the watchlist."""

from __future__ import annotations

import flet as ft

from app import components as ui
from app.controller import RecCard
from recsys import config
from recsys.timing import ANYTIME, TONIGHT, WEEKEND

ANSWER_LABELS = {"interested": "On your list", "not_interested": "Not interested"}


class ResultsView:
    def __init__(self, shell, trigger: str = "voluntary"):
        self.shell = shell
        self.ctl = shell.ctl
        self.trigger = trigger
        self.rating_open: int | None = None  # movie_id whose star picker is showing
        self.cards = self.ctl.open_recommendations(trigger)
        self.list_col = ft.Column(spacing=10)

    def build(self) -> ft.Control:
        width = self.shell.width(760)
        self._fill_list()
        intro = (
            "Here's what we think so far. How did we do? Rate the ones you've seen and save the ones you want to watch."
            if self.trigger.startswith("checkin")
            else "Rate the ones you've seen and save the ones you want to watch. Every answer sharpens the next list."
        )
        return ft.Container(
            width=width,
            padding=ft.Padding.only(top=20, bottom=8),
            content=ft.Column(
                [
                    ft.Row(
                        [ui.section_title("Your recommendations", 24),
                         ft.TextButton("Keep swiping", icon=ft.Icons.STYLE_OUTLINED, on_click=lambda e: self.shell.go("swipe"))],
                        alignment=ft.MainAxisAlignment.SPACE_BETWEEN, wrap=True,
                    ),  # fmt: skip
                    ft.Text(intro, color=ft.Colors.ON_SURFACE_VARIANT),
                    self._dial(),
                    self.list_col,
                    ft.Row(
                        [ft.OutlinedButton("Refresh", icon=ft.Icons.REFRESH, on_click=self._refresh),
                         ft.FilledButton("Keep swiping", icon=ft.Icons.STYLE_OUTLINED, on_click=lambda e: self.shell.go("swipe"))],
                        alignment=ft.MainAxisAlignment.CENTER,
                    ),  # fmt: skip
                    ft.Divider(height=32),
                    self._watchlist(),
                ],
                spacing=12,
            ),
        )

    # ------------------------------------------------------------ dial

    def _dial(self) -> ft.Control:
        slider = ft.Slider(
            value=self.ctl.lam,
            min=0,
            max=config.LAMBDA_MAX,
            divisions=6,
            expand=True,
            active_color=ui.ACCENT,
            on_change_end=self._on_dial,  # fires once, where the user lets go
        )
        return ft.Container(
            padding=ft.Padding.symmetric(horizontal=12, vertical=4),
            border_radius=ft.BorderRadius.all(12),
            bgcolor=ft.Colors.SURFACE_CONTAINER_LOW,
            content=ft.Row(
                [ft.Text("Familiar", size=13, weight=ft.FontWeight.W_600), slider,
                 ft.Text("Adventurous", size=13, weight=ft.FontWeight.W_600)],
                vertical_alignment=ft.CrossAxisAlignment.CENTER,
            ),  # fmt: skip
        )

    def _on_dial(self, e) -> None:
        self.cards = self.ctl.set_lambda(float(e.control.value))
        self.rating_open = None
        self._fill_list()
        self.shell.page.update()

    def _refresh(self, e) -> None:
        self.cards = self.ctl.refresh()
        self.rating_open = None
        self._fill_list()
        self.shell.page.update()

    # ------------------------------------------------------------ cards

    def _fill_list(self) -> None:
        if not self.cards:
            self.list_col.controls = [ft.Text("No recommendations right now. Swipe a few more cards.", color=ui.MUTED)]
            return
        self.list_col.controls = [self._rec_card(c) for c in self.cards]

    def _rec_card(self, c: RecCard) -> ft.Control:
        narrow = self.shell.is_narrow
        pw, ph = (72, 108) if narrow else (96, 144)
        details = [
            ft.Text(f"{c.rank}. {c.title}", size=16, weight=ft.FontWeight.BOLD, max_lines=2, overflow=ft.TextOverflow.ELLIPSIS),
            ft.Text(ui.meta_line(c.year, c.genres), size=12, color=ft.Colors.ON_SURFACE_VARIANT),
            ft.Text(c.explanation, size=13, italic=True),
        ]
        if c.tags:
            details.append(ft.Text(c.tags, size=12, color=ui.MUTED))
        if c.when:
            details.append(self._when_row(c))
        if not narrow:
            details.append(self._answer_row(c))
        top = ft.Row(
            [ui.poster(c.poster_url, c.title, pw, ph, radius=8), ft.Column(details, spacing=4, expand=True)],
            vertical_alignment=ft.CrossAxisAlignment.START,
            spacing=12,
        )
        body = ft.Column([top, self._answer_row(c)], spacing=8) if narrow else top
        return ft.Card(
            opacity=0.55 if c.answer else 1.0,
            content=ft.Container(padding=ft.Padding.all(10), content=body),
        )

    def _when_row(self, c: RecCard) -> ft.Control:
        def vote(up: bool):
            self.ctl.vote_timing(c.movie_id, up)
            self._fill_list()
            self.shell.page.update()

        return ft.Row(
            [
                ft.Icon(ft.Icons.SCHEDULE, size=16, color=ui.ACCENT),
                # On phones the text may wrap so the thumbs stay on screen; on desktop they sit beside it.
                ft.Text(c.when, size=13, color=ui.ACCENT, weight=ft.FontWeight.W_600, expand=self.shell.is_narrow),
                ft.IconButton(
                    ft.Icons.THUMB_UP if c.timing_vote == "up" else ft.Icons.THUMB_UP_OUTLINED,
                    icon_size=16, tooltip="Good suggestion", on_click=lambda e: vote(True),
                    visual_density=ft.VisualDensity.COMPACT, padding=ft.Padding.all(4),
                ),
                ft.IconButton(
                    ft.Icons.THUMB_DOWN if c.timing_vote == "down" else ft.Icons.THUMB_DOWN_OUTLINED,
                    icon_size=16, tooltip="Not for me", on_click=lambda e: vote(False),
                    visual_density=ft.VisualDensity.COMPACT, padding=ft.Padding.all(4),
                ),
            ],
            spacing=0,
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
        )  # fmt: skip

    def _answer_row(self, c: RecCard) -> ft.Control:
        if c.answer is not None:
            a = c.answer
            label = f"You rated it {'★' * a.rating}" if a.kind == "rated" else ANSWER_LABELS[a.kind]
            return ft.Row([ft.Icon(ft.Icons.CHECK_CIRCLE, size=16, color=ui.SEEN), ft.Text(label, size=13)], spacing=4)
        if self.rating_open == c.movie_id:
            return ft.Row(
                [ui.star_picker(lambda n, mid=c.movie_id: self._answer(mid, "rated", n), size=28),
                 ft.TextButton("Back", on_click=lambda e: self._open_rating(None))],
                wrap=True, spacing=0,
            )  # fmt: skip
        compact = ft.ButtonStyle(padding=ft.Padding.symmetric(horizontal=10), visual_density=ft.VisualDensity.COMPACT)
        return ft.Row(
            [
                ft.OutlinedButton("Seen it", icon=ft.Icons.VISIBILITY_OUTLINED, style=compact,
                                  on_click=lambda e: self._open_rating(c.movie_id)),
                ft.OutlinedButton("Add to my list", icon=ft.Icons.BOOKMARK_ADD_OUTLINED, style=compact,
                                  on_click=lambda e: self._answer(c.movie_id, "interested")),
                ft.TextButton("Not interested", style=compact,
                              on_click=lambda e: self._answer(c.movie_id, "not_interested")),
            ],
            wrap=True, spacing=4, run_spacing=4,
        )  # fmt: skip

    def _open_rating(self, movie_id: int | None) -> None:
        self.rating_open = movie_id
        self._fill_list()
        self.shell.page.update()

    def _answer(self, movie_id: int, kind: str, rating: int | None = None) -> None:
        self.ctl.answer_recommendation(movie_id, kind, rating)
        self.rating_open = None
        self._fill_list()
        self.watch_box.content = self._watchlist_body()
        self.shell.page.update()

    # ------------------------------------------------------------ watchlist

    def _watchlist(self) -> ft.Control:
        self.watch_box = ft.Container(content=self._watchlist_body())
        return self.watch_box

    def _watchlist_body(self) -> ft.Control:
        groups = self.ctl.watchlist_groups()
        total = sum(len(v) for v in groups.values())
        rows: list[ft.Control] = [ui.section_title(f"My list ({total})")]
        if total == 0:
            rows.append(ft.Text('Movies you "Add to my list" show up here, grouped by when to watch them.', color=ui.MUTED))
        icons = {TONIGHT: ft.Icons.NIGHTLIGHT_OUTLINED, WEEKEND: ft.Icons.WEEKEND_OUTLINED, ANYTIME: ft.Icons.SCHEDULE}
        for name in (TONIGHT, WEEKEND, ANYTIME):
            if not groups[name]:
                continue
            rows.append(ft.Row([ft.Icon(icons[name], size=18, color=ui.ACCENT), ft.Text(name, weight=ft.FontWeight.BOLD)], spacing=6))
            rows.append(
                ft.Row(
                    [self._watch_item(m) for m in groups[name]],
                    scroll=ft.ScrollMode.AUTO, spacing=10,
                )  # fmt: skip
            )
        return ft.Column(rows, spacing=8)

    @staticmethod
    def _watch_item(m: dict) -> ft.Control:
        return ft.Column(
            [ui.poster(m["poster_url"], m["title"], 80, 120, radius=8),
             ft.Text(m["title"], size=11, width=80, max_lines=2, overflow=ft.TextOverflow.ELLIPSIS)],
            spacing=4, tight=True,
        )  # fmt: skip
