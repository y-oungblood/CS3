"""Seed picks: search for 3-5 movies the user loves."""

from __future__ import annotations

import flet as ft
import pandas as pd

from app import components as ui
from app.controller import MAX_SEEDS, MIN_SEEDS


class OnboardingView:
    def __init__(self, shell):
        self.shell = shell
        self.ctl = shell.ctl
        self.picks: list[dict] = []
        self.search = ft.TextField(
            hint_text="Search for a movie title",
            prefix_icon=ft.Icons.SEARCH,
            autofocus=True,
            on_change=self._on_search,
        )
        self.results = ft.Column(spacing=2, tight=True)
        self.chips = ft.Row(wrap=True, spacing=8, run_spacing=8)
        self.status = ft.Text("", size=13, color=ft.Colors.ON_SURFACE_VARIANT)
        self.continue_btn = ft.FilledButton("Continue", icon=ft.Icons.ARROW_FORWARD, on_click=self._continue)

    def build(self) -> ft.Control:
        self._refresh_picks(update=False)
        width = self.shell.width(560)
        self.search.width = width
        return ft.Container(
            width=width,
            padding=ft.Padding.only(top=32, bottom=16),
            content=ft.Column(
                [
                    ui.section_title("Pick 3 to 5 movies you love", 24),
                    ft.Text("They give the recommender a head start. You'll rate more on the next screen.",
                            color=ft.Colors.ON_SURFACE_VARIANT),  # fmt: skip
                    self.search,
                    self.results,
                    ft.Divider(),
                    self.chips,
                    ft.Row([self.status, self.continue_btn], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
                ],
                spacing=12,
            ),
        )

    def _on_search(self, e) -> None:
        query = self.search.value or ""
        rows: list[ft.Control] = []
        if len(query.strip()) >= 2:
            chosen = {p["movie_id"] for p in self.picks}
            hits = self.ctl.search(query)
            hits = hits[~hits["movie_id"].isin(chosen)]
            if hits.empty:
                rows.append(ft.Text("No matches. Try another title.", color=ui.MUTED))
            rows += [self._result_row(m) for _, m in hits.iterrows()]
        if (self.search.value or "") != query:
            return  # the user kept typing; a newer search will draw the results
        self.results.controls = rows
        self.shell.page.update()

    def _result_row(self, m: pd.Series) -> ft.Control:
        year = None if pd.isna(m["year"]) else int(m["year"])
        url = None if pd.isna(m["poster_url"]) else m["poster_url"]
        return ft.ListTile(
            leading=ui.poster(url, m["title"], 36, 54, radius=4),
            title=ft.Text(m["title"], max_lines=1, overflow=ft.TextOverflow.ELLIPSIS),
            subtitle=ft.Text(ui.meta_line(year, list(m["genres"]), 2), size=12),
            on_click=lambda e, m=m, year=year, url=url: self._add(int(m["movie_id"]), m["title"], year, url),
            dense=True,
        )

    def _add(self, movie_id: int, title: str, year, url) -> None:
        if len(self.picks) >= MAX_SEEDS:
            self.shell.toast(f"That's {MAX_SEEDS}. Remove one to swap it out.")
            return
        self.picks.append({"movie_id": movie_id, "title": title, "year": year, "poster_url": url})
        self.search.value = ""
        self.results.controls.clear()
        self._refresh_picks()

    def _remove(self, movie_id: int) -> None:
        self.picks = [p for p in self.picks if p["movie_id"] != movie_id]
        self._refresh_picks()

    def _refresh_picks(self, update: bool = True) -> None:
        self.chips.controls = [
            ft.Chip(
                label=ft.Text(f"{p['title']} ({p['year']})" if p["year"] else p["title"]),
                leading=ui.poster(p["poster_url"], p["title"], 20, 30, radius=3),
                on_delete=lambda e, mid=p["movie_id"]: self._remove(mid),
            )
            for p in self.picks
        ]
        n = len(self.picks)
        self.status.value = (
            f"{n} picked. Pick {MIN_SEEDS - n} more." if n < MIN_SEEDS else f"{n} picked. Add up to {MAX_SEEDS - n} more, or continue."
        ) if n < MAX_SEEDS else f"{n} picked."  # fmt: skip
        self.continue_btn.disabled = n < MIN_SEEDS
        if update:
            self.shell.page.update()

    def _continue(self, e) -> None:
        self.ctl.save_seeds([p["movie_id"] for p in self.picks])
        self.shell.go("swipe")
