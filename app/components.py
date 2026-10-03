"""Shared UI pieces: theme, posters, stars, attribution."""

from __future__ import annotations

from collections.abc import Callable

import flet as ft

ACCENT = "#2a78d6"
SEEN = "#1baf7a"     # "Seen it" zone
UNSEEN = "#eb6834"   # "Haven't seen" zone
MUTED = "#898781"
STAR = "#eda100"

TMDB_NOTICE = "Movie posters and runtimes from TMDB. This product uses the TMDB API but is not endorsed or certified by TMDB."


def theme() -> ft.Theme:
    return ft.Theme(color_scheme_seed=ACCENT)


def dark_theme() -> ft.Theme:
    return ft.Theme(color_scheme_seed=ACCENT)  # page.dark_theme derives the dark palette


def poster(url: str | None, title: str, width: float, height: float, radius: float = 12) -> ft.Control:
    """A poster image, or a titled placeholder when TMDB has no poster."""
    placeholder = ft.Container(
        width=width,
        height=height,
        border_radius=ft.BorderRadius.all(radius),
        bgcolor=ft.Colors.SURFACE_CONTAINER_HIGHEST,
        alignment=ft.Alignment.CENTER,
        padding=ft.Padding.all(8),
        content=ft.Column(
            [
                ft.Icon(ft.Icons.MOVIE_OUTLINED, color=MUTED, size=min(48, width / 3)),
                ft.Text(title, text_align=ft.TextAlign.CENTER, size=max(10, min(16, width / 12)),
                        color=ft.Colors.ON_SURFACE_VARIANT, max_lines=4, overflow=ft.TextOverflow.ELLIPSIS),
            ],
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            alignment=ft.MainAxisAlignment.CENTER,
            tight=True,
        ),
    )  # fmt: skip
    if not url:
        return placeholder
    return ft.Image(
        src=url,
        width=width,
        height=height,
        fit=ft.BoxFit.COVER,
        border_radius=ft.BorderRadius.all(radius),
        error_content=placeholder,
        semantics_label=f"Poster for {title}",
    )


def star_picker(on_pick: Callable[[int], None], size: int = 36) -> ft.Row:
    """Five stars; hovering previews, clicking picks."""
    buttons: list[ft.IconButton] = []

    def preview(n: int) -> None:
        for i, b in enumerate(buttons, 1):
            b.icon = ft.Icons.STAR if i <= n else ft.Icons.STAR_BORDER
        row.update()

    for n in range(1, 6):
        buttons.append(
            ft.IconButton(
                icon=ft.Icons.STAR_BORDER,
                icon_color=STAR,
                icon_size=size,
                tooltip=f"{n} star{'s' if n > 1 else ''}",
                on_click=lambda e, n=n: on_pick(n),
                on_hover=lambda e, n=n: preview(n if e.data in (True, "true") else 0),
            )
        )
    # Semantics gives each star an accessible name for screen readers.
    labeled = [ft.Semantics(content=b, label=f"{n} star{'s' if n > 1 else ''}", button=True) for n, b in enumerate(buttons, 1)]
    row = ft.Row(labeled, alignment=ft.MainAxisAlignment.CENTER, spacing=0, tight=True)
    return row


def footer() -> ft.Control:
    return ft.Container(
        padding=ft.Padding.symmetric(horizontal=16, vertical=12),
        content=ft.Text(
            spans=[
                ft.TextSpan(TMDB_NOTICE + " "),
                ft.TextSpan("themoviedb.org", url="https://www.themoviedb.org",
                            style=ft.TextStyle(decoration=ft.TextDecoration.UNDERLINE)),
                ft.TextSpan(" · Ratings data: MovieLens (GroupLens)."),
            ],
            size=11,
            color=MUTED,
            text_align=ft.TextAlign.CENTER,
        ),  # fmt: skip
        alignment=ft.Alignment.CENTER,
    )


def section_title(text: str, size: int = 18) -> ft.Text:
    return ft.Text(text, size=size, weight=ft.FontWeight.BOLD)


def meta_line(year: int | None, genres: list[str], n_genres: int = 3) -> str:
    parts = [str(year)] if year else []
    if genres:
        parts.append(" · ".join(genres[:n_genres]))
    return " · ".join(parts)
