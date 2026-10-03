"""Welcome: start fresh with a generated handle, or come back with one."""

from __future__ import annotations

import flet as ft

from app import components as ui


class WelcomeView:
    def __init__(self, shell):
        self.shell = shell
        self.ctl = shell.ctl
        self.handle_field = ft.TextField(
            label="Your handle",
            hint_text="e.g. brave-otter-42",
            on_submit=self._returning,
            expand=True,
        )
        self.error = ft.Text("", color=ft.Colors.ERROR, size=13, visible=False)

    def build(self) -> ft.Control:
        w = self.shell.width(480)
        return ft.Container(
            width=w,
            padding=ft.Padding.only(top=48 if not self.shell.is_narrow else 24, bottom=16),
            content=ft.Column(
                [
                    ft.Icon(ft.Icons.MOVIE_FILTER_OUTLINED, size=56, color=ui.ACCENT),
                    ft.Text("Movie Swipe", size=36, weight=ft.FontWeight.BOLD),
                    ft.Text(
                        "Swipe through a few movies, and get recommendations that explain themselves "
                        "and tell you when to watch them.",
                        size=15, text_align=ft.TextAlign.CENTER, color=ft.Colors.ON_SURFACE_VARIANT,
                    ),  # fmt: skip
                    ft.Container(height=16),
                    self._card(
                        "New here?",
                        "No sign-up and no email. You'll get a random handle to come back with.",
                        ft.FilledButton("Get started", icon=ft.Icons.ARROW_FORWARD, on_click=self._new, width=w - 48),
                    ),
                    self._card(
                        "Returning?",
                        "Enter your handle to pick up where you left off.",
                        ft.Column(
                            [
                                ft.Row([self.handle_field]),
                                self.error,
                                ft.OutlinedButton("Continue", on_click=self._returning, width=w - 48),
                            ],
                            tight=True,
                        ),
                    ),
                ],
                horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                spacing=10,
            ),
        )

    @staticmethod
    def _card(title: str, subtitle: str, body: ft.Control) -> ft.Control:
        return ft.Card(
            content=ft.Container(
                padding=ft.Padding.all(20),
                content=ft.Column(
                    [ft.Text(title, size=18, weight=ft.FontWeight.BOLD),
                     ft.Text(subtitle, size=13, color=ft.Colors.ON_SURFACE_VARIANT), body],
                    spacing=10, tight=True,
                ),  # fmt: skip
            ),
        )

    def _new(self, e) -> None:
        user = self.ctl.start_new()

        def proceed(_):
            self.shell.page.pop_dialog()
            self.shell.go("onboarding")

        self.shell.page.show_dialog(
            ft.AlertDialog(
                modal=True,
                title=ft.Text("Your handle"),
                content=ft.Column(
                    [
                        ft.Text("Save this to come back later. It's the only way to find your profile again."),
                        ft.Container(
                            padding=ft.Padding.symmetric(vertical=12, horizontal=16),
                            border_radius=ft.BorderRadius.all(8),
                            bgcolor=ft.Colors.SURFACE_CONTAINER_HIGHEST,
                            content=ft.Text(user.handle, size=22, weight=ft.FontWeight.BOLD, selectable=True),
                        ),
                    ],
                    tight=True,
                    spacing=12,
                ),
                actions=[ft.FilledButton("I've saved it", on_click=proceed)],
            )
        )

    def _returning(self, e) -> None:
        text = (self.handle_field.value or "").strip()
        if not text:
            return self._show_error("Enter the handle you were given, like brave-otter-42.")
        if not self.ctl.login(text):
            return self._show_error("We couldn't find that handle. Check the spelling, or start fresh above.")
        self.shell.go("onboarding" if self.ctl.needs_seeds else "swipe")

    def _show_error(self, message: str) -> None:
        self.error.value = message
        self.error.visible = True
        self.shell.page.update()
