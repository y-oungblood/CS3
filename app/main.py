"""Movie Swipe app entry point.

    python app/main.py            # desktop window (local development)
    python app/main.py --web      # web server at http://localhost:8550 (or $PORT)

Environment: STORAGE_BACKEND (sqlite | firestore), PORT, HOST, and
FLET_FORCE_WEB_SERVER=1 to serve without opening a browser (used in the container).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import flet as ft

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app import components as ui  # noqa: E402
from app.controller import Controller  # noqa: E402
from app.views.onboarding import OnboardingView  # noqa: E402
from app.views.results import ResultsView  # noqa: E402
from app.views.swipe import SwipeView  # noqa: E402
from app.views.welcome import WelcomeView  # noqa: E402
from recsys.artifacts import load  # noqa: E402
from recsys.storage import get_storage  # noqa: E402

VIEWS = {"welcome": WelcomeView, "onboarding": OnboardingView, "swipe": SwipeView, "results": ResultsView}
NARROW = 560  # px; below this, layouts switch to their phone sizes


class Shell:
    """Owns the page for one visitor and swaps screens in place."""

    def __init__(self, page: ft.Page, ctl: Controller):
        self.page = page
        self.ctl = ctl
        self.view = None
        self._narrow = self.is_narrow
        page.on_keyboard_event = self._on_key
        page.on_resize = self._on_resize

    @property
    def is_narrow(self) -> bool:
        return (self.page.width or 1000) < NARROW

    def width(self, max_width: int) -> int:
        """Content width: max_width on large screens, the page minus gutters on phones."""
        return int(min(max_width, (self.page.width or max_width + 32) - 32))

    def go(self, name: str, **kwargs) -> None:
        self.view = VIEWS[name](self, **kwargs)
        self.render()

    def render(self) -> None:
        self.page.controls.clear()
        self.page.controls.append(
            ft.SafeArea(
                expand=True,
                content=ft.Column(
                    [self.view.build(), ui.footer()],
                    scroll=ft.ScrollMode.AUTO,
                    horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                    expand=True,
                ),
            )
        )
        self.page.update()

    def toast(self, message: str) -> None:
        self.page.show_dialog(ft.SnackBar(ft.Text(message), duration=2500))

    def _on_key(self, e: ft.KeyboardEvent) -> None:
        handler = getattr(self.view, "on_key", None)
        if handler:
            handler(e)

    def _on_resize(self, e) -> None:
        # Re-render only when crossing the phone/desktop breakpoint, so transient
        # state (like an open rating step) survives ordinary window resizing.
        if self.is_narrow != self._narrow:
            self._narrow = self.is_narrow
            self.render()


def main(page: ft.Page) -> None:
    page.title = "Movie Swipe"
    page.theme = ui.theme()
    page.dark_theme = ui.dark_theme()
    page.theme_mode = ft.ThemeMode.SYSTEM
    page.padding = 0
    Shell(page, Controller(load("deploy"), get_storage())).go("welcome")


def run() -> None:
    web = "--web" in sys.argv or os.environ.get("FLET_WEB") == "1"
    # Load the movie data and connect to storage before accepting visitors, so on a
    # cold start this happens behind the host's loading page instead of after a click.
    # A bad storage config also fails here, loudly, instead of on the first visitor.
    load("deploy")
    get_storage()
    ft.run(
        main,
        view=ft.AppView.WEB_BROWSER if web else ft.AppView.FLET_APP,
        host=os.environ.get("HOST"),
        port=int(os.environ.get("PORT", 8550)),
    )


if __name__ == "__main__":
    run()
