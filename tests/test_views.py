"""Build every screen with a stand-in page, to catch UI API mistakes without a browser."""

from datetime import date

import flet as ft
import numpy as np
import pytest

from app.controller import Controller
from app.views.onboarding import OnboardingView
from app.views.results import ResultsView
from app.views.swipe import SwipeView
from app.views.welcome import WelcomeView
from recsys.storage import SQLiteStorage


class FakePage:
    def __init__(self, width):
        self.width = width
        self.controls = []
        self.dialogs = []

    def update(self, *controls):
        pass

    def show_dialog(self, d):
        self.dialogs.append(d)

    def pop_dialog(self):
        self.dialogs.pop()


class FakeShell:
    def __init__(self, ctl, width):
        self.page = FakePage(width)
        self.ctl = ctl
        self.went = []

    @property
    def is_narrow(self):
        return self.page.width < 560

    def width(self, max_width):
        return int(min(max_width, self.page.width - 32))

    def go(self, name, **kw):
        self.went.append((name, kw))

    def render(self):
        pass

    def toast(self, msg):
        pass


def walk(control):
    """Every control in a tree (to assert on what was built)."""
    yield control
    for attr in ("content", "controls", "leading", "title", "subtitle", "label"):
        child = getattr(control, attr, None)
        if isinstance(child, list):
            for c in child:
                if isinstance(c, ft.BaseControl):
                    yield from walk(c)
        elif isinstance(child, ft.BaseControl):
            yield from walk(child)


@pytest.fixture(params=[1200, 390], ids=["desktop", "phone"])
def shell(request, real_art):
    ctl = Controller(real_art, SQLiteStorage(":memory:"), rng=np.random.default_rng(0), today=lambda: date(2026, 10, 2))
    return FakeShell(ctl, request.param)


def texts(control):
    return [c.value for c in walk(control) if isinstance(c, ft.Text) and c.value]


def test_welcome_and_new_user(shell):
    v = WelcomeView(shell)
    assert "Movie Swipe" in texts(v.build())
    v._new(None)
    assert shell.page.dialogs and shell.ctl.user is not None
    v.handle_field.value = "nobody-here-11"
    v._returning(None)
    assert v.error.visible and not shell.went


def test_onboarding(shell):
    shell.ctl.start_new()
    v = OnboardingView(shell)
    v.build()
    v.search.value = "matrix"
    v._on_search(None)
    assert len(v.results.controls) > 0
    for title in ("The Matrix", "Toy Story", "Arrival"):
        m = shell.ctl.search(title).iloc[0]
        v._add(int(m["movie_id"]), m["title"], int(m["year"]), m["poster_url"])
    assert not v.continue_btn.disabled
    v._continue(None)
    assert shell.went[-1][0] == "swipe" and not shell.ctl.needs_seeds


def test_swipe_steps_and_results(shell):
    ctl = shell.ctl
    ctl.start_new()
    ctl.save_seeds([int(ctl.search(t).iloc[0]["movie_id"]) for t in ("The Matrix", "Toy Story", "Arrival")])
    v = SwipeView(shell)
    v.build()
    for step in ("seen", "unseen", None):
        v.step = step
        v.build()
    for _ in range(10):
        v.build()
        v.card_box = None  # skip the exit animation (needs a live page)
        v._record("rated", 4)
    assert shell.page.dialogs  # the 10-answer check-in
    r = ResultsView(shell, trigger="checkin_10")
    built = texts(r.build())
    assert "Your recommendations" in built and any(t.startswith("Because you") for t in built)
    first = r.cards[0].movie_id
    r._open_rating(first)
    r._answer(first, "rated", 5)
    r._answer(r.cards[1].movie_id, "interested")
    assert "My list (1)" in texts(r.watch_box)
    tables = ctl.store.export()
    assert set(tables["rec_feedback"]["kind"]) == {"rated", "interested"}


def test_keyboard(shell):
    ctl = shell.ctl
    ctl.start_new()
    ctl.save_seeds([int(ctl.search(t).iloc[0]["movie_id"]) for t in ("The Matrix", "Toy Story", "Arrival")])
    v = SwipeView(shell)
    v.build()
    v.card_box = None
    v.on_key(ft.KeyboardEvent(name="keyboard_event", control=None, key="Arrow Right", shift=False, ctrl=False, alt=False, meta=False))
    assert v.step == "seen"
    v.on_key(ft.KeyboardEvent(name="keyboard_event", control=None, key="5", shift=False, ctrl=False, alt=False, meta=False))
    assert ctl.session.answers[-1].rating == 5 and v.step is None


def test_themes_build():
    from app import components as ui

    ui.theme()
    ui.dark_theme()
    ui.footer()
