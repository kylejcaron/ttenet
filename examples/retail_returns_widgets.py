"""Interactive figures for the retail returns blog.

Every figure receives plain JSON computed by the notebook from the fitted model and the
simulated ledger; the browser code only draws it and handles its own buttons. Nothing here
calls back into Python, so the figures keep working in a static export of the notebook.
"""

from pathlib import Path

import anywidget
import traitlets

WIDGETS = Path(__file__).with_name("widgets")


class ScenarioForecast(anywidget.AnyWidget):
    """Posterior-predictive storm scenarios against a fixed baseline.

    ``data`` holds every scenario, already computed; the buttons inside the figure switch
    between them in the browser.
    """

    _esm = WIDGETS / "scenario.js"
    _css = WIDGETS / "scenario.css"
    data = traitlets.Dict().tag(sync=True)


class HistoryRange(anywidget.AnyWidget):
    """Three forecast panels with a browser-owned two-handle history window."""

    _esm = WIDGETS / "controls.js"
    _css = WIDGETS / "controls.css"
    data = traitlets.Dict().tag(sync=True)


class SaleDay(anywidget.AnyWidget):
    """Follow any forecast sale day through both return stages in the browser."""

    _esm = WIDGETS / "controls.js"
    _css = WIDGETS / "controls.css"
    data = traitlets.Dict().tag(sync=True)


class BaselineComparison(anywidget.AnyWidget):
    """Compare the chain with one of four precomputed baselines, without Python."""

    _esm = WIDGETS / "controls.js"
    _css = WIDGETS / "controls.css"
    data = traitlets.Dict().tag(sync=True)
