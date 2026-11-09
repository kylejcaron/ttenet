"""Interactive figures for the retail returns blog.

Each view receives plain JSON: computed forecasts or model configuration and source.
The browser code draws it and handles its own controls without calling Python, so
interactions keep working in a static export of the notebook.
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


class ModelOverview(anywidget.AnyWidget):
    """Three configured stages and one browser-owned mechanism/source detail panel."""

    _esm = WIDGETS / "overview.js"
    _css = WIDGETS / "overview.css"
    data = traitlets.Dict().tag(sync=True)
