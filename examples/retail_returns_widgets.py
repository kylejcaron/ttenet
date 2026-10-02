"""Interactive figures for the retail returns blog.

Every figure receives plain JSON computed by the notebook from the fitted model and the
simulated ledger; the browser code only draws it and handles its own buttons. Nothing here
calls back into Python, so the figures keep working in a static export of the notebook.
"""

from pathlib import Path

import anywidget
import traitlets

WIDGETS = Path(__file__).with_name("widgets")


class PurchaseStory(anywidget.AnyWidget):
    """Three real ledger records scrubbed through time: a censoring lesson, not a forecast."""

    _esm = WIDGETS / "purchase.js"
    _css = WIDGETS / "purchase.css"
    data = traitlets.Dict().tag(sync=True)


class ChainStepper(anywidget.AnyWidget):
    """Four chapters of the fitted chain: sales, initiation timing, receipt timing, receipts."""

    _esm = WIDGETS / "chain.js"
    _css = WIDGETS / "chain.css"
    data = traitlets.Dict().tag(sync=True)


class ScenarioForecast(anywidget.AnyWidget):
    """Posterior-predictive storm scenarios against a fixed baseline.

    ``data`` holds every scenario, already computed; the buttons inside the figure switch
    between them in the browser.
    """

    _esm = WIDGETS / "scenario.js"
    _css = WIDGETS / "scenario.css"
    data = traitlets.Dict().tag(sync=True)
