"""Test, lint, and complexity sessions."""

import nox

nox.options.default_venv_backend = "uv"
nox.options.reuse_existing_virtualenvs = True
nox.options.sessions = ["lint", "complexity", "tests"]

PYTHON_VERSIONS = ["3.12", "3.13", "3.14"]


def _sync(session: nox.Session, *groups: str, extras: tuple[str, ...] = ()) -> None:
    """Install the locked environment into the session's virtualenv."""
    args = ["uv", "sync", "--locked"]
    for group in groups:
        args += ["--group", group]
    for extra in extras:
        args += ["--extra", extra]
    session.run_install(*args, env={"UV_PROJECT_ENVIRONMENT": session.virtualenv.location})


@nox.session(python=PYTHON_VERSIONS)
def tests(session: nox.Session) -> None:
    """Run the unit tests; the forecast extra keeps numpyro_forecast tests from skipping."""
    _sync(session, "dev", extras=("forecast",))
    session.run("pytest", "-q", *session.posargs)


@nox.session(python="3.12")
def lint(session: nox.Session) -> None:
    """Check lint rules and formatting."""
    _sync(session, "dev")
    session.run("ruff", "check", ".")
    session.run("ruff", "format", "--check", ".")


@nox.session(python="3.12")
def complexity(session: nox.Session) -> None:
    """Fail on new or worsened complexity relative to complexipy-snapshot.json."""
    _sync(session, "dev")
    session.run("complexipy", "src", "--plain")
