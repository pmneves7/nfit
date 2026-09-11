"""PyInstaller entry: handle multiprocessing before importing the application."""
import multiprocessing
import sys

if __name__ == "__main__":
    multiprocessing.freeze_support()
    startup_splash = None
    background_modes = {
        "--benchmark-worker",
        "--install-portable-update",
        "--run-script",
        "--smoke-test",
    }
    if not background_modes.intersection(sys.argv[1:]):
        from startup_splash import show_startup_splash

        _startup_app, startup_splash = show_startup_splash()
    from nfit.app_bootstrap import main

    raise SystemExit(main(startup_splash=startup_splash))
