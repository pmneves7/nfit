/* Keep the process inside nfit.app so macOS can identify and pin the app. */
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include "launcher_config.h"

int main(int argc, char **argv) {
    int smoke_test = argc == 2 && strcmp(argv[1], "--smoke-test") == 0;
    if (!smoke_test) {
        freopen(NFIT_LOG, "a", stdout);
        freopen(NFIT_LOG, "a", stderr);
    }
    if (chdir(NFIT_SOURCE) != 0) {
        perror("Cannot open nfit checkout");
        return 1;
    }
    setenv("PYTHONHOME", NFIT_PREFIX, 1);
    setenv("PYTHONPATH", NFIT_SOURCE "/src", 1);
    void *python = dlopen(NFIT_LIBRARY, RTLD_NOW | RTLD_GLOBAL);
    if (!python) {
        fprintf(stderr, "Cannot load nfit Python: %s\n", dlerror());
        return 1;
    }
    int (*python_main)(int, char **) = dlsym(python, "Py_BytesMain");
    if (!python_main) {
        fprintf(stderr, "Cannot find Python entry point\n");
        return 1;
    }
    char *code = smoke_test
        ? "from nfit.project_gui import _qt_app; "
          "app = _qt_app(); assert not app.windowIcon().isNull(); "
          "print('nfit launcher smoke test passed')"
        : "from nfit.project_gui import main; raise SystemExit(main())";
    char *python_argv[] = {NFIT_PYTHON, "-c", code, NULL};
    return python_main(3, python_argv);
}
