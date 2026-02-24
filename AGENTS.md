- This is the only AGENTS.md, there are no recursive AGENTS.md
- When you are working on a bug, first create a standalone file that
  reproduces the bug and verify it fails in the expected way.  Use this to
  test if your changes work.  Once the change is passing, find an appropriate
  test file to add the test to and make sure to follow local conventions on
  the test file.
- If you are running the real test suite, DO NOT run the entire test suite.
  Instead run only a single test case, e.g., 'python test/test_torch.py TestTorch.test_dir'
- Do NOT run setup.py, you do not have a working build environment
- Do NOT run pre-commit, it is not setup
- To run lint, run 'lintrunner -a' (which will autoapply changes)
- Do NOT attempt to install dependencies, you do not have Internet access
- Do NOT create summary files unless explicitly asked
- When you are ready to make a PR, do exactly these steps:
  - git stash -u
  - git reset --hard $(cat /tmp/orig_work.txt) # NB: reset to the LOCAL branch, do NOT fetch
  - git stash pop
  - Resolve conflicts if necessary

## Cursor Cloud specific instructions

- The development environment uses a Python 3.12 venv at `/workspace/.venv`. Activate it with `source /workspace/.venv/bin/activate` or ensure `/workspace/.venv/bin` is on PATH.
- PyTorch is built CPU-only (editable install) with `CC=gcc CXX=g++`. The default `c++`/`cc` symlinks point to Clang 18 which cannot find `-lstdc++`; always set `CC=gcc CXX=g++` when rebuilding.
- `libstdc++-13-dev` and `python3-dev` must be installed (system packages) for the C++ build to succeed. These are already installed in the snapshot.
- The `uv` binary is symlinked into the venv at `/workspace/.venv/bin/uv`; lintrunner's linter adapters depend on it being on PATH.
- Lint: `lintrunner -a` (auto-apply) or `lintrunner <file>` to check a single file. `spin lint` / `spin fixlint` also work. `lintrunner init` must have been run (already done in the snapshot).
- Tests: Run individual tests only, e.g., `python test/test_torch.py TestTorch.test_dir`. See CLAUDE.md for the test class convention.
- Build: See CLAUDE.md. The canonical command is `pip install -e . -v --no-build-isolation`. For CPU-only rebuild: `CC=gcc CXX=g++ USE_CUDA=0 MAX_JOBS=4 pip install --no-build-isolation -v -e .`
- Python-only changes in `torch/` are picked up immediately (editable install). C++ changes require a rebuild.
- `torch.compile` (Dynamo/Inductor) works for CPU targets. CUDA is not available in this environment.
