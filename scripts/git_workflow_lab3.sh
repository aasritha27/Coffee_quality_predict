#!/usr/bin/env bash
# Lab 3 walkthrough: commits, branches, merge, CONFLICT + resolution, revert, tag.
# Run from the project root in Git Bash / Linux / macOS:  bash scripts/git_workflow_lab3.sh
# Read each step; screenshot `git log --oneline --graph --all` at the end for your report.
set -e
[ -d .git ] || git init -b main
git add .
git commit -m "Initial project structure and Lab 1 setup" || true

# --- feature branch 1: change a hyperparameter default -------------------------------
git checkout -b feature/tune-baseline
sed -i 's/max_depth=10/max_depth=12/' src/train.py
git commit -am "Tune baseline: max_depth 10 -> 12"

# --- feature branch 2 (from main): conflicting edit on the SAME line -----------------
git checkout main
git checkout -b feature/regularize-baseline
sed -i 's/max_depth=10/max_depth=8/' src/train.py
git commit -am "Regularize baseline: max_depth 10 -> 8"

# --- merge the first branch cleanly, then merge the second => conflict ---------------
git checkout main
git merge --no-ff feature/tune-baseline -m "Merge feature/tune-baseline"
if ! git merge --no-ff feature/regularize-baseline -m "Merge feature/regularize-baseline"; then
  echo ">>> CONFLICT (expected). Inspect with: git status ; git diff"
  git diff --name-only --diff-filter=U
  # resolve: keep the reference value (max_depth=10) by restoring the line from the first commit.
  # (In a real project you would edit the <<<<<<< / ======= / >>>>>>> markers by hand.)
  INIT=$(git rev-list --max-parents=0 HEAD)
  git show "$INIT:src/train.py" > src/train.py
  git add src/train.py
  git commit -m "Resolve merge conflict in train.py: keep max_depth=10"
fi

# --- rollback / revert practice ------------------------------------------------------
echo "# temporary experimental change" >> src/train.py
git commit -am "Temporary experimental change"
BAD=$(git rev-parse --short HEAD)
git revert --no-edit "$BAD"

# --- tag the final state -------------------------------------------------------------
git tag -a v1.0-lab4 -m "Labs 1-4 complete: preprocess, baseline, git workflow, MLflow"
git log --oneline --graph --all
git tag
