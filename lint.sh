#!/usr/bin/bash

if [[ -z "$CHECK_ONLY" ]]
then
  files=$(git ls-files | grep ".py$")
else
  files=$(git diff --cached --name-only --diff-filter=ACM | grep ".py$")
fi

set -e
if [[ -z "$files" ]]
then
  echo "No files to lint" >&2
  exit 0
fi
echo $files

if [[ -z "$CHECK_ONLY" ]]
then
  echo "!"
  black $files
  isort $files
else
  black --check $files
  isort --check $files
fi
