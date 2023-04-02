set -e
# Script for deploying the gh-pages branch
git rebase main
python3 -m build
git add -f dist/
git commit --amend --no-edit
git push origin gh-pages -f
