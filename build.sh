# Script for deploying the gh-pages branch
git merge main
python3 -m build
git add -f dist/
git commit -m "update" --allow-empty
git push origin gh-pages
