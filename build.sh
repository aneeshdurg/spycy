# Script for deploying the gh-pages branch
set -e

# generate spycy whl
pip install .
python3 -m build

# setup hosted contents
mkdir build
git log --pretty=oneline | head -n1 | cut -d\  -f1 > build/version
cd build
mv ../dist .
cp -r ../static .
cp ../index.html .
