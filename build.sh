# Script for deploying the gh-pages branch
set -e

# generate spycy whl
pip install .
python3 -m build

# setup hosted contents
mkdir build
cd build
mv ../dist .
cp -r ../static .
cp ../index.html .
