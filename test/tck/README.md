## TCK tests

To run these, you will need to install `behave`:


```bash
git clone https://github.com/aneeshdurg/behave
cd behave
python3 setup.py install
python3 run_tck.py
```

To update the list of expected failures, run:
```
python3 run_tck.py --update
```
