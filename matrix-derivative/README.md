## Potential improvements
+ Reshape the data such that the threads and blocks can be used most efficiently for the used gpu device
+ There might be better ways to synchronize over the cuda threads, e.g. if the results is not needed immediately. Maybe there is also some lazy mode...

## Requires
+ `torch`
+ `pytest`
+ `pytest-benchmark`

## Install
```bash
python setup.py install
```

### Test and bechmark
```bash
pytest .
```
