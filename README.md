# Densely Connected Convolutional Networks (DenseNets)

A Tensorflow 2.0 reimplementation of DenseNet.


## Citation

[Densely Connected Convolutional Networks](http://arxiv.org/abs/1608.06993) won Best Paper at CVPR 2017.

```[text]
@inproceedings{huang2017densely,
    title={Densely connected convolutional networks},
    author={Huang, Gao and Liu, Zhuang and van der Maaten, Laurens and Weinberger, Kilian Q },
    booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
    year={2017}
}
```

## Requirements

This implementation assumes `python==3.7.3` and `tensorflow-gpu==2.0.0`. All requirements are listed in `requirements.txt`. If you wish to run without gpu support, specify `tensorflow==2.0.0` in `requirements.txt` before continuing. To install, run:

```[bash]
pip install -r requirements.txt
```

## Formatting and Linting

We use `black`, `flake8`, `mypy`, and `isort` to format and lint code. Before contributing, make sure these packages are installed:

```[bash]
pip install black flake8 mypy isort
```

Then run the following:

```[bash]
# fix formatting
black .

# check for errors
flake8 --config=.flake8

# fix import sort order
isort

# type checking
mypy --config-file mypy.ini code
```

Formatting configuration files should automatically be picked up if you are using an editor that supports these plugins, e.g., `vscode`.
