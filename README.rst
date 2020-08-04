NN Compress
===========


- The goal is to compress the Ubuntu20.4 image into < 500 MiB while the original image is around 2.6GiB.
- PyTorch will need to be installed separately since poetry does not play nice with it.
- `python -m nncompress path/to/iso/file`
- `config.json` contains config parameters.
- Use tensorboard to see logs.
