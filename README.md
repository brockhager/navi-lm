# NaviLM

An open research project on more compute-efficient language model architectures, built on
[Next-Concept Modeling (NCM)](https://arxiv.org/abs/2610.14162) and run on a single 8 GB consumer GPU.

**Status:** early. The released NCM LM1B checkpoint has been reproduced locally; training code is next.

## What we are doing

- Reproduce NCM's published results on small hardware.
- Write the missing pieces: a generator training loop and a matched autoregressive baseline.
- Change one variable at a time (backbone, quantizer, level count) and report loss and tokens/sec at equal compute, 3 seeds.

## Docs

- [Changelog](CHANGELOG.md)
- [Index](docs/00-index.md)
- [Summary](docs/01-summary.md)
- [Handoff and background](docs/handoff.md)
- [Setup: WSL2 + PyTorch on Blackwell](docs/SETUP.md)
- [Experiments and run log](docs/EXPERIMENTS.md)

## Contributing

Issues and pull requests are welcome, especially for the AR baseline, training loop, and experiment ideas.

## Credits

NaviLM builds on NCM by Kumbong et al. (Stanford Hazy Research and Together AI):
[paper](https://arxiv.org/abs/2610.14162), [code](https://github.com/HazyResearch/ncm) (MIT).

## License

MIT. See [LICENSE](LICENSE).
