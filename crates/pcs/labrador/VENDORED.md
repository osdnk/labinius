# Vendored LaBRADOR

This directory is an in-tree copy of https://github.com/osdnk/labrador at
commit `ece652f1f156dea3e0bf000dd88ae85609ae96ae` (branch `perf`), a fork of
Gregor Seiler's LaBRADOR (https://github.com/lattice-dogs/labrador, last
upstream commit `8b6626b`). It is no longer a git submodule; changes are made
here directly. `crates/pcs/build.rs` builds `liblabrador.a` from it with
`LOGQ=48`. See `LICENSE` for the library's licence.
