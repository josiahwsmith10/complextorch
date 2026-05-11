# Examples

Runnable, re-executed-on-every-build examples that demonstrate `complextorch`
on small but realistic problems. If an example breaks against the latest
`complextorch` source, the docs build fails — so these are guaranteed not to
rot.

```{toctree}
:maxdepth: 1

getting_started
```

## Suggesting an example

Examples live as MyST-NB notebooks under `docs/source/examples/`. Open a PR
with a new `.md` file in that directory and add it to the toctree above; the
docs build will execute it on every commit.
