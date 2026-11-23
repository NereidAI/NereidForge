# Vendored label-studio-ml (Apache-2.0, HumanSignal/label-studio-ml-backend)

`label-studio-ml/` is a verbatim snapshot of the upstream
[HumanSignal/label-studio-ml-backend](https://github.com/HumanSignal/label-studio-ml-backend)
package, commit `7b0d01f3593e327ad9cd2a92dbd08aa89f6403f4` (master, 2026-10),
package version `2.0.1dev0`. Upstream `LICENSE` (Apache-2.0) is included.

## Why vendored

This backend targets the 2.x API (`label_studio_ml.response.ModelResponse`,
Flask 2 `app.post` routes). The newest PyPI release `label-studio-ml==1.0.9`
predates both and also mis-declares its `rq` dependency. The Dockerfile
installs the package's declared runtime deps from PyPI and then installs this
tree with `pip install --no-deps`, so no build needs GitHub access (required
for offline builds).

The `examples/` and `tests/` directories of the upstream tree were removed
(they are example backends, ~50 MB, not part of the runtime package).

## Refreshing

```sh
curl -L -o /tmp/lsmb.tar.gz \
  https://ghfast.top/https://github.com/HumanSignal/label-studio-ml-backend/archive/<commit>.tar.gz
tar xzf /tmp/lsmb.tar.gz -C /tmp
cp -r /tmp/label-studio-ml-backend-<commit>/label_studio_ml ./label-studio-ml/
cp /tmp/label-studio-ml-backend-<commit>/{setup.py,requirements.txt,README.md,MANIFEST.in,LICENSE} ./label-studio-ml/
rm -rf ./label-studio-ml/label_studio_ml/{examples,tests}
```

Once PyPI publishes a release that contains `response.py` and pairs with
Flask 2+, this vendor tree can be deleted in favour of a pinned wheel.
