# Third-party notices

Aria's first-party code, documentation and authored fixtures use the repository
[MIT License](LICENSE), unless a file declares different terms. Copyright (c) 2026
梁世娟. Dependency licenses and upstream data rights are separate. The package name
remains `aria-eldercare-agent`; this attribution update does not rename the project.
Current release status is recorded only in [PROGRESS](docs/PROGRESS.md).

## Dependency review — 2026-09-30

The following inventory checks the installed Windows / CPython 3.13.11 environment
against each exact version's public PyPI metadata and its local distribution's
license files. All 40 listed distributions had a local license file and a matching
public version. Links identify the publisher-provided metadata, not a new license
grant from Aria. `OR` preserves an upstream dual-license choice.

On 2026-10-04, a local read-only recheck of the Windows / CPython 3.13.11
development environment resolved the eight runtime, optional, development and
build-tool roots through their active dependencies. All 40 installed distributions
still matched the versions below and retained local license files. This recheck
read installed metadata and license bytes only; it did not re-fetch upstream terms,
review a different operating system, or establish the versions of a future fresh
installation. Its version-specific receipt is retained with task `20261004-05`.

The curated source archive does not bundle these packages, their wheels, Python,
or the virtual environment. Installation resolves the ranges in `pyproject.toml`;
these observed versions are **not a dependency lock**. Recheck the actual resolved
environment before distributing an installer or environment image. Python 3.10's
conditional `exceptiongroup` / `tomli` and Linux-specific dependencies were not
installed here and are outside this version-specific check. Native libraries,
Rust crates, all vendored subcomponents, and the complete CI toolchain were not
individually audited; this inventory is not a complete SBOM or a security audit.

| Package / checked version (public source) | Relationship | Declared license / local confirmation |
| --- | --- | --- |
| [pydantic 2.13.4](https://pypi.org/project/pydantic/2.13.4/) | Direct runtime | MIT |
| [PyYAML 6.0.3](https://pypi.org/project/PyYAML/6.0.3/) | Direct runtime | MIT |
| [annotated-types 0.8.0](https://pypi.org/project/annotated-types/0.8.0/) | pydantic | MIT |
| [pydantic-core 2.46.4](https://pypi.org/project/pydantic-core/2.46.4/) | pydantic | MIT; native dependency closure not audited |
| [typing-extensions 4.16.0](https://pypi.org/project/typing-extensions/4.16.0/) | pydantic / pydantic-core / typing-inspection | PSF-2.0; preserve the full upstream LICENSE |
| [typing-inspection 0.4.2](https://pypi.org/project/typing-inspection/0.4.2/) | pydantic | MIT |
| [cryptography 49.0.0](https://pypi.org/project/cryptography/49.0.0/) | Direct optional `secure` | Apache-2.0 OR BSD-3-Clause |
| [cffi 2.1.0](https://pypi.org/project/cffi/2.1.0/) | cryptography on CPython | MIT-0, subject to upstream file exceptions |
| [pycparser 3.0](https://pypi.org/project/pycparser/3.0/) | cffi | BSD-3-Clause |
| [pytest 9.1.1](https://pypi.org/project/pytest/9.1.1/) | Direct `dev` | MIT |
| [ruff 0.16.1](https://pypi.org/project/ruff/0.16.1/) | Direct `dev` | MIT; native dependency closure not audited |
| [colorama 0.4.6](https://pypi.org/project/colorama/0.4.6/) | pytest / build on Windows | BSD-3-Clause, identified from LICENSE.txt |
| [iniconfig 2.3.0](https://pypi.org/project/iniconfig/2.3.0/) | pytest | MIT |
| [packaging 26.2](https://pypi.org/project/packaging/26.2/) | pytest / build / twine | Apache-2.0 OR BSD-2-Clause |
| [pluggy 1.6.0](https://pypi.org/project/pluggy/1.6.0/) | pytest | MIT |
| [Pygments 2.20.0](https://pypi.org/project/Pygments/2.20.0/) | pytest / readme-renderer / rich | BSD-2-Clause; upstream also includes AUTHORS |
| [setuptools 83.0.0](https://pypi.org/project/setuptools/83.0.0/) | Direct build backend | MIT at package level; vendored exceptions below |

The next table covers the additional installed `build` / `twine` tooling used by
the documented build checks and CI, following active `Requires-Dist` recursively.
Shared dependencies already appear above. These tools are not runtime dependencies
and were not used to upload or publish a package during this review.

| Package / checked version (public source) | Declared license / local confirmation |
| --- | --- |
| [build 1.5.0](https://pypi.org/project/build/1.5.0/) | MIT |
| [twine 7.0.0](https://pypi.org/project/twine/7.0.0/) | Apache-2.0 |
| [pyproject-hooks 1.2.0](https://pypi.org/project/pyproject-hooks/1.2.0/) | MIT |
| [readme-renderer 46.0](https://pypi.org/project/readme-renderer/46.0/) | Apache-2.0 |
| [requests 2.34.2](https://pypi.org/project/requests/2.34.2/) | Apache-2.0; includes an upstream NOTICE |
| [requests-toolbelt 1.0.0](https://pypi.org/project/requests-toolbelt/1.0.0/) | Apache-2.0 |
| [urllib3 2.7.0](https://pypi.org/project/urllib3/2.7.0/) | MIT |
| [keyring 25.7.0](https://pypi.org/project/keyring/25.7.0/) | MIT |
| [rfc3986 2.0.0](https://pypi.org/project/rfc3986/2.0.0/) | Apache-2.0 |
| [rich 15.0.0](https://pypi.org/project/rich/15.0.0/) | MIT |
| [id 1.6.1](https://pypi.org/project/id/1.6.1/) | Apache-2.0 |
| [nh3 0.3.7](https://pypi.org/project/nh3/0.3.7/) | MIT; native dependency closure not audited |
| [docutils 0.23](https://pypi.org/project/docutils/0.23/) | Public-domain dedication with file exceptions; see below |
| [charset-normalizer 3.5.1](https://pypi.org/project/charset-normalizer/3.5.1/) | MIT |
| [idna 3.19](https://pypi.org/project/idna/3.19/) | BSD-3-Clause |
| [certifi 2026.7.22](https://pypi.org/project/certifi/2026.7.22/) | MPL-2.0 |
| [pywin32-ctypes 0.2.3](https://pypi.org/project/pywin32-ctypes/0.2.3/) | BSD-3-Clause |
| [jaraco.classes 3.4.0](https://pypi.org/project/jaraco.classes/3.4.0/) | MIT |
| [jaraco.functools 4.6.0](https://pypi.org/project/jaraco.functools/4.6.0/) | MIT |
| [jaraco.context 6.1.2](https://pypi.org/project/jaraco.context/6.1.2/) | MIT |
| [markdown-it-py 4.2.0](https://pypi.org/project/markdown-it-py/4.2.0/) | MIT; preserve both LICENSE and LICENSE.markdown-it |
| [more-itertools 11.1.0](https://pypi.org/project/more-itertools/11.1.0/) | MIT |
| [mdurl 0.1.2](https://pypi.org/project/mdurl/0.1.2/) | MIT; LICENSE also attributes upstream Node.js URL code |

Package-level labels do not override component notices. The checked setuptools
installation contains, for example, vendored `autocommand` 2.2.2 with LGPL-3.0 text,
and other vendored packages with their own licenses. Docutils' `COPYING.rst` lists
BSD file exceptions and a GPL-3.0-or-later Emacs helper; it does not label every
Docutils file GPL. These components are not copied into Aria's source archive.
If dependency binaries or vendor code are later redistributed, preserve all
applicable upstream licenses, notices and source obligations for that exact build.

## Retained resource provenance

| Retained material | Source and distribution boundary |
| --- | --- |
| `aria/assets/voice/minimal.html`, `care.html` | Project-authored HTML/CSS/JavaScript and inline SVG stage; no linked third-party script, font, image or video asset was found in these files. The stage is presentation, not sensor evidence. |
| `aria/assets/personas/laochen_demo.yaml` | Marked `synthetic_demo`; authored demonstration profile and synthetic evidence, not a real person's record or measured learning. |
| `aria/assets/strategy_matrix.yaml`, `aria/scenes/companion_elder/elder_kb.yaml` | Project-authored strategy and domain summaries. The matrix records migration from this project's earlier benchmark. Literature names are references, not bundled publications or validated diagnostic/medical instruments. This license review does not validate the claims. |
| `aria/bench/cases/refocus_e1_v1.yaml`, `tests/fixtures/followthrough_synthetic.json`, test literals | Authored fictional/synthetic fixtures. They are not CPED exports or real-model effectiveness evidence. |
| `aria/bench/cases/e1b_v1_draft.yaml`, `aria/bench/ctest_cases.py`, `tests/fixtures/external_emotion_no_input_prompts.json` | Authored fictional dialogue scripts and regression digests. Simulated emotion readings, scripted assistants and software fixtures are not real-person observations or proof of model effectiveness. |
| `aria/bench/datasets/cped_v1.yaml`, local CPED preparation code and protocols | Aria's source manifest and local tools; upstream data and license boundaries are described below. |
| Public experiment contracts and metadata | Authored protocols plus attributed public endpoint metadata. CPED preparation contains aggregate counts only; source text, role identities and private sampling records are excluded. Public endpoint URLs remain in the metadata records. |

The release policy excludes old preview PNG/MP4 files, third-party datasets,
CosyVoice samples, reference videos, model weights and private recovery archives.
Old notices remain with the archived materials; this update does not relicense
them. Future screenshots or media need their own source and privacy review before
being added to the release package.

## CPED preparation and evaluation

The optional local tools retain a source manifest and CSV loader; they do not
bundle or download dialogue data. The upstream project is
[scutcyr/CPED](https://github.com/scutcyr/CPED). On 2026-09-30, the local
[LICENSES/Apache-2.0.txt](LICENSES/Apache-2.0.txt) matched both the upstream
[current license](https://github.com/scutcyr/CPED/blob/main/LICENSE) and its
[recorded revision](https://github.com/scutcyr/CPED/blob/1e4b81c28a123f22387e06664f37e5dc9322380f/LICENSE)
byte for byte. SHA-256:
`c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4`.

That file records the linked CPED repository's Apache-2.0 statement; it does not
apply Apache-2.0 to all Aria source or establish rights in underlying television
dialogue, characters, performances or media. The retained CPED manifest therefore
keeps **data rights `NOASSERTION / unverified`**, separately from the upstream code
license. Aria's MIT license does not cover CPED data. The operator's recorded
acceptance of local evaluation / aggregate publication is not permission to
redistribute raw dialogue or submit it to a cloud provider.

Only aggregate reports may be exported by the retained runner, excluding raw text,
speaker identities and evidence quotes. See [source preparation](docs/CPED_SOURCE_PREPARATION.md)
and the [holdout protocol](docs/CPED_HOLDOUT_PROTOCOL.md). The curated build selects
the manifest, tools and aggregate metadata; it excludes CSV/TSV, runtime data,
private experiment artifacts, Git history and recovery archives. A final package
must still pass the [release checks](docs/MINIMAL_RELEASE_CHECKLIST.md); this notice
does not certify arbitrary packages built by other methods.

Remote model access, when explicitly enabled, remains governed by the selected
service's terms. No model weights, service rights, or rights to private inputs and
outputs are transferred by this source distribution.

## PrefEval local evaluation tools

The first-party `aria/bench/prefeval_*.py` modules and `scripts/prefeval_prepare.py`
prepare local evaluation material for the upstream
[amazon-science/PrefEval](https://github.com/amazon-science/PrefEval) project.
The local source record pins revision
`50795054b5ff5f418d2b768a331d71e480f93331`; its saved
[upstream license](https://github.com/amazon-science/PrefEval/blob/50795054b5ff5f418d2b768a331d71e480f93331/LICENSE)
is Attribution-NonCommercial 4.0 International (CC BY-NC 4.0), checked on
2026-10-03 against the retained local original.

Upstream data, official judge templates, private sample manifests, generated
responses and grading materials remain in the ignored local task directory.
They are excluded from the curated ZIP and the staged wheel/sdist build. Aria's
MIT license does not relicense these external materials. The recorded use is
local noncommercial evaluation with aggregate-only public reporting; the source
archive does not include the LMSYS dialogue collection or its derived distractor
file. The tools use separately authored distractor topics instead. See
[VALIDATION](docs/VALIDATION.md) for measurement eligibility and its limitations;
shipping preparation code does not mean the experiment has run or passed.
