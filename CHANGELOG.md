# Changelog

All notable changes to this project are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

From v1.2.0 onwards this file is maintained automatically by
[release-please](https://github.com/googleapis/release-please). Anything
above v1.1.0 was seeded retroactively from the GitHub Releases history.

<!-- next-version-placeholder -->

## [1.5.0](https://github.com/VforVitorio/F1-StratLab/compare/v1.4.3...v1.5.0) (2026-05-15)


### Features

* **agents:** N27 detects deployed Safety Car via RCM events and forces sc_prob=1.0 ([ea8ac95](https://github.com/VforVitorio/F1-StratLab/commit/ea8ac95f59d4ef6eb13f6800033c0ebe281a1e4e))
* **agents:** N28 honors active Safety Car (banner prompt + STAY_OUT-&gt;PIT_NOW guard-rail) ([c19d887](https://github.com/VforVitorio/F1-StratLab/commit/c19d887d8485a092b39b0dab2ae5942815f68ecc))
* **agents:** orchestrator threads RCM events to N27 and forces N28+N30 routing under SC ([497afb2](https://github.com/VforVitorio/F1-StratLab/commit/497afb2536ecfd5546ff3715f46c515c06c5eb99))
* **arcade:** add RaceEventsPanel HUD card (Yellow/SC/VSC/Red flag pill with fade) ([e18c6af](https://github.com/VforVitorio/F1-StratLab/commit/e18c6af6374aeef3b4bb76ee899184e65e48b99c))
* **arcade:** cache per-lap FastF1 TrackStatus on SessionData (cache v6) ([6f86554](https://github.com/VforVitorio/F1-StratLab/commit/6f8655416dfe35805af9ac41ef388ace6d8f7759))
* **arcade:** pass sc_currently_active through MoE routing for parity with main orchestrator ([be5b127](https://github.com/VforVitorio/F1-StratLab/commit/be5b127c2302eb2a462447863122505e6c46e89d))
* **arcade:** wire RaceEventsPanel into F1ArcadeView (anchored under leaderboard) ([cd1b6b5](https://github.com/VforVitorio/F1-StratLab/commit/cd1b6b5bb4e1da13aaee8e26a59de69850ca365c))


### Bug Fixes

* **arcade:** SimConnector waits for arcade playback before processing each lap ([e30778e](https://github.com/VforVitorio/F1-StratLab/commit/e30778ee05fc8442a415e3541244bfd4beacee94))
* **arcade:** skip stale laps when arcade seeks ahead of the strategy loop ([0e5ec0e](https://github.com/VforVitorio/F1-StratLab/commit/0e5ec0e70985c59c79728ee7363b41a3ae3cd218))
* **arcade:** wire arcade lap provider into SimConnector so pause stops the agent flow ([bb6ef56](https://github.com/VforVitorio/F1-StratLab/commit/bb6ef564000821eb15608a04f8e2e9b40ef76900))


### Documentation

* **multi-agent:** document RCM Safety Car override (N27 + N28 + routing) ([9a740dd](https://github.com/VforVitorio/F1-StratLab/commit/9a740ddd44e2cee934099a30bd6906ef09fa22bd))

## [1.4.3](https://github.com/VforVitorio/F1-StratLab/compare/v1.4.2...v1.4.3) (2026-05-14)


### Bug Fixes

* **chat:** wire Download Report button to live /tool-message endpoint (submodule a921032) ([1dd018d](https://github.com/VforVitorio/F1-StratLab/commit/1dd018d29a874b5622650794c0cd72102392c889))

## [1.4.2](https://github.com/VforVitorio/F1-StratLab/compare/v1.4.1...v1.4.2) (2026-05-13)


### Documentation

* **dev:** update chat smoke commands to /tool-message + MCP examples ([b7526d0](https://github.com/VforVitorio/F1-StratLab/commit/b7526d05cc2cde1e08e56545e0730df7298074d5))

## [1.4.1](https://github.com/VforVitorio/F1-StratLab/compare/v1.4.0...v1.4.1) (2026-05-13)


### Documentation

* **api:** document MCP-driven chat endpoints and module layout ([30a5120](https://github.com/VforVitorio/F1-StratLab/commit/30a51208d7e5666d7ffc872629d053620ad8edc5))
* **diagrams:** rename lmstudio_service.py to llm_service.py in chat MCP flow ([68efe8f](https://github.com/VforVitorio/F1-StratLab/commit/68efe8ff469c9cf5f68c2aadb5572a0013390484))
* **frontend:** point chat tool-result renderer at /chat/tool-message-stream ([fdaeb6e](https://github.com/VforVitorio/F1-StratLab/commit/fdaeb6ec3356df8eb39dd37da9301779be3e3b01))

## [1.4.0](https://github.com/VforVitorio/F1-StratLab/compare/v1.3.1...v1.4.0) (2026-05-12)


### Features

* **docs:** add agents API reference with entry points and schemas ([5ec358a](https://github.com/VforVitorio/F1-StratLab/commit/5ec358aea2f0bd7faca5843857347f36f45003ae))
* **docs:** add app entry with hash router and global click delegate ([359a831](https://github.com/VforVitorio/F1-StratLab/commit/359a83116fb748d8e1ce813fb51342610e067c6f))
* **docs:** add apple-touch-icon for iOS home-screen install ([293957f](https://github.com/VforVitorio/F1-StratLab/commit/293957f8be0ac24693bbdec06e32bfae82ed6728))
* **docs:** add arcade dashboard architecture page ([2959832](https://github.com/VforVitorio/F1-StratLab/commit/29598326746dc1ff811eeedb9361ae3b997b70c1))
* **docs:** add arcade quick start page with three-window boot ([6f34e66](https://github.com/VforVitorio/F1-StratLab/commit/6f34e6683f38c34a63bbf7a2eb3273304c9f14a0))
* **docs:** add arcade strategy pipeline page on local N31 duplicate ([84facc6](https://github.com/VforVitorio/F1-StratLab/commit/84facc65a146e22be57a705473ded2247dc49865))
* **docs:** add architecture page with end-to-end layer tour ([f98e1f6](https://github.com/VforVitorio/F1-StratLab/commit/f98e1f698b53dd3d9f9139b83185406696b7a7e2))
* **docs:** add brand design tokens mirroring f1stratlab.com palette ([2aeccad](https://github.com/VforVitorio/F1-StratLab/commit/2aeccad7da2b951e84dd08ac4c964371b58b0c9f))
* **docs:** add changelog mirror page sourced from repo root CHANGELOG ([2d9c6df](https://github.com/VforVitorio/F1-StratLab/commit/2d9c6df7ffd9c295029147e15f29003a698e79ea))
* **docs:** add CI/CD pipeline page covering release-please and deploy ([caef8e2](https://github.com/VforVitorio/F1-StratLab/commit/caef8e2c3d333a9eb1259f8c22e9fd50b70ac18c))
* **docs:** add custom home page with hero agent grid stats and graph teaser ([2081e25](https://github.com/VforVitorio/F1-StratLab/commit/2081e25db096ada8e35ef88787cdb4f7330524e7))
* **docs:** add design upload pasted-1778611374655 to uploads ([e077698](https://github.com/VforVitorio/F1-StratLab/commit/e0776988f9f393c100b0cab743f5f7482e98b813))
* **docs:** add design upload pasted-1778611401163 to uploads ([3e37fef](https://github.com/VforVitorio/F1-StratLab/commit/3e37fefe57530b102a9f9135661040f2da5edc77))
* **docs:** add design upload pasted-1778611468905 to uploads ([b307929](https://github.com/VforVitorio/F1-StratLab/commit/b3079290adfdafa227bbd706e0eea441e118a9ab))
* **docs:** add design upload pasted-1778612008927 to uploads ([2bada27](https://github.com/VforVitorio/F1-StratLab/commit/2bada276922afbd4b130a4ee2de635fe5403a3fd))
* **docs:** add design upload pasted-1778612233726 to uploads ([76e2ccb](https://github.com/VforVitorio/F1-StratLab/commit/76e2ccbf4d9edc4b37fb03080c85c44413cf813c))
* **docs:** add design upload pasted-1778612297400 to uploads ([661263f](https://github.com/VforVitorio/F1-StratLab/commit/661263f0e0e35e6e12d6dd0a96d20cd900c3e5ac))
* **docs:** add development hub page with contributor map ([b2e835d](https://github.com/VforVitorio/F1-StratLab/commit/b2e835d17a56517cccac7ce5eb8c52bdde290c3b))
* **docs:** add docs maintenance page covering build and theming ([81c943a](https://github.com/VforVitorio/F1-StratLab/commit/81c943a6e82b0d8c12273aebb9772b379f7c813f))
* **docs:** add docs.css with three-pane layout and component styles ([575f70b](https://github.com/VforVitorio/F1-StratLab/commit/575f70b60c34eb45cb90393c27ff4e0ee6fd1230))
* **docs:** add driver colors page describing year-aware palette ([48d6937](https://github.com/VforVitorio/F1-StratLab/commit/48d69379212cb7d0933a9dd59a4eca2613110511))
* **docs:** add FastAPI backend page with router map and SSE flow ([4c98f84](https://github.com/VforVitorio/F1-StratLab/commit/4c98f84f079bd3a70c78109a154230ca4b8e05cc))
* **docs:** add favicon copied from f1stratlab-web landing assets ([bab2720](https://github.com/VforVitorio/F1-StratLab/commit/bab2720015daf52acd4e8fc6f960796f4dda5d31))
* **docs:** add force-directed Obsidian-style knowledge graph with tags ([d042c9d](https://github.com/VforVitorio/F1-StratLab/commit/d042c9d4edadc891e9502c3c93b4c7502b07d4f9))
* **docs:** add getting started page with dynamic wheel URL placeholder ([47c8964](https://github.com/VforVitorio/F1-StratLab/commit/47c8964227b3123ae62ff3bb37d2f4c717fd85bc))
* **docs:** add home page content with current release row ([932eac5](https://github.com/VforVitorio/F1-StratLab/commit/932eac5c508b351273b0b08c68c303db5db590d1))
* **docs:** add markdown renderer with mermaid and prism highlighting ([b1d2fdd](https://github.com/VforVitorio/F1-StratLab/commit/b1d2fddf0059d7d395463d8ad7976d226488964b))
* **docs:** add meet the author page with bio and contact links ([a33053d](https://github.com/VforVitorio/F1-StratLab/commit/a33053d11ac085ddcffc9b917de1af53f2c7b224))
* **docs:** add Meet the author visual section to home with avatar and link cards ([628c774](https://github.com/VforVitorio/F1-StratLab/commit/628c77458d4a03523a596cdee74115026fee4f19))
* **docs:** add multi-agent page covering N25 through N31 ([4a28b91](https://github.com/VforVitorio/F1-StratLab/commit/4a28b91af7936213dec2305acfe8ba4055f61b3c))
* **docs:** add nav.js with PAGES config and meet-the-author entry ([ff69cf4](https://github.com/VforVitorio/F1-StratLab/commit/ff69cf456096371a19db237278739da08b5dbec9))
* **docs:** add Open Graph card image from landing banner ([d4035ba](https://github.com/VforVitorio/F1-StratLab/commit/d4035bade0058c052d7d717dcf48380af54552e5))
* **docs:** add race replay engine page with lap_state schema ([79acd96](https://github.com/VforVitorio/F1-StratLab/commit/79acd9607aa950af5e46f507b1de34dd2c5ed03d))
* **docs:** add React docs entry HTML with CDN imports and favicon ([6c4011d](https://github.com/VforVitorio/F1-StratLab/commit/6c4011d8e4c51f9a08a6cdb5da5e4fa5f4da6ef5))
* **docs:** add responsive overrides for 1280 1024 768 and 480 breakpoints ([c768e6a](https://github.com/VforVitorio/F1-StratLab/commit/c768e6a1bd8e3d84bc3b2be50dcbf95897927a96))
* **docs:** add robots.txt allowing all and disallowing uploads ([6a25602](https://github.com/VforVitorio/F1-StratLab/commit/6a25602c5593b4cb98941db43a4ea639e4e62647))
* **docs:** add setup and deployment page with platform matrix ([58ff652](https://github.com/VforVitorio/F1-StratLab/commit/58ff65295f0248ce412b965583ea53d4149ec093))
* **docs:** add sidebar backdrop body scroll lock and Escape to close ([8150e2c](https://github.com/VforVitorio/F1-StratLab/commit/8150e2c913deb9b50fd16e24da97b786fea58021))
* **docs:** add sitemap.xml listing all 18 docs site pages ([d8e4f64](https://github.com/VforVitorio/F1-StratLab/commit/d8e4f649e3fc07183a667b3ad4f644597f6287c1))
* **docs:** add Streamlit frontend page with tab tour ([11feebd](https://github.com/VforVitorio/F1-StratLab/commit/11feebdee3f6b4fd9b0a02e40d3f7d8378d6ede2))
* **docs:** add tags index page grouped by Concepts Surfaces Operations Data ([de7d876](https://github.com/VforVitorio/F1-StratLab/commit/de7d876d9c9315129d676b2d3a08308a12050493))
* **docs:** add thesis results page with verified benchmark metrics ([3a0da70](https://github.com/VforVitorio/F1-StratLab/commit/3a0da70f50667644e63a951967388b29ecae22d7))
* **docs:** add top nav sidebar TOC search footer with version placeholder ([281016e](https://github.com/VforVitorio/F1-StratLab/commit/281016e75a44f8d5c23f4c9b32c13c5d311f57f5))
* **docs:** brand favicon meet-the-author SEO analytics tags changelog and mobile graph ([58ba7b9](https://github.com/VforVitorio/F1-StratLab/commit/58ba7b9ab6a082b96c0763c2035f29988401848a))
* **docs:** bump components cache buster to v7 for responsive sidebar ([debc2c7](https://github.com/VforVitorio/F1-StratLab/commit/debc2c792bcc1f594f983eaea92b80128660e0f2))
* **docs:** full responsive overhaul for mobile tablet and print ([7a6fc7f](https://github.com/VforVitorio/F1-StratLab/commit/7a6fc7ff75e9673e3a1e91a28ba60457d549ed85))
* **docs:** register tags index and changelog mirror pages in nav ([b6328f1](https://github.com/VforVitorio/F1-StratLab/commit/b6328f169ab587900b4ec8a7d34966c139fb1de9))
* **docs:** replace cube brand-mark with favicon image and add author card styles ([eb14471](https://github.com/VforVitorio/F1-StratLab/commit/eb14471d953d493ce01281682100aad93c98e55e))
* **docs:** replace mkdocs with React docs site and dynamic version ([8eebbd6](https://github.com/VforVitorio/F1-StratLab/commit/8eebbd6fe777373f48640ae04e8bbe6c066990e2))
* **docs:** tighter graph physics on mobile so full layout fits viewport ([3934c0c](https://github.com/VforVitorio/F1-StratLab/commit/3934c0c9a0dfa3bab5c20c75fe9748f77cc18ab7))
* **docs:** wire favicon OG meta Twitter card analytics and cache busters ([a4207f8](https://github.com/VforVitorio/F1-StratLab/commit/a4207f80656809a04baf84c73a35597d01c33d1b))


### Documentation

* **components:** point Connect column at HF dataset URL not profile ([7f937b9](https://github.com/VforVitorio/F1-StratLab/commit/7f937b9aae481e981cff3fe6c4aef10216d12ab2))
* **diagrams:** rescue arcade three-window architecture drawio source ([5626a1c](https://github.com/VforVitorio/F1-StratLab/commit/5626a1cc58bc8e4b828d199eba20d0f90205ff0c))
* **diagrams:** rescue backend API drawio source ([9abf8ce](https://github.com/VforVitorio/F1-StratLab/commit/9abf8cee511da5dac0e64f524d6191969887e518))
* **diagrams:** rescue chat MCP flow drawio source ([06fa01d](https://github.com/VforVitorio/F1-StratLab/commit/06fa01d45172ddc708b13c92c1e03d0eb795f06e))
* **diagrams:** rescue data pipeline drawio source ([7e95139](https://github.com/VforVitorio/F1-StratLab/commit/7e95139abf44c5feb66f3b49b82769bb0dad0ba5))
* **diagrams:** rescue docker deployment drawio source ([cfc842e](https://github.com/VforVitorio/F1-StratLab/commit/cfc842edd3416a860e1aee64521b0eedc9d52c64))
* **diagrams:** rescue frontend pages drawio source ([ae27a63](https://github.com/VforVitorio/F1-StratLab/commit/ae27a638657abccd47f6a5f22f49f21d1a17ea71))
* **diagrams:** rescue multi-agent flow drawio source ([6ab6de4](https://github.com/VforVitorio/F1-StratLab/commit/6ab6de46c2fe5991e37b3c2c6f2683eacbda4f1b))
* **diagrams:** rescue strategy pipeline flow drawio source ([33b368c](https://github.com/VforVitorio/F1-StratLab/commit/33b368cd34a0e960af39fa89c83695fca5ba2706))
* **diagrams:** rescue subprocess launch sequence drawio source ([0a44781](https://github.com/VforVitorio/F1-StratLab/commit/0a44781a2f5825ceae06c7a9e969d7d3573f0d0d))
* **diagrams:** rescue system architecture drawio source ([c3779cb](https://github.com/VforVitorio/F1-StratLab/commit/c3779cbb6b5a5a15f94409d3a6857868166189bc))
* **diagrams:** rescue TCP broadcast dataflow drawio source ([f6ac9e5](https://github.com/VforVitorio/F1-StratLab/commit/f6ac9e5027a60facfa50f89f98f8084d1800689c))
* **pages:** point meet-the-author HF link at dataset URL ([5f56064](https://github.com/VforVitorio/F1-StratLab/commit/5f560640ab435b563999d49808b63a2bfc3775bb))
* **readme:** add status badges and link to docs.f1stratlab.com ([922f462](https://github.com/VforVitorio/F1-StratLab/commit/922f4626a9d1f42b8698b03f335fb3f907729a47))

## [1.3.1](https://github.com/VforVitorio/F1-StratLab/compare/v1.3.0...v1.3.1) (2026-05-12)


### Bug Fixes

* **docs:** repair slate scheme contrast and add brand-aligned theme variables ([6516709](https://github.com/VforVitorio/F1-StratLab/commit/6516709ffbe2d6ff36f8b96f0c0aca7b9ea58512))


### Documentation

* add architecture hub landing with sequence diagram and key contracts ([3fb717e](https://github.com/VforVitorio/F1-StratLab/commit/3fb717ea90c8fbb335d2b19dfebba9d8785fc8a4))
* add branded 404 page with hero styling and recovery links ([188491b](https://github.com/VforVitorio/F1-StratLab/commit/188491b64c67527aa320358c059badcbb7e27e90))
* add CI/CD pipeline narrative covering branching releases and deployment ([db48476](https://github.com/VforVitorio/F1-StratLab/commit/db48476ba7db5198981e01803c7a79b6cfd34855))
* add development hub landing with conventional commits cheat sheet ([5e28bb1](https://github.com/VforVitorio/F1-StratLab/commit/5e28bb11e2f09233021013894a170f2831cecf9e))
* redesign landing with hero, agent grid, mermaid system diagram and stats ([e6525cc](https://github.com/VforVitorio/F1-StratLab/commit/e6525ccb7e4f51c391d31c5ca8aee4112c3ead7c))
* ship docs site revamp with CI/CD narrative and contrast fix ([b64ce12](https://github.com/VforVitorio/F1-StratLab/commit/b64ce12471fd1ecc0cd4272f403ca079224e0f3c))

## [1.3.0](https://github.com/VforVitorio/F1-StratLab/compare/v1.2.0...v1.3.0) (2026-05-12)


### Features

* **docs:** point GitHub Pages at docs.f1stratlab.com via CNAME ([52b3238](https://github.com/VforVitorio/F1-StratLab/commit/52b3238454e1ac702cc80d4976ed55a1809ed818))
* **docs:** update mkdocs site_url to docs.f1stratlab.com ([cf6cda6](https://github.com/VforVitorio/F1-StratLab/commit/cf6cda6679d1eb14d3624046fa812484320acc7c))
* **docs:** wire docs.f1stratlab.com custom domain ([1fcf6ea](https://github.com/VforVitorio/F1-StratLab/commit/1fcf6ea756a661d554676b0359b018adfba1f187))

## [1.2.0](https://github.com/VforVitorio/F1-StratLab/compare/v1.1.1...v1.2.0) (2026-05-12)


### Features

* **docs:** add F1 StratLab brand theme and external-image hook ([0dc3db1](https://github.com/VforVitorio/F1-StratLab/commit/0dc3db1f84540864c8ac2d20a5452d5ca1ddd31f))
* **docs:** launch mkdocs-material docs site with F1 StratLab branding ([7f7006d](https://github.com/VforVitorio/F1-StratLab/commit/7f7006de940dc8c7054a6f119e907a75119ab4b7))


### Documentation

* add landing, getting started, thesis results and maintenance pages ([defed4e](https://github.com/VforVitorio/F1-StratLab/commit/defed4e689508bcbce397292f1bbc8469abfcad8))

## [1.1.1](https://github.com/VforVitorio/F1-StratLab/compare/v1.1.0...v1.1.1) (2026-05-12)


### Bug Fixes

* **ci:** use PAT for release-please so required checks run on release PRs ([b293342](https://github.com/VforVitorio/F1-StratLab/commit/b29334288c1c852dd0617b337de6451fd187652a))
* **ci:** use PAT for release-please so required checks run on release PRs ([e276d45](https://github.com/VforVitorio/F1-StratLab/commit/e276d454d01d2a4fbc11d784aed667247290ab30))


### Documentation

* add data/eval/README inventory for benchmark outputs ([1f533fb](https://github.com/VforVitorio/F1-StratLab/commit/1f533fbcb7d935bac99c670b1dbf62f1d39528b6))
* add data/rag_eval/README for the RAG ground-truth set ([f0c54a1](https://github.com/VforVitorio/F1-StratLab/commit/f0c54a1ec19c8eb7ae105ff7028df1044415cf6e))
* add documents/images/README manifest for thesis figures ([f7c25d3](https://github.com/VforVitorio/F1-StratLab/commit/f7c25d3d06d2c86a41a594e0e780b98c9d47c961))
* document Conventional Commits convention in CONTRIBUTING ([78049cb](https://github.com/VforVitorio/F1-StratLab/commit/78049cb0551bf8a6bdad5842aa2738d4c324b644))
* seed CHANGELOG retroactively with release-please marker ([c870bcc](https://github.com/VforVitorio/F1-StratLab/commit/c870bcc27cfa8fcd52c80a1db24c577fdefb00ff))

## [1.1.0] - 2026-05-11

Benchmark suite for the TFG thesis chapter 5 plus full English localization of
strategy notebooks, scripts and evaluation artefacts. No model retraining and
no breaking changes to runtime APIs.

- Four standalone benchmark scripts under `scripts/bench_*.py` with a shared
  `BenchResult` dataclass and Rich panel layout: pace baselines vs production
  XGBoost (MAE matches the 0.4104 s anchor within +/-0.001 s), Whisper turbo
  latency (P50 / P95 / mean), six sub-agent latency on a Suzuka 2025 fixture,
  and the sentiment + intent + NER pipeline on CPU and GPU.
- New `notebooks/agents/N33_thresholds_and_calibration.ipynb` with
  precision-recall sweeps for overtake (N12), safety car (N14) and undercut
  (N16), plus MC Dropout empirical coverage on the 20,284 tire-degradation
  sequences.
- New `notebooks/agents/N30B_rag_benchmark.ipynb` evaluating BGE-M3 1024d
  (production), MiniLM-L6-v2 384d and BGE-M3 chunk 256 over 15 ground-truth
  queries with Precision@1 / 3 / 5, MRR and latency.
- Figures relocated to `documents/images/05_results/` (300 DPI), CSV + Markdown
  bench outputs under `data/eval/` and `data/rag_eval/`.
- `jiwer>=3.0.0` added to `pyproject.toml` as a forward-looking dependency.
  All bench scripts pass `ruff check` and `ruff format --check` on CI.
- Console entry points (`f1-strat`, `f1-sim`, `f1-arcade`, `f1-streamlit`)
  unchanged from v1.0.0.

## [1.0.0] - 2026-04-20

First stable release. Ships the three-window arcade experience, the full
seven-model ML stack and the N25 to N31 multi-agent LangGraph orchestrator
with FIA RAG over Qdrant.

- Three surfaces from one install: `f1-sim` CLI, `f1-arcade` three-window
  replay (2D circuit + PySide6 strategy dashboard + live telemetry window)
  and `f1-streamlit` post-race dashboard.
- Arcade runs the strategy pipeline locally without the FastAPI backend.
- Per-agent model outputs rendered live: lap time predicted vs actual with CI
  band, tire cliff percentiles, overtake and SC probabilities, stop duration
  percentiles, radio intents and regulation snippets.
- Six-tab reasoning panel with syntax-highlighted LLM narratives for each
  sub-agent plus the N31 orchestrator.
- Live telemetry window with 2x2 delta / speed / brake / throttle grid and
  rival overlay in two-driver mode.
- README slimmed to 85 lines with landing page link and F1 trademark
  disclaimer. Docs reorganised under `docs/arcade` plus five drawio
  architecture diagrams.
- Install via `uv tool install git+https://github.com/VforVitorio/F1_Strat_Manager.git`.

## [0.12.0] - 2026-04-15

Interfaces and distribution milestone. Closes R3 (Streamlit + Backend) and
lands infrastructure for R2 (Arcade). The CLI (R1) stays untouched.

- Voice chat full rewrite: STT migrated from Nemotron to
  `openai/whisper-small` via transformers pipeline; TTS migrated from Qwen3
  to edge-tts with a curated four-voice catalogue (Aria, Guy, Ryan, Sonia);
  LLM is now provider-agnostic via `F1_LLM_PROVIDER`.
- Voice chat UI redesigned end-to-end: Material icons, triadic palette,
  audio-reactive orb, native `st.audio_input` replacing the third-party
  recorder, voice selector dropdown wired end-to-end, health-check polling
  with spinner during cold starts.
- Chat charts: `lap_times` and `race_data` now show tyre compound on hover
  with per-driver pit-stop vlines annotated `DRIVER - COMPOUND`. Shared
  `COMPOUND_COLORS` palette mirrors the Rich palette used by the CLI.
- New `POST /api/v1/strategy/simulate` SSE endpoint streaming start / lap /
  summary events; ready for Arcade consumption.
- Breaking: `streamlit` bumped to `>=1.37`, `audio-recorder-streamlit`
  removed from deps. Backend Dockerfile now installs `ffmpeg` and
  `libsndfile1` for browser WebM decoding.

## [0.11.0] - 2026-03-30

Multi-agent system complete plus the RAG regulation layer. Seven specialized
agents coordinate under a Strategy Orchestrator to produce real-time pit
strategy recommendations from live race data.

- N25 Pace Agent (XGBoost lap time + bootstrap CI), N26 Tire Agent (TCN with
  MC Dropout), N27 Race Situation Agent (LightGBM overtake plus safety car
  prior), N28 Pit Strategy Agent (pit duration quantiles plus undercut
  scorer), N29 Radio Agent (RoBERTa sentiment + SetFit intent + BERT-large
  NER + RCM parser), N30 RAG Agent (Qdrant + BGE-M3) and N31 Strategy
  Orchestrator (three-layer MoE-style routing into Monte Carlo simulation
  into GPT-4o structured synthesis).
- `scripts/build_rag_index.py` indexes the FIA Sporting Regulations into
  2,279 BGE-M3 chunks. Retrieval scores 0.62 to 0.76 on demo queries.
- `src/rag/retriever.py` exports `RagRetriever` and `query_rag_tool` as
  reusable LangChain components imported by N31.
- GitHub Actions CI added: lint (ruff), typecheck (mypy), tests (pytest).
- SRP refactors across every agent notebook plus LangGraph computation graph
  visualization cells.

## [0.10.0] - 2026-03-22

Multi-agent infrastructure milestone. Two of seven sub-agents complete plus
the full RAG indexing pipeline and the importable `src/rag/` module.

- N25 Pace Agent wraps the N06 XGBoost model as a LangGraph ReAct agent and
  returns `PaceOutput` (lap time + delta vs session median + bootstrap CI
  P10 / P90 with N=200).
- N30 RAG Agent runs retrieval-augmented generation over FIA Sporting
  Regulations 2023 to 2025. Embedding via `BAAI/bge-m3` (1024-dim), Qdrant
  local vector store, 2,279 indexed chunks.
- First active `src/` module outside telemetry: `src/rag/` exposes
  `RagRetriever` (singleton via `get_retriever()`) and the `query_rag_tool`
  LangChain tool.
- `scripts/download_fia_pdfs.py` scrapes FIA PDF URLs via `DownloadConfig`.
  `scripts/build_rag_index.py` performs PDF chunking, embedding and Qdrant
  upsert with hash-based deduplication.
- README files added for `src/rag/`, `src/agents/`, `src/nlp/`,
  `src/strategy/` and `src/data_extraction/` covering API surface and
  legacy status.

## [0.9.0] - 2026-03-17

NLP pipeline complete. All notebooks N17 to N24 shipped; the radio analysis
pipeline is operational and integrated into the unified inference entry
point used by the Strategy Agent.

- N17 labels 659 messages (610 clean after manual inspection of 49 post-race
  removals). N18 runs Whisper turbo ASR. N19 establishes a VADER rule-based
  baseline.
- N20 fine-tunes RoBERTa-base for three-class sentiment. N21 uses SetFit
  with ModernBERT-base for five-class intent (370 examples). N22 fine-tunes
  BERT-large CoNLL-03 with BIO tagging for nine F1 entity types
  (weighted F1 = 0.42 on 399 examples). N23 ships a deterministic
  rule-based RCM parser covering 25 event types with 100% Flag / DRS / SC
  coverage.
- N24 unified pipeline exposes `run_pipeline(text)` for team radio and
  `run_rcm_pipeline(rcm_row)` for race control messages on a single JSON
  schema. GPU end-to-end latency: mean 47.8 ms, P95 59.4 ms.
- Model weights and configs uploaded to
  `VforVitorio/f1-strategy-models` on Hugging Face, plus the N16 undercut
  artefacts that were missing from v0.8.1.

## [0.8.1] - 2026-03-13

Strategy ML suite: pit-stop prediction and undercut intelligence.

- N15 Pit Stop Duration: HistGradientBoostingRegressor at P05 / P50 / P95
  on the normal physical window of 2.0 to 4.5 s. P50 MAE 0.487 s vs
  baseline 0.555 s. Coverage P05 to P95 is 70.5% on the test set.
- N16 Undercut Success: LightGBM binary classifier on 1,032 labeled
  pair-laps (2023 to 2025) with DRY_COMPOUNDS filter. AUC-PR 0.6739,
  AUC-ROC 0.7708, Platt-calibrated threshold 0.522. SHAP top features:
  `pos_gap_at_pit`, `pace_delta`, `circuit_undercut_rate`,
  `tyre_life_diff`.
- N12B Causal TCN Overtake archived as a valid negative result
  (AUC-PR ~0.10 vs N12's 0.5491). Confirms feature-engineered LightGBM
  wins on this dataset.
- Roadmap lists N17 to N24 for the upcoming NLP radio pipeline.

## [0.7.0] - 2026-03-05

ML foundation phase closes out. Two predictive models trained, validated on
held-out 2025 data and exported under `data/models/`.

- N06 Lap Time Predictor: XGBoost delta-lap-time model with circuit
  clustering features, trained on 2023 to 2024 and tested on 2025.
  MAE 0.392 s. Features include fuel-corrected lap time, tyre life,
  compound, circuit cluster and race phase.
- N07 to N10 Tire Degradation Predictor: Temporal Convolutional Network in
  PyTorch with per-compound fine-tuning (SOFT / MEDIUM / HARD) and MC
  Dropout for uncertainty (N=50 forward passes at inference). Calibration
  JSON exported alongside the model weights.
- `src/` module integration deferred to v0.9.0 (post-notebook phase). Tire
  compound mapping to C1 through C5 flagged as a future enhancement.

## [0.6.0] - 2026-02-12

Data engineering phase closes out. End-to-end pipeline from raw FastF1
telemetry to a clean feature-rich dataset ready to feed the ML models.

- Repo restructure: previous notebooks and code moved to `legacy/` to
  preserve the original work. New structure built around the TFG
  architecture: `notebooks/data_engineering/`, `notebooks/strategy/`,
  `src/strategy/`, `src/agents/`, `src/telemetry/`.
- N01 download pipeline extended to support the 2025 season alongside 2023
  to 2024. FastF1 naming inconsistencies aliased (Miami_Gardens, Spain
  vs Barcelona) for canonical cross-season names.
- N03 circuit clustering: K-Means with k=4 fitted on 2023 to 2024 and
  serialized with joblib. 2025 inference runs `kmeans.predict()` on the
  saved model without refitting. Las Vegas missing speed-trap data imputed
  with training means from the scaler.
- N04 feature engineering: 48-column dataset across ~45,000 clean racing
  laps. Fuel-corrected degradation (0.055 s/lap from Pirelli literature),
  sequential lap features, rolling 3-lap degradation rate via polyfit
  clipped to +/-2 s/lap, race-context fields, circuit cluster merge from
  N03. 2025 saved as a held-out test set.
- Dataset published to `VforVitorio/f1-strategy-dataset` on Hugging Face;
  `scripts/download_data.py` pulls everything locally.

## [0.1.1] - 2026-04-09

First CLI release (R1 milestone). Distributed as the
`f1_strat_manager-0.1.1-py3-none-any.whl` wheel.

- Seven-agent multi-agent system (N25 to N31) on LangGraph.
- `f1-sim` CLI simulation with Rich Live rendering.
- No-LLM mode (ML + Monte Carlo simulation only).
- OpenF1 radio corpus with Whisper transcription.
- F1 strategic guard-rails baked into every sub-agent.
- Lazy Hugging Face data download on first run.
- Eight ML models (pace, tire degradation, overtake, safety car, pit
  duration, undercut) plus the NLP pipeline (sentiment, intent, NER) and
  RAG over FIA regulations.

[1.1.0]: https://github.com/VforVitorio/F1-StratLab/releases/tag/v1.1.0
[1.0.0]: https://github.com/VforVitorio/F1-StratLab/releases/tag/v1.0.0
[0.12.0]: https://github.com/VforVitorio/F1-StratLab/releases/tag/v0.12.0
[0.11.0]: https://github.com/VforVitorio/F1-StratLab/releases/tag/v0.11.0
[0.10.0]: https://github.com/VforVitorio/F1-StratLab/releases/tag/v0.10.0
[0.9.0]: https://github.com/VforVitorio/F1-StratLab/releases/tag/v0.9.0
[0.8.1]: https://github.com/VforVitorio/F1-StratLab/releases/tag/v0.8.1
[0.7.0]: https://github.com/VforVitorio/F1-StratLab/releases/tag/v0.7.0
[0.6.0]: https://github.com/VforVitorio/F1-StratLab/releases/tag/v0.6
[0.1.1]: https://github.com/VforVitorio/F1-StratLab/releases/tag/v0.1.1
