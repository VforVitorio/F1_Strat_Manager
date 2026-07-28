# Changelog

All notable changes to this project are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

From v1.2.0 onwards this file is maintained automatically by
[release-please](https://github.com/googleapis/release-please). Anything
above v1.1.0 was seeded retroactively from the GitHub Releases history.

<!-- next-version-placeholder -->

## [2.5.0](https://github.com/VforVitorio/F1-StratLab/compare/v2.4.0...v2.5.0) (2026-07-28)


### Features

* **arcade:** show the decision-memory block when the call changes ([6877788](https://github.com/VforVitorio/F1-StratLab/commit/687778820b897aba724bed1a666682f2c120f827)), closes [#694](https://github.com/VforVitorio/F1-StratLab/issues/694)
* **backend:** bump telemetry to the lap payload carrying the decision-memory block ([c4d7bb4](https://github.com/VforVitorio/F1-StratLab/commit/c4d7bb4d52f0aa20fd580fbcb9cadab97dcdeb5d)), closes [#694](https://github.com/VforVitorio/F1-StratLab/issues/694)
* **cli:** show the decision-memory block when the call changes ([41dd804](https://github.com/VforVitorio/F1-StratLab/commit/41dd8041ba71e1056a9b805873ec956ad134bfb1)), closes [#694](https://github.com/VforVitorio/F1-StratLab/issues/694)
* **memory:** expose whether the last recorded call changed ([7b69c13](https://github.com/VforVitorio/F1-StratLab/commit/7b69c1358ce8f5535667ceaae4fc4b02ef93a7b3)), closes [#694](https://github.com/VforVitorio/F1-StratLab/issues/694)


### Bug Fixes

* **tests:** skip the payload tests when model weights are absent ([4aa4424](https://github.com/VforVitorio/F1-StratLab/commit/4aa4424e432171730a8b5539f2d440ae9d596093))

## [2.4.0](https://github.com/VforVitorio/F1-StratLab/compare/v2.3.0...v2.4.0) (2026-07-28)


### Features

* **engine:** accept a caller-owned DecisionMemory and render it into the prompt ([10e4832](https://github.com/VforVitorio/F1-StratLab/commit/10e4832429d39d6c5285ee8fabce73b510bdc67b)), closes [#684](https://github.com/VforVitorio/F1-StratLab/issues/684)
* **orchestrator:** render a decision-memory block in the Layer 3 prompt ([5f8f789](https://github.com/VforVitorio/F1-StratLab/commit/5f8f789d378250b712dd87128728cf7e5e4a71a0)), closes [#680](https://github.com/VforVitorio/F1-StratLab/issues/680)
* **orchestrator:** scope the STAY_OUT continuation framing to actual holds ([282f668](https://github.com/VforVitorio/F1-StratLab/commit/282f668c8f4265e61c65334a300cc55db3676b75)), closes [#685](https://github.com/VforVitorio/F1-StratLab/issues/685)
* **surfaces:** give the CLI, arcade and backend a per-race decision memory ([c463c8c](https://github.com/VforVitorio/F1-StratLab/commit/c463c8c0a7e956679d789131c06b0196dbd13f16)), closes [#684](https://github.com/VforVitorio/F1-StratLab/issues/684)


### Documentation

* **audit:** record what Sprint 2 built and the three findings the build produced ([349384f](https://github.com/VforVitorio/F1-StratLab/commit/349384f409d4edc8ae0e4934f49467929593e745))
* **memory:** say precisely when the block does and does not change the call ([b9a93c5](https://github.com/VforVitorio/F1-StratLab/commit/b9a93c52e3532be0bd6c8f3ffe9f734148a72194))
* **orchestrator:** declare that /recommend and MCP have no decision memory ([243e6d2](https://github.com/VforVitorio/F1-StratLab/commit/243e6d2f81ed1f05d4c255ec9e9a1097b184ea50)), closes [#681](https://github.com/VforVitorio/F1-StratLab/issues/681)

## [2.3.0](https://github.com/VforVitorio/F1-StratLab/compare/v2.2.4...v2.3.0) (2026-07-27)


### Features

* **strategy:** add the per-race decision memory accumulator ([7ffd527](https://github.com/VforVitorio/F1-StratLab/commit/7ffd5277e008611586a0df05500b32cddeca02ad))


### Bug Fixes

* **agents:** band threat_level on the served calibrated scale, not raw-scale operating points ([ec0601e](https://github.com/VforVitorio/F1-StratLab/commit/ec0601e3aa2ca0876c020d199b2bb60257429fad)), closes [#665](https://github.com/VforVitorio/F1-StratLab/issues/665)
* **orchestrator:** warn when the client discards the requested temperature ([169376e](https://github.com/VforVitorio/F1-StratLab/commit/169376e8bfda15fc3782a8cd5edf5b50c2ec0a34))


### Documentation

* **audits:** add the adversarial audit of the orchestrator memory layer ([25e8566](https://github.com/VforVitorio/F1-StratLab/commit/25e85668742c05bfc18909544ef8d4f997221bca))

## [2.2.4](https://github.com/VforVitorio/F1-StratLab/compare/v2.2.3...v2.2.4) (2026-07-27)


### Bug Fixes

* **orchestrator:** frame a held STAY_OUT as monitoring instead of as a blocked pit ([3f0a118](https://github.com/VforVitorio/F1-StratLab/commit/3f0a118460a934df304670fa74ae8a726543480b))


### Documentation

* **mc:** price the lap-number neutralisation union in all three places it lives ([86c4659](https://github.com/VforVitorio/F1-StratLab/commit/86c4659c04e1f721fd1e5d307043269d9d3543c8))

## [2.2.3](https://github.com/VforVitorio/F1-StratLab/compare/v2.2.2...v2.2.3) (2026-07-27)


### Bug Fixes

* **engine:** repath the test files the module docstring names after the move ([2502ec3](https://github.com/VforVitorio/F1-StratLab/commit/2502ec3be8e8a1e5ad999226515f9534297e2180))
* **mc:** break an exact tie by a stated rule instead of by dict insertion order ([e1723bc](https://github.com/VforVitorio/F1-StratLab/commit/e1723bcb1411fcf9ba9df10f133b6f16bf37dcb1)), closes [#645](https://github.com/VforVitorio/F1-StratLab/issues/645)


### Documentation

* **audits:** record the MONITOR layer audit and the decision not to build it ([185d0f5](https://github.com/VforVitorio/F1-StratLab/commit/185d0f5335c6882c53675a4460552b5cd13adb47))

## [2.2.2](https://github.com/VforVitorio/F1-StratLab/compare/v2.2.1...v2.2.2) (2026-07-26)


### Bug Fixes

* **agents,arcade:** land six filed behaviour bugs and kill the constant that kept drifting ([4000f84](https://github.com/VforVitorio/F1-StratLab/commit/4000f84e3f9f7a1d9363ab6fd8f9a1af1fba6c04)), closes [#613](https://github.com/VforVitorio/F1-StratLab/issues/613) [#614](https://github.com/VforVitorio/F1-StratLab/issues/614) [#615](https://github.com/VforVitorio/F1-StratLab/issues/615) [#616](https://github.com/VforVitorio/F1-StratLab/issues/616) [#620](https://github.com/VforVitorio/F1-StratLab/issues/620) [#628](https://github.com/VforVitorio/F1-StratLab/issues/628)
* **arcade:** widen the weather catch and stop claiming a drift is closed when it is not ([3e56432](https://github.com/VforVitorio/F1-StratLab/commit/3e56432b527216cc5ee3877aa03e2971cc03123c))
* **eval:** measure the RCM parser that ships, not a private copy of it ([10311f9](https://github.com/VforVitorio/F1-StratLab/commit/10311f9300c9a3e9ab1945452ec596d19c02ecba)), closes [#632](https://github.com/VforVitorio/F1-StratLab/issues/632)
* **eval:** resolve team rebrands in the pit holdout and make an unknown team loud ([9366346](https://github.com/VforVitorio/F1-StratLab/commit/9366346202f7f6df4ff85a920c4d74940da12aa6)), closes [#629](https://github.com/VforVitorio/F1-StratLab/issues/629)
* **nlp:** classify the black-and-white flag, which the copy we replaced could and we could not ([fb8c208](https://github.com/VforVitorio/F1-StratLab/commit/fb8c208f8b83aaa98143554d07fb28c226453e05)), closes [#641](https://github.com/VforVitorio/F1-StratLab/issues/641)
* **tests:** correct a golden that was born red, and say why CI could never tell us ([fdf49bb](https://github.com/VforVitorio/F1-StratLab/commit/fdf49bb90e147b3d5cd28a0f0711e0302f31f7fc)), closes [#634](https://github.com/VforVitorio/F1-StratLab/issues/634)


### Refactoring

* **arcade:** name the unit conversion the previous commit claimed to have named ([8c419f8](https://github.com/VforVitorio/F1-StratLab/commit/8c419f8043e194390196fced64695744019053b1))
* **readability:** apply the four findings that earn it and close the audit ([9d71873](https://github.com/VforVitorio/F1-StratLab/commit/9d71873ca5f6ae876a96ecf364d75704ac4f9454)), closes [#643](https://github.com/VforVitorio/F1-StratLab/issues/643)

## [2.2.1](https://github.com/VforVitorio/F1-StratLab/compare/v2.2.0...v2.2.1) (2026-07-26)


### Bug Fixes

* **agents:** define DEFAULT_RACING_LAPS_UNDER_VSC and turn on the check that would have caught it ([14093cc](https://github.com/VforVitorio/F1-StratLab/commit/14093cc7c9c7a6ddcf974358dcf315115fb53feb)), closes [#619](https://github.com/VforVitorio/F1-StratLab/issues/619)
* **domain:** correct eight places that assert a stronger F1 fact than the code measures ([f1c3f1c](https://github.com/VforVitorio/F1-StratLab/commit/f1c3f1cc9a4f830c460dcd08f873660689e08b10)), closes [#617](https://github.com/VforVitorio/F1-StratLab/issues/617)


### Documentation

* **src:** fix contradicting docstrings, thin prose, and LLM-sounding wording ([3c6f038](https://github.com/VforVitorio/F1-StratLab/commit/3c6f0383d21708fb6e52a7b6e494a78cf9f8ba01)), closes [#621](https://github.com/VforVitorio/F1-StratLab/issues/621)

## [2.2.0](https://github.com/VforVitorio/F1-StratLab/compare/v2.1.2...v2.2.0) (2026-07-26)


### Features

* **eval:** publish the projection accuracy and the measured tables as a report ([00e66ba](https://github.com/VforVitorio/F1-StratLab/commit/00e66babd39ecc07dc0bb80fb7d1aa18dc4b38d1)), closes [#609](https://github.com/VforVitorio/F1-StratLab/issues/609)


### Bug Fixes

* **engine:** thread cliff_p50 and total_laps so the stint-end guard is not dead ([0c95c1b](https://github.com/VforVitorio/F1-StratLab/commit/0c95c1b5eaa2fea3eabbcb576e9087ddcccb7f48)), closes [#566](https://github.com/VforVitorio/F1-StratLab/issues/566)

## [2.1.2](https://github.com/VforVitorio/F1-StratLab/compare/v2.1.1...v2.1.2) (2026-07-26)


### Documentation

* bring the drawio sources back in line with the code ([9733ca8](https://github.com/VforVitorio/F1-StratLab/commit/9733ca8a842216c2ee26683ba0f0b82e0b2026e2)), closes [#592](https://github.com/VforVitorio/F1-StratLab/issues/592)

## [2.1.1](https://github.com/VforVitorio/F1-StratLab/compare/v2.1.0...v2.1.1) (2026-07-26)


### Bug Fixes

* **readme:** animate the CLI demo instead of showing a still frame ([34d16c6](https://github.com/VforVitorio/F1-StratLab/commit/34d16c6cce77fbdb5fb456f129ec43f69b0b50f0))


### Documentation

* **agents:** correct the agent signatures and delete a ritual that recreates deleted drift ([18c2933](https://github.com/VforVitorio/F1-StratLab/commit/18c2933c97a5d098e08b5bb71d522e0764fb5e8c)), closes [#585](https://github.com/VforVitorio/F1-StratLab/issues/585)
* caveat the two-compound article, and state what the API lap_state really carries ([5a0bb3f](https://github.com/VforVitorio/F1-StratLab/commit/5a0bb3f89d00ef50d84216d06eac83490244251c)), closes [#590](https://github.com/VforVitorio/F1-StratLab/issues/590)
* correct the package READMEs against the packages they describe ([c6ed75c](https://github.com/VforVitorio/F1-StratLab/commit/c6ed75c9c559a844837bc5e9b10c8559b63826cb)), closes [#589](https://github.com/VforVitorio/F1-StratLab/issues/589)
* diagram the three things the docs site had no picture of ([a1045f7](https://github.com/VforVitorio/F1-StratLab/commit/a1045f776933934bb884b71c897de610152e3c6b)), closes [#592](https://github.com/VforVitorio/F1-StratLab/issues/592)
* make the first example runnable, and stop describing fields that do not exist ([e6d52a3](https://github.com/VforVitorio/F1-StratLab/commit/e6d52a3b5a2597bd725c8cb963b03ef6e91f5d9a)), closes [#588](https://github.com/VforVitorio/F1-StratLab/issues/588) [#567](https://github.com/VforVitorio/F1-StratLab/issues/567)
* remove em-dashes from prose across every documentation surface ([eb5c380](https://github.com/VforVitorio/F1-StratLab/commit/eb5c380f377dae55338d26241889b20f3abccced)), closes [#594](https://github.com/VforVitorio/F1-StratLab/issues/594)
* send contributors at dev, and repair six links that resolve nowhere ([0d5983a](https://github.com/VforVitorio/F1-StratLab/commit/0d5983ada82c0e8384a4ad8f4beb5a6212af16be))
* stop describing Streamlit as a live surface, and document the one that replaced it ([f01bd55](https://github.com/VforVitorio/F1-StratLab/commit/f01bd55e811188b3f1198d7590c4e3e0b297639c)), closes [#587](https://github.com/VforVitorio/F1-StratLab/issues/587)

## [2.1.0](https://github.com/VforVitorio/F1-StratLab/compare/v2.0.1...v2.1.0) (2026-07-25)


### Features

* **cli:** retire f1-streamlit and add the f1-webapp launcher ([d9151d4](https://github.com/VforVitorio/F1-StratLab/commit/d9151d455abccfa3255a4de8ea369a28573ed98e)), closes [#551](https://github.com/VforVitorio/F1-StratLab/issues/551)
* **mc:** measure the tables the projection layer needs ([48092d6](https://github.com/VforVitorio/F1-StratLab/commit/48092d6bdeec2ac91dfdacebc4ed22699c7aaf9a)), closes [#553](https://github.com/VforVitorio/F1-StratLab/issues/553)
* **mc:** name the target we will actually be racing ([f36464e](https://github.com/VforVitorio/F1-StratLab/commit/f36464e0219f0326637c017981d3a836239e8219))
* **mc:** price clean air per circuit so the overcut becomes a real move ([0e30689](https://github.com/VforVitorio/F1-StratLab/commit/0e30689b0fa12656411aff7f79f39fe998ea56a5)), closes [#550](https://github.com/VforVitorio/F1-StratLab/issues/550)
* **mc:** price the lap an overcut spends waiting for a neutralisation ([8044fd5](https://github.com/VforVitorio/F1-StratLab/commit/8044fd52beb57ab24396f9e4ba3f9780932ecdb8)), closes [#550](https://github.com/VforVitorio/F1-StratLab/issues/550)
* **mc:** project track position from per-rival gaps ([7f91d1d](https://github.com/VforVitorio/F1-StratLab/commit/7f91d1de94b0807ae1f8c04f0279487c804e539c)), closes [#554](https://github.com/VforVitorio/F1-StratLab/issues/554)
* **mc:** score the candidates in projected track position ([8052343](https://github.com/VforVitorio/F1-StratLab/commit/8052343368bdaf20698828216f77d338ab1c8175)), closes [#555](https://github.com/VforVitorio/F1-StratLab/issues/555)
* **mc:** thread race context to the MC boundary ([fa0932b](https://github.com/VforVitorio/F1-StratLab/commit/fa0932b67e3e787af916fefc1a73524d6cb5708a)), closes [#552](https://github.com/VforVitorio/F1-StratLab/issues/552)
* **mc:** wire the projection into every surface ([97514b4](https://github.com/VforVitorio/F1-StratLab/commit/97514b4812ed9e96e117586cbf88af9f26e877e7)), closes [#556](https://github.com/VforVitorio/F1-StratLab/issues/556)


### Bug Fixes

* **ci:** pin ruff so a linter release cannot turn every branch red ([7e69340](https://github.com/VforVitorio/F1-StratLab/commit/7e69340bc4dea4c6954d75dd5da61d0e90a3f48a))
* **data:** share one FastF1 cache and stop download_data pulling the whole hub ([cc612ad](https://github.com/VforVitorio/F1-StratLab/commit/cc612adb8555f9d786a924e893819039e75f4a7d))
* **deps:** drop python-jose, which nothing imported ([8993b69](https://github.com/VforVitorio/F1-StratLab/commit/8993b69bceb0eff5afbd6c8861ee81a4f6a4c389))
* **docs:** correct two published diagram errors ([e040fcf](https://github.com/VforVitorio/F1-StratLab/commit/e040fcf2d5aead66b4c6d6281a4df472d0e66d93))
* **mc:** bound the window by the race end and gate the undercut by regime ([6767490](https://github.com/VforVitorio/F1-StratLab/commit/67674905282a9fcc5bd2d6c2902b481ebcfc8ec8)), closes [#550](https://github.com/VforVitorio/F1-StratLab/issues/550)
* **mc:** floor the hazard, resolve every circuit spelling, restore the O(1) lap state ([f1563c7](https://github.com/VforVitorio/F1-StratLab/commit/f1563c7108f7dd8f30b296cc2dcd1f268be4c585)), closes [#550](https://github.com/VforVitorio/F1-StratLab/issues/550)
* **mc:** reject NaN, honour an explicit zero, and project on every surface ([bb4b4c6](https://github.com/VforVitorio/F1-StratLab/commit/bb4b4c64b1030cdfac90ca05f8f28d5350ade1c8)), closes [#550](https://github.com/VforVitorio/F1-StratLab/issues/550)


### Documentation

* bring every surface up to the shipped Monte Carlo and data layer ([e5e5882](https://github.com/VforVitorio/F1-StratLab/commit/e5e58826fd501da36154dfb1203668754dbe21f8))
* **mc:** add the waiting term to the overcut section ([4a9d527](https://github.com/VforVitorio/F1-StratLab/commit/4a9d527af2de0c3600d9ad71628e2a66fb4d4ec1))
* **mc:** describe what the Monte Carlo now scores ([fbbae7f](https://github.com/VforVitorio/F1-StratLab/commit/fbbae7f110715a17f00605dcc2dc487b91182667)), closes [#557](https://github.com/VforVitorio/F1-StratLab/issues/557)
* **mc:** describe where the overcut works instead of calling it a limitation ([1fab451](https://github.com/VforVitorio/F1-StratLab/commit/1fab451b28e71e01a3ed604ae027b4202877f664))
* **readme:** restore the arcade hero, pair the CLI and web app demos, add a citation section ([049316e](https://github.com/VforVitorio/F1-StratLab/commit/049316e8a70c1a856d0752ae44a44e4fe456857a))


### Refactoring

* **mc:** name the racing bucket for what it holds ([406a0e8](https://github.com/VforVitorio/F1-StratLab/commit/406a0e810cce2a329ea1f0c47d864116dd1f60b7)), closes [#553](https://github.com/VforVitorio/F1-StratLab/issues/553)

## [2.0.1](https://github.com/VforVitorio/F1-StratLab/compare/v2.0.0...v2.0.1) (2026-07-22)


### Bug Fixes

* **deps:** bump gitpython to 3.1.54 to clear a high advisory ([3ae5af2](https://github.com/VforVitorio/F1-StratLab/commit/3ae5af2a072fea6458e6d56831e2faaa517ff433))


### Documentation

* **site:** expand the roadmap (v2.5 modern arcade, v2.8 rival, v3.0 live) ([7564e73](https://github.com/VforVitorio/F1-StratLab/commit/7564e733b5f01f5eed20b6820edbe74b7a60c28f))
* **site:** mark the v2 migration shipped, refresh author bio and getting-started hero ([c9681b9](https://github.com/VforVitorio/F1-StratLab/commit/c9681b9130c78f97a4c591cda2f775f369947926))


### Refactoring

* retire the voice surface on the parent side; bump telemetry submodule ([d916368](https://github.com/VforVitorio/F1-StratLab/commit/d916368b4b98449ae5742a691fcc15bb0b3ed8b8))

## [2.0.0](https://github.com/VforVitorio/F1-StratLab/compare/v1.10.6...v2.0.0) (2026-07-21)


### Bug Fixes

* **agents:** bound the retired-car lap fallback by presence and make searchable defaults loud ([662102c](https://github.com/VforVitorio/F1-StratLab/commit/662102c337bf0afc32bd7ec3b7622a07196632cd)), closes [#477](https://github.com/VforVitorio/F1-StratLab/issues/477)
* **agents:** guard the pit prompt-build NaN crash and correct the calibration docstring ([df988e7](https://github.com/VforVitorio/F1-StratLab/commit/df988e70ac38ecf03b018a9c1a8f388a24f4f6db))
* **agents:** resolve the pit-agent circuit lookups by slug on the FastF1 path too ([ba0be59](https://github.com/VforVitorio/F1-StratLab/commit/ba0be59bd403f1675543f003e49c9fb542d15ffd))
* **agents:** split Safety Car from Virtual Safety Car in the strategy engine ([4f124cf](https://github.com/VforVitorio/F1-StratLab/commit/4f124cf3c63b8dcd6b3e90e2dd07b9b4d963ee2e)), closes [#471](https://github.com/VforVitorio/F1-StratLab/issues/471)
* **agents:** validate LLM tool inputs and fix model-fidelity defects across the strategy engine ([5f84218](https://github.com/VforVitorio/F1-StratLab/commit/5f842187a3b22f20f313085619f0f8fef703e891))
* **agents:** yellow laps read as green, empty-roster vs unknown, and two raw parquet readers ([52aa560](https://github.com/VforVitorio/F1-StratLab/commit/52aa5605cfe046c57ebf50f8d8c14de0bf016e50)), closes [#486](https://github.com/VforVitorio/F1-StratLab/issues/486)
* **arcade:** stop reading the featured parquet raw ([79ce0ba](https://github.com/VforVitorio/F1-StratLab/commit/79ce0ba1563b52f7a82c6d3f6236237d7ca17594)), closes [#447](https://github.com/VforVitorio/F1-StratLab/issues/447)
* **cli:** derive the data defaults from --year so a 2024 run loads 2024 data ([8951cb3](https://github.com/VforVitorio/F1-StratLab/commit/8951cb3959de8360b4534bae26807f7d39190048)), closes [#443](https://github.com/VforVitorio/F1-StratLab/issues/443)
* **deps:** bump pyasn1 to 0.6.4 to clear two high advisories ([5123d42](https://github.com/VforVitorio/F1-StratLab/commit/5123d42e2e9e4cd4075de53f29ec58e0378353ea))
* **engine:** guard DNF laps in arcade and bump telemetry for rival threading ([3b4606c](https://github.com/VforVitorio/F1-StratLab/commit/3b4606cf2094e199939c406914459ea65da45a98)), closes [#431](https://github.com/VforVitorio/F1-StratLab/issues/431) [#441](https://github.com/VforVitorio/F1-StratLab/issues/441)
* **engine:** thread live_drivers, guard both undercut drivers, and stop promising a test that never existed ([fbcb690](https://github.com/VforVitorio/F1-StratLab/commit/fbcb6909d3bc7a43a91f6d4ba7a62fd0776c89af)), closes [#462](https://github.com/VforVitorio/F1-StratLab/issues/462)
* **N26:** stop overwriting trained TCN features and NaN-fill before scaling ([775bd13](https://github.com/VforVitorio/F1-StratLab/commit/775bd1371d6cc9664902a8f02ea8589a132424b1)), closes [#485](https://github.com/VforVitorio/F1-StratLab/issues/485)
* **orchestrator:** charge OVERCUT for its pit stop so the Monte Carlo decides ([bc1c34a](https://github.com/VforVitorio/F1-StratLab/commit/bc1c34ad05711f6a195669d89c3fac20b9c05036)), closes [#470](https://github.com/VforVitorio/F1-StratLab/issues/470)


### Performance

* **agents:** lazy package __init__ so one agent doesn't load the whole family ([b7f4cda](https://github.com/VforVitorio/F1-StratLab/commit/b7f4cda98ada41f4418ae5d4c590ba7f2f95e75b))


### Documentation

* **agents:** regenerate the N31 Monte Carlo figure with the OVERCUT fix ([305778b](https://github.com/VforVitorio/F1-StratLab/commit/305778baa4e30429b938f487d3379a10cf188b3d)), closes [#470](https://github.com/VforVitorio/F1-StratLab/issues/470)
* **demo:** re-record the web-app demo with the brand favicon; bump telemetry submodule ([efb7a83](https://github.com/VforVitorio/F1-StratLab/commit/efb7a832d83fd10b7470eaec372ec582563e00f1))
* fact-check and refresh the project + site documentation against the current code ([9490fe4](https://github.com/VforVitorio/F1-StratLab/commit/9490fe419f67c398d31a8987be89713e9d71d23e))
* **orchestrator:** correct the post-fix sweep split and a stale [0,1] score claim ([6777998](https://github.com/VforVitorio/F1-StratLab/commit/6777998848ee7d893dd2796be6d0af67715a9aee)), closes [#470](https://github.com/VforVitorio/F1-StratLab/issues/470)
* **readme:** v2 web-app demo GIF + Streamlit-&gt;web-app rewrite; bump telemetry submodule ([586aed9](https://github.com/VforVitorio/F1-StratLab/commit/586aed94883f2b75fb8d112c3d64a907ef44a71e))


### Refactoring

* **agents:** extract _clamp_expected_stint_end so the [#433](https://github.com/VforVitorio/F1-StratLab/issues/433) clamp is CI-testable ([bf1ede8](https://github.com/VforVitorio/F1-StratLab/commit/bf1ede8a6453ef69e595357cbe6ed5129deaff7c))


### Chores

* release 2.0.0 ([5f3804c](https://github.com/VforVitorio/F1-StratLab/commit/5f3804c18e64d03d499522f2eacbc15020fc20b8))

## [1.10.6](https://github.com/VforVitorio/F1-StratLab/compare/v1.10.5...v1.10.6) (2026-07-16)


### Bug Fixes

* **agents:** force what the SC regulation makes certain, not a strategy opinion ([281fb32](https://github.com/VforVitorio/F1-StratLab/commit/281fb3210026c7a52354708f50d3c333c7ef0b97)), closes [#464](https://github.com/VforVitorio/F1-StratLab/issues/464)
* **cli:** feed the PMV the real overtake gap, and give the augmentation one home ([dc9914d](https://github.com/VforVitorio/F1-StratLab/commit/dc9914d21e0667e90fb84b852bd287e589420763))
* **N26:** scope the TCN window to this stint, up to now, and zero-pad as trained ([20a7a06](https://github.com/VforVitorio/F1-StratLab/commit/20a7a068c1de5f328607fc36ae7e2fdfccd71aaf)), closes [#449](https://github.com/VforVitorio/F1-StratLab/issues/449)
* **N28:** refuse to score an undercut against a car that is not racing ([018f910](https://github.com/VforVitorio/F1-StratLab/commit/018f9107abdc5f72a13d0eb895d0bff433ed3628)), closes [#462](https://github.com/VforVitorio/F1-StratLab/issues/462)
* **orchestrator:** stop the LLM overriding the validated undercut target ([fc2c42f](https://github.com/VforVitorio/F1-StratLab/commit/fc2c42f30bf51b96b6b089db28b2a5816f8e696b)), closes [#462](https://github.com/VforVitorio/F1-StratLab/issues/462)


### Documentation

* **multi-agent:** document the SC rails as rules, not as a forced stop ([d5f8709](https://github.com/VforVitorio/F1-StratLab/commit/d5f87094d397fe5e940beb14d0b36547804e1fc8)), closes [#464](https://github.com/VforVitorio/F1-StratLab/issues/464)

## [1.10.5](https://github.com/VforVitorio/F1-StratLab/compare/v1.10.4...v1.10.5) (2026-07-16)


### Bug Fixes

* **agents:** compute FuelEffect from the stint baseline, not a hardcoded fuel_load factor ([7ac17e5](https://github.com/VforVitorio/F1-StratLab/commit/7ac17e59495c4d8a3a8cc735c442cc8efe99139e)), closes [#446](https://github.com/VforVitorio/F1-StratLab/issues/446)
* **agents:** encode N15's compound_id as the ordinal rank it was trained on ([6bb025f](https://github.com/VforVitorio/F1-StratLab/commit/6bb025f2ce62557b154ebf2e952a5210a467d64a)), closes [#445](https://github.com/VforVitorio/F1-StratLab/issues/445)
* **agents:** honour the Safety-Car guard-rail on the final answer, and stop simulating free pit stops ([b423acd](https://github.com/VforVitorio/F1-StratLab/commit/b423acd3d62be44974ea7336ee6c0087604214f6)), closes [#436](https://github.com/VforVitorio/F1-StratLab/issues/436)
* **agents:** let the overtake model see the elapsed-time gap it was trained on ([46c320f](https://github.com/VforVitorio/F1-StratLab/commit/46c320f8a47b23d41a9974a2e98aa3dfc4daab93))
* **agents:** re-key the circuit lookup tables to the slug keyspace the agents query with ([02e7a79](https://github.com/VforVitorio/F1-StratLab/commit/02e7a79d5e76e8b9e7fa54e863409bfc69c25b32)), closes [#448](https://github.com/VforVitorio/F1-StratLab/issues/448)
* **agents:** un-invert the undercut model's top feature and stop feeding it the race lap ([ec0f444](https://github.com/VforVitorio/F1-StratLab/commit/ec0f4448d2737b21146c2abf6349139adb9e52db)), closes [#444](https://github.com/VforVitorio/F1-StratLab/issues/444)
* **engine:** scope the laps frame to the analysed Grand Prix in run_lap ([454db40](https://github.com/VforVitorio/F1-StratLab/commit/454db40bd213a29b4fffd1b4d10c9086093cf364)), closes [#429](https://github.com/VforVitorio/F1-StratLab/issues/429)
* **simulation:** guard the stint-baseline precompute against frames without Stint ([4262859](https://github.com/VforVitorio/F1-StratLab/commit/4262859d8b85f62064dc2a4c34a6dc9824abaf73)), closes [#446](https://github.com/VforVitorio/F1-StratLab/issues/446)


### Documentation

* **agents:** fix the entry-point table and make the examples actually run ([59b02e0](https://github.com/VforVitorio/F1-StratLab/commit/59b02e071b296802f12489f2ebe187e1394d008e)), closes [#438](https://github.com/VforVitorio/F1-StratLab/issues/438)
* **strategy:** correct the pages the engine fixes made wrong ([62fa3c5](https://github.com/VforVitorio/F1-StratLab/commit/62fa3c55c3f03cb65cff280b2bdfbaac35c4f6a1)), closes [#438](https://github.com/VforVitorio/F1-StratLab/issues/438)

## [1.10.4](https://github.com/VforVitorio/F1-StratLab/compare/v1.10.3...v1.10.4) (2026-07-15)


### Bug Fixes

* **deps:** bump click/kafka-python/transformers past CVEs; waive blocked pillow and setuptools ([97d273d](https://github.com/VforVitorio/F1-StratLab/commit/97d273d3bce732b6f772e56c53d41e045427f113))
* **deps:** resolve OSV Python CVE backlog (bump click/kafka-python/transformers; waive blocked pillow and setuptools) ([fbd6df2](https://github.com/VforVitorio/F1-StratLab/commit/fbd6df27ba0f826f33859ae9971b7675a7d78a4a))

## [1.10.3](https://github.com/VforVitorio/F1-StratLab/compare/v1.10.2...v1.10.3) (2026-07-12)


### Bug Fixes

* **nlp:** DOUBLE YELLOW severity + time-penalty routing ([#398](https://github.com/VforVitorio/F1-StratLab/issues/398) follow-up) ([7f3f7a7](https://github.com/VforVitorio/F1-StratLab/commit/7f3f7a79d9ef96eb92538e702ca4d579332101e1))
* **nlp:** RCM parser superset + penalty/red-flag routing + arcade SC persistence ([#398](https://github.com/VforVitorio/F1-StratLab/issues/398)) ([47cf9a7](https://github.com/VforVitorio/F1-StratLab/commit/47cf9a74ce2fc92095725dbe97bbe1b3e55c9216))
* **nlp:** render DOUBLE YELLOW at yellow severity + route time penalties to N30 ([#398](https://github.com/VforVitorio/F1-StratLab/issues/398) follow-up) ([b45d7f1](https://github.com/VforVitorio/F1-StratLab/commit/b45d7f158c6b524da4f0a2c5ca66fe778c1c19ef))
* **nlp:** widen the RCM parser + route penalty/red-flag alerts + persist SC in arcade ([#398](https://github.com/VforVitorio/F1-StratLab/issues/398)) ([6a223e1](https://github.com/VforVitorio/F1-StratLab/commit/6a223e1ac5809e43c2287959353a6ace860a06ec))


### Performance

* **cli:** skip double model load in --no-llm prewarm ([#389](https://github.com/VforVitorio/F1-StratLab/issues/389)) ([56c24f7](https://github.com/VforVitorio/F1-StratLab/commit/56c24f7aa37695f2e2a0026f9abcbf84dd686048))
* **cli:** skip the tire/situation/pit singleton prewarm in --no-llm mode ([#389](https://github.com/VforVitorio/F1-StratLab/issues/389)) ([f0ba4b4](https://github.com/VforVitorio/F1-StratLab/commit/f0ba4b4b96ecca4146f353252eeb92302bacf626))

## [1.10.2](https://github.com/VforVitorio/F1-StratLab/compare/v1.10.1...v1.10.2) (2026-07-12)


### Bug Fixes

* **data:** resolve every GP's radio + compound labels via a canonical name (F-01) ([2ab70d1](https://github.com/VforVitorio/F1-StratLab/commit/2ab70d12fb9d127c7ec95efdfcc647f2dbd70f06))
* **data:** resolve every GP's radio + compound labels via a canonical name (F-01) ([29d1cac](https://github.com/VforVitorio/F1-StratLab/commit/29d1cac6dc76c228d7f080cebbe5b72d2f103667)), closes [#243](https://github.com/VforVitorio/F1-StratLab/issues/243)
* **data:** validate the parquet read boundary + surface FastF1 quality flags (F-02) ([80b12b6](https://github.com/VforVitorio/F1-StratLab/commit/80b12b6459cdd7358c4896f80114fbd46488a544))
* **data:** validate the parquet read boundary and surface FastF1 quality flags (F-02) ([dd44451](https://github.com/VforVitorio/F1-StratLab/commit/dd4445150b13d3984bccae8ddf907618bea0e808)), closes [#244](https://github.com/VforVitorio/F1-StratLab/issues/244)
* **deps:** drop fitz dummy, declare pypdf, override frozendict (DX-05) ([3a7612c](https://github.com/VforVitorio/F1-StratLab/commit/3a7612ce7d83a8334bb304ee210d3447ddf8cfa7))
* **deps:** drop the unused fitz dummy, declare pypdf, override frozendict (DX-05) ([794a39b](https://github.com/VforVitorio/F1-StratLab/commit/794a39bd72fc8969cf09e79d5b3469a4c4d20f25)), closes [#253](https://github.com/VforVitorio/F1-StratLab/issues/253)
* **devex:** unbreak the Docker/Streamlit quickstart + redirected-output crash ([#388](https://github.com/VforVitorio/F1-StratLab/issues/388)) ([75d1b3e](https://github.com/VforVitorio/F1-StratLab/commit/75d1b3ec0d681abe7b6b78afe6e7061cab22e02e))
* **devex:** unbreak the Docker/Streamlit quickstart and the redirected-output crash ([f7f988a](https://github.com/VforVitorio/F1-StratLab/commit/f7f988a619224bd3adea5cda20cb7b117f689cfa)), closes [#252](https://github.com/VforVitorio/F1-StratLab/issues/252) [#388](https://github.com/VforVitorio/F1-StratLab/issues/388)
* **nlp:** persist Safety-Car state across laps so the override survives the stint (NR-02) ([c713705](https://github.com/VforVitorio/F1-StratLab/commit/c713705d364271c55bbee42085d1e7c6e5742ac1))
* **nlp:** persist Safety-Car state across laps so the override survives the stint (NR-02) ([215007b](https://github.com/VforVitorio/F1-StratLab/commit/215007be7d6198d6fd36386d0f3603a018200fdd)), closes [#305](https://github.com/VforVitorio/F1-StratLab/issues/305)
* **packaging:** bundle the telemetry submodule in the release wheel (PK-01) ([a047b48](https://github.com/VforVitorio/F1-StratLab/commit/a047b485957c3f6fe65bad4b503459a234805d74))
* **packaging:** bundle the telemetry submodule in the release wheel (PK-01) ([40ae581](https://github.com/VforVitorio/F1-StratLab/commit/40ae581df29022476c7def8f795a1db885b5b293)), closes [#289](https://github.com/VforVitorio/F1-StratLab/issues/289)


### Documentation

* **security:** add the Phase A security-boundary design for review ([#224](https://github.com/VforVitorio/F1-StratLab/issues/224)) ([c71eac9](https://github.com/VforVitorio/F1-StratLab/commit/c71eac9e6fb26da59ca3572f7817ff8de3c46a28))
* **security:** Phase A security-boundary design for review ([#224](https://github.com/VforVitorio/F1-StratLab/issues/224)) ([ed4e829](https://github.com/VforVitorio/F1-StratLab/commit/ed4e8295b200cab373769378f9095335c090d113))

## [1.10.1](https://github.com/VforVitorio/F1-StratLab/compare/v1.10.0...v1.10.1) (2026-07-12)


### Bug Fixes

* **cli:** wire the simulation CLI to the shared inference engine ([#236](https://github.com/VforVitorio/F1-StratLab/issues/236)) ([32dcc4b](https://github.com/VforVitorio/F1-StratLab/commit/32dcc4b38a71afd31debf7bccb4359b76a911c2f))
* **cli:** wire the simulation CLI to the shared inference engine ([#236](https://github.com/VforVitorio/F1-StratLab/issues/236)) ([0694299](https://github.com/VforVitorio/F1-StratLab/commit/06942991dfbd31521ff087841fb2e95754132bf0))

## [1.10.0](https://github.com/VforVitorio/F1-StratLab/compare/v1.9.0...v1.10.0) (2026-07-11)


### Features

* **agents:** give every sub-agent ChatOpenAI a finite provider timeout (L-1) ([3eb0eb8](https://github.com/VforVitorio/F1-StratLab/commit/3eb0eb8b48ac88b49e3119d99946017447c34fec))
* **agents:** give every sub-agent ChatOpenAI a finite provider timeout (L-1) ([401765f](https://github.com/VforVitorio/F1-StratLab/commit/401765fffa132b81d02f862c4c12362ef9d4567d)), closes [#263](https://github.com/VforVitorio/F1-StratLab/issues/263)


### Bug Fixes

* **ci:** skip the tiktoken vocab roundtrip on an upstream network outage ([6c26ec2](https://github.com/VforVitorio/F1-StratLab/commit/6c26ec23134029213c365cde787afebfc0069f56))
* **ci:** skip the tiktoken vocab roundtrip on an upstream network outage ([cdd9f4a](https://github.com/VforVitorio/F1-StratLab/commit/cdd9f4ac895f7a26f21bfd306069d64423b57398))

## [1.9.0](https://github.com/VforVitorio/F1-StratLab/compare/v1.8.0...v1.9.0) (2026-07-11)


### Features

* **eval:** reproduce the pace MAE 0.4104 from the featured-laps holdout ([2a2c982](https://github.com/VforVitorio/F1-StratLab/commit/2a2c9827e46e093d35ba426b1e839de395116f0f))
* **eval:** reproduce the pace MAE 0.4104 from the featured-laps holdout ([998a800](https://github.com/VforVitorio/F1-StratLab/commit/998a800afb838208be17a7469763c614a5513561))
* **eval:** reproduce the tire MAE 0.7078 and validate the MC-Dropout sigma ([8ce8653](https://github.com/VforVitorio/F1-StratLab/commit/8ce8653dfb4f8cc6e80fc2be00c45443ac030976))
* **eval:** reproduce the tire MAE 0.7078 and validate the MC-Dropout sigma ([762b030](https://github.com/VforVitorio/F1-StratLab/commit/762b03068d4b709d00447b81621bbc966c2cd720))


### Bug Fixes

* **eval:** correct the circuit_cluster hygiene verdict to a real coarse leak ([4328f95](https://github.com/VforVitorio/F1-StratLab/commit/4328f9550be402a68f7383c1aa9ea8d52abf4335))
* **eval:** correct the circuit_cluster hygiene verdict to a real coarse leak ([679208e](https://github.com/VforVitorio/F1-StratLab/commit/679208e70952540f2238e491235d0e7412571e32))


### Documentation

* **eval:** resolve the circuit_cluster hygiene item as an accepted non-target limitation ([33ff301](https://github.com/VforVitorio/F1-StratLab/commit/33ff301fdf756c6870bee2a7e8eaf4c854afdf3e))
* **eval:** resolve the circuit_cluster hygiene item as an accepted non-target limitation ([c72c9fe](https://github.com/VforVitorio/F1-StratLab/commit/c72c9fe35e4155f21459bceac7f6f39a8e5e46a2))

## [1.8.0](https://github.com/VforVitorio/F1-StratLab/compare/v1.7.0...v1.8.0) (2026-07-11)


### Features

* **eval:** add LLM-judge alert-precision module over unlabeled radios ([b727baa](https://github.com/VforVitorio/F1-StratLab/commit/b727baabba5304fa55eac99df2b03e0a8de06812))
* **eval:** add metrics-registry + calibration harness (f1-eval) ([c08780d](https://github.com/VforVitorio/F1-StratLab/commit/c08780d1373fe265496a86e6d38a997ac7c3e8e7)), closes [#206](https://github.com/VforVitorio/F1-StratLab/issues/206)
* **eval:** add NLP per-stage eval harness (f1-eval nlp) ([dbf76b1](https://github.com/VforVitorio/F1-StratLab/commit/dbf76b18bf303989a5e3fe3da52c7b323b3ae56a)), closes [#304](https://github.com/VforVitorio/F1-StratLab/issues/304)
* **eval:** add threshold-provenance + leakage hygiene report (f1-eval hygiene) ([3788167](https://github.com/VforVitorio/F1-StratLab/commit/37881672650ba14404faf7d56a45ec83e62d224f)), closes [#207](https://github.com/VforVitorio/F1-StratLab/issues/207)
* **eval:** compute radio alert precision from gold intent labels ([16bd305](https://github.com/VforVitorio/F1-StratLab/commit/16bd305ed7bc2ea0d35277dd2bafd10e4ed67cc1)), closes [#304](https://github.com/VforVitorio/F1-StratLab/issues/304)
* **eval:** LLM-judge alert precision over unlabeled radios ([#304](https://github.com/VforVitorio/F1-StratLab/issues/304) follow-up) ([1651c3b](https://github.com/VforVitorio/F1-StratLab/commit/1651c3bbb51406c66206896eae7623e8f3ab6671))
* **eval:** ML metrics-registry + calibration harness (f1-eval, [#206](https://github.com/VforVitorio/F1-StratLab/issues/206)) ([380f1d7](https://github.com/VforVitorio/F1-StratLab/commit/380f1d766e4e99f08af9cf2eb176665ce714f9c7))
* **eval:** NLP per-stage eval harness (f1-eval nlp, [#304](https://github.com/VforVitorio/F1-StratLab/issues/304)) ([1c8ff7c](https://github.com/VforVitorio/F1-StratLab/commit/1c8ff7c850c54c613b54b72e7f0bd55b6b2dfbaa))
* **eval:** radio alert precision from gold labels (closes [#304](https://github.com/VforVitorio/F1-StratLab/issues/304)) ([da1db0c](https://github.com/VforVitorio/F1-StratLab/commit/da1db0c6f51d8419734756f9ddd828c979c71546))
* **eval:** recompute SC + undercut calibration + AUC-PR (part of [#364](https://github.com/VforVitorio/F1-StratLab/issues/364)) ([5c97461](https://github.com/VforVitorio/F1-StratLab/commit/5c974617f201bfb215d6525866fb494ecbb0428a))
* **eval:** recompute SC + undercut calibration and AUC-PR from in-memory holdouts ([7365b50](https://github.com/VforVitorio/F1-StratLab/commit/7365b5074a544f7deb02a2d0f0592b078aa660b9)), closes [#364](https://github.com/VforVitorio/F1-StratLab/issues/364)
* **eval:** regenerate pit holdout from raw laps (closes [#364](https://github.com/VforVitorio/F1-StratLab/issues/364)) ([d66ddae](https://github.com/VforVitorio/F1-StratLab/commit/d66ddae72a9cfd873348bc6c8541ca5b82ced2b8))
* **eval:** regenerate pit holdout from raw laps + recompute coverage and MAE ([ccc213d](https://github.com/VforVitorio/F1-StratLab/commit/ccc213d069e8d1f47056d8207652f211b0fe29fb)), closes [#364](https://github.com/VforVitorio/F1-StratLab/issues/364)
* **eval:** reproduce intent setfit-free + [#303](https://github.com/VforVitorio/F1-StratLab/issues/303) NLP hygiene verdicts ([6bc5413](https://github.com/VforVitorio/F1-StratLab/commit/6bc54135a8c9d08ebaa5f048013e1b8e3a916c60))
* **eval:** reproduce intent setfit-free and record [#303](https://github.com/VforVitorio/F1-StratLab/issues/303) NLP hygiene verdicts ([275c487](https://github.com/VforVitorio/F1-StratLab/commit/275c487a456c45c61b1371504f668df8f5dd0d35))
* **eval:** reproduce NER entity-F1 + RCM coverage ([#304](https://github.com/VforVitorio/F1-StratLab/issues/304)) ([bc0350d](https://github.com/VforVitorio/F1-StratLab/commit/bc0350d62d13d0f4b7673a389af949e07dfca58b))
* **eval:** reproduce NER entity-F1 and RCM coverage in the NLP harness ([1efb133](https://github.com/VforVitorio/F1-StratLab/commit/1efb1331d8ddef3efef70dcdd23631a3ef71268d)), closes [#304](https://github.com/VforVitorio/F1-StratLab/issues/304)
* **eval:** threshold-provenance + leakage hygiene report ([#207](https://github.com/VforVitorio/F1-StratLab/issues/207)) ([5f1f29d](https://github.com/VforVitorio/F1-StratLab/commit/5f1f29d8ab7bdd6499f098c8e1fcef3d675838fd))
* **eval:** wire f1-eval alert-llm + report the unlabeled-corpus finding ([1e9fed1](https://github.com/VforVitorio/F1-StratLab/commit/1e9fed178ba2e25582f673a1083d1aebd65e978f)), closes [#304](https://github.com/VforVitorio/F1-StratLab/issues/304)


### Bug Fixes

* **eval:** align undercut provenance wording with in-train relabeling (Fable re-verify) ([54e5d42](https://github.com/VforVitorio/F1-StratLab/commit/54e5d42ff757467a4e478f29e9b97b70dcf722ca))
* **eval:** correct SC operating threshold + window on honest splits ([#363](https://github.com/VforVitorio/F1-StratLab/issues/363)) ([1275101](https://github.com/VforVitorio/F1-StratLab/commit/12751013815c02b4242a50e6465321c77c6d7d37))
* **eval:** correct SC threshold + window on honest splits (closes [#363](https://github.com/VforVitorio/F1-StratLab/issues/363)) ([e1ee603](https://github.com/VforVitorio/F1-StratLab/commit/e1ee60386b5b91feddde03cc35ea280f20fc6803))
* **eval:** relabel SC/overtake threshold corrections as in-train (Fable gate) ([c38c94e](https://github.com/VforVitorio/F1-StratLab/commit/c38c94e9fe9641711d99ecb31211f91e33379e9c))


### Documentation

* **eval:** regenerate hygiene report with in-train relabeling + correct provenance stamp ([c1af9c1](https://github.com/VforVitorio/F1-StratLab/commit/c1af9c1f84576137cef35a08d7be5dede9f465f6))
* **roadmap:** reconcile published metrics to thesis/IEEE finals ([2758ec5](https://github.com/VforVitorio/F1-StratLab/commit/2758ec524c37cc79bf3f7a9bc315d5cf058a8bca))
* **roadmap:** reconcile published metrics to thesis/IEEE finals ([#213](https://github.com/VforVitorio/F1-StratLab/issues/213)) ([50b7ecf](https://github.com/VforVitorio/F1-StratLab/commit/50b7ecfc01681dec372b1bcdeb20f8d0c90bd30d))

## [1.7.0](https://github.com/VforVitorio/F1-StratLab/compare/v1.6.8...v1.7.0) (2026-07-11)


### Features

* **strategy:** add no-llm engine profile (fixes the --no-llm crash) ([f32e38b](https://github.com/VforVitorio/F1-StratLab/commit/f32e38b32acbca0a21defcc569a9aba32433d083))
* **strategy:** add shared inference engine (rich profile) + arcade delegate ([43c2717](https://github.com/VforVitorio/F1-StratLab/commit/43c271741cf400496280a84cbec1efce6685b401))
* **strategy:** no-llm engine profile (fixes the --no-llm crash, [#166](https://github.com/VforVitorio/F1-StratLab/issues/166)) ([91b98d3](https://github.com/VforVitorio/F1-StratLab/commit/91b98d3274c0027d3c846ebffcac79a09923f1a7))
* **strategy:** shared inference engine (rich profile) + arcade delegate ([0e1b793](https://github.com/VforVitorio/F1-StratLab/commit/0e1b793e1a60a637870aa972bad446937a2c7a03))

## [1.6.8](https://github.com/VforVitorio/F1-StratLab/compare/v1.6.7...v1.6.8) (2026-07-09)


### Documentation

* **audits:** add program completeness review + reconcile roadmap and Rival design ([d4bdb4e](https://github.com/VforVitorio/F1-StratLab/commit/d4bdb4e29c8c636b82ace36615a4220a79348dec))
* **audits:** add RAG layer audit plan ([b747de6](https://github.com/VforVitorio/F1-StratLab/commit/b747de6f8a89be0a9e382b9f47e7c59d517485ea))
* **audits:** add RAG layer audit plan ([3603e73](https://github.com/VforVitorio/F1-StratLab/commit/3603e73f4391937fa7b0b59e55f193e0ed360def))
* **audits:** add voice stack audit plan ([e6594af](https://github.com/VforVitorio/F1-StratLab/commit/e6594aff61e1bc79f85637ba9f75cf3e9f364d8b))
* **audits:** add voice stack audit plan ([e61356e](https://github.com/VforVitorio/F1-StratLab/commit/e61356e927f0b41f50a2ad7f41c67d54a09827a6))
* **audits:** program completeness review + roadmap/Rival reconciliation ([d41e957](https://github.com/VforVitorio/F1-StratLab/commit/d41e957bb083d8800f212c12bd7f38bdeb7f72a1))
* **audits:** promote program completeness review + reconciliation to main ([c7409c1](https://github.com/VforVitorio/F1-StratLab/commit/c7409c1a4d3daaa6cfcd84045eb22c65caea682b))
* **audits:** promote RAG layer audit to main ([1366f8c](https://github.com/VforVitorio/F1-StratLab/commit/1366f8c43ee4064390d0c3e2563dccedd247b92b))
* **audits:** promote voice stack audit to main ([51527f3](https://github.com/VforVitorio/F1-StratLab/commit/51527f37bb296bf058714e2f27e64ae8dc7f5021))
* fix broken copy-paste commands (promote to main) ([2b07e22](https://github.com/VforVitorio/F1-StratLab/commit/2b07e229d9b7cb4f2cd6ed567949d3b0e966a932))
* fix broken copy-paste commands in README/INSTALL/CONTRIBUTING ([5a7783b](https://github.com/VforVitorio/F1-StratLab/commit/5a7783be3e5072e013cc3ad65ccb82b66f88d4ac))
* fix broken copy-paste commands in README/INSTALL/CONTRIBUTING ([be4f058](https://github.com/VforVitorio/F1-StratLab/commit/be4f05847ff0af136cc9cd291e03bb87e1e12c16)), closes [#212](https://github.com/VforVitorio/F1-StratLab/issues/212)
* **research:** add agent-orchestration-flow design ([e18c4d6](https://github.com/VforVitorio/F1-StratLab/commit/e18c4d60b0814391c00b5f9098dfdcac2d8141bb))
* **research:** add agent-orchestration-flow design ([2a9d691](https://github.com/VforVitorio/F1-StratLab/commit/2a9d691c491319e65d1a617ff222935b26c37c07))
* **research:** add ecosystem data-contracts spec ([4bc37ba](https://github.com/VforVitorio/F1-StratLab/commit/4bc37ba50e0d01eb7c817fed61b4755db9c7233d))
* **research:** add ecosystem data-contracts spec ([7b2c38e](https://github.com/VforVitorio/F1-StratLab/commit/7b2c38e0b8236741cdaf611ab7b2d030328896aa))
* **research:** promote agent-orchestration-flow design to main ([f5b61f4](https://github.com/VforVitorio/F1-StratLab/commit/f5b61f420224da15a752e56acdabad423f8da7eb))
* **research:** promote ecosystem data-contracts spec to main ([f8e7303](https://github.com/VforVitorio/F1-StratLab/commit/f8e7303082dcceed77c169a324dad4a01d308805))

## [1.6.7](https://github.com/VforVitorio/F1-StratLab/compare/v1.6.6...v1.6.7) (2026-07-07)


### Documentation

* **audits:** add cross-audit implementation roadmap ([3e42914](https://github.com/VforVitorio/F1-StratLab/commit/3e42914a3b5687a16a52b8c31cbaa3aa892eafb8))
* **audits:** add cross-audit implementation roadmap ([2852af5](https://github.com/VforVitorio/F1-StratLab/commit/2852af5d4728259137722c2f283fb35cd1b71a79))
* **audits:** promote cross-audit implementation roadmap to main ([cec6297](https://github.com/VforVitorio/F1-StratLab/commit/cec6297e03323e32bb73989cab2ddea6585d9040))
* **research:** add box-bot multi-platform design ([7739201](https://github.com/VforVitorio/F1-StratLab/commit/77392014faa16c155f00f4cdc03a54c45287e1d2))
* **research:** add box-bot multi-platform design ([6677167](https://github.com/VforVitorio/F1-StratLab/commit/6677167c3bd3ce8630e0f5b731e0df661ec0743e))
* **research:** promote box-bot design to main ([793afd7](https://github.com/VforVitorio/F1-StratLab/commit/793afd75eab7c691d672571e02521e018d598258))

## [1.6.6](https://github.com/VforVitorio/F1-StratLab/compare/v1.6.5...v1.6.6) (2026-07-07)


### Documentation

* **audits:** add DevEx contributor-setup audit plan ([3c5c0b3](https://github.com/VforVitorio/F1-StratLab/commit/3c5c0b3994406bdd6ee13421b7d1b40faca53d99))
* **audits:** add DevEx contributor-setup audit plan ([b832509](https://github.com/VforVitorio/F1-StratLab/commit/b832509dbeaced3dae1c06f13396e3773f7bfbaf))
* **audits:** add LLM cost & latency audit plan ([7417c75](https://github.com/VforVitorio/F1-StratLab/commit/7417c757b3764011b7e04793faae09375e44a5d4))
* **audits:** add LLM cost & latency audit plan ([2266a41](https://github.com/VforVitorio/F1-StratLab/commit/2266a414b007a0c55a42aebce48c97cfb14a7449))
* **audits:** add NLP / team-radio pipeline audit plan ([2990a50](https://github.com/VforVitorio/F1-StratLab/commit/2990a501f88f919b0fcfcecf0a4ded7dc2889d45))
* **audits:** add NLP / team-radio pipeline audit plan ([f26661d](https://github.com/VforVitorio/F1-StratLab/commit/f26661da6ba605db8d94ce0f3d71bed2e61c654b))
* **audits:** add packaging, release & CI/CD audit plan ([a04f65e](https://github.com/VforVitorio/F1-StratLab/commit/a04f65ee2c74daef6db039bef9f1c1c492f97292))
* **audits:** add packaging, release & CI/CD audit plan ([59d9e0b](https://github.com/VforVitorio/F1-StratLab/commit/59d9e0bf90a01bdbe61fc51e257588f0090967e5))
* **audits:** promote DevEx audit plan to main ([e85f4db](https://github.com/VforVitorio/F1-StratLab/commit/e85f4db3020dee12dd447767a7243609c38fc262))
* **audits:** promote LLM cost & latency audit plan to main ([7bc8481](https://github.com/VforVitorio/F1-StratLab/commit/7bc8481df48c199717e5dc78a49c2d59d5cabc10))
* **audits:** promote NLP / team-radio pipeline audit to main ([7e256fe](https://github.com/VforVitorio/F1-StratLab/commit/7e256feec7d3d2de08c09506fa70c1f5443bee2a))
* **audits:** promote packaging & CI/CD audit plan to main ([a6ff04f](https://github.com/VforVitorio/F1-StratLab/commit/a6ff04f721c5dafc269857ff5c739daa85c2319c))
* **research:** add ecosystem repo-integration architecture note ([d40b6ea](https://github.com/VforVitorio/F1-StratLab/commit/d40b6eae1bafe9168eb6994ddcc3f918af9c1ad3))
* **research:** add ecosystem repo-integration architecture note ([b51bf97](https://github.com/VforVitorio/F1-StratLab/commit/b51bf971d74d1f836f07eaac32c0d1ddd36780e5))
* **research:** add gridmind LoRA design ([7c2cb54](https://github.com/VforVitorio/F1-StratLab/commit/7c2cb5438606d863215a887b211e12ff9f344864))
* **research:** add gridmind LoRA design ([5068343](https://github.com/VforVitorio/F1-StratLab/commit/50683433dfdb6935ef508a1a01c918bf80e5a043))
* **research:** add pit-wall realism + telemetry-surface design ([db7fb44](https://github.com/VforVitorio/F1-StratLab/commit/db7fb44cb25d51b6dcdc2d9abba4e3947ac57452))
* **research:** add pit-wall realism + telemetry-surface design ([ad8dfc2](https://github.com/VforVitorio/F1-StratLab/commit/ad8dfc2337d2755a3874e26b1145a4744fae0cd8))
* **research:** add pitlab Studio design ([3302648](https://github.com/VforVitorio/F1-StratLab/commit/33026484ac8fbe53cc9dc073129e3158bb8399bd))
* **research:** add pitlab Studio design ([2ae226b](https://github.com/VforVitorio/F1-StratLab/commit/2ae226b6be7eff29cf3fe314c3e48c8abdd2c75d))
* **research:** add radiogate deception + auto-labeling design ([6a53ffe](https://github.com/VforVitorio/F1-StratLab/commit/6a53ffed92e81e9d1640a4d1fa6f08b0309fe373))
* **research:** add radiogate deception + auto-labeling design ([1e11430](https://github.com/VforVitorio/F1-StratLab/commit/1e11430f2421bc9ca3aca9c1b761d5ac679948f5))
* **research:** add real-time OpenF1 consumer design ([5b659ff](https://github.com/VforVitorio/F1-StratLab/commit/5b659ff8a83ab8e582cb5782497d1efa50d85daa))
* **research:** add real-time OpenF1 consumer design ([faf2937](https://github.com/VforVitorio/F1-StratLab/commit/faf2937dfc562c0687538faab2894cddeaa872dd))
* **research:** add Rival Agent TFM design ([b458d08](https://github.com/VforVitorio/F1-StratLab/commit/b458d0890f5561b66c9f4bce86144cf6a0a26780))
* **research:** add Rival Agent TFM design ([4bcd4ea](https://github.com/VforVitorio/F1-StratLab/commit/4bcd4ea32ef0f2ab1f61581fa96b76d93f65bf27))
* **research:** promote ecosystem repo-integration note to main ([3af311b](https://github.com/VforVitorio/F1-StratLab/commit/3af311b997289378167cc41d27a9f2f28d5bd2be))
* **research:** promote gridmind LoRA design to main ([373fab9](https://github.com/VforVitorio/F1-StratLab/commit/373fab9fe5faf29af68efca94602eaba9411dcd9))
* **research:** promote pit-wall realism + telemetry-surface design to main ([8e12e15](https://github.com/VforVitorio/F1-StratLab/commit/8e12e15dd81518d9f0085ed362a002d6c7b43c38))
* **research:** promote pitlab Studio design to main ([72b3e5e](https://github.com/VforVitorio/F1-StratLab/commit/72b3e5eb8767cff826fa98b4fc64bf8de7c5ed3b))
* **research:** promote radiogate design to main ([4113ff0](https://github.com/VforVitorio/F1-StratLab/commit/4113ff0d381fe29a3ebf42faaa0da0233032eb1b))
* **research:** promote real-time OpenF1 consumer design to main ([b220ed1](https://github.com/VforVitorio/F1-StratLab/commit/b220ed1eeb0692bb854fcc73d0bb996f4876f8ae))
* **research:** promote Rival Agent TFM design to main ([7a314f5](https://github.com/VforVitorio/F1-StratLab/commit/7a314f5e13789b64c738b70117b2f611b8e0cf8a))

## [1.6.5](https://github.com/VforVitorio/F1-StratLab/compare/v1.6.4...v1.6.5) (2026-07-05)


### Documentation

* **audits:** add P4 CLI surface audit plan ([fef83ab](https://github.com/VforVitorio/F1-StratLab/commit/fef83abb1762a2a48edf0767bc440d19431976e0))
* **audits:** add P4 CLI surface audit plan ([7a4da77](https://github.com/VforVitorio/F1-StratLab/commit/7a4da7735f6be4f5f2b38a3434fa7d7223175e73))
* **audits:** add P5 data-engineering audit plan ([5679490](https://github.com/VforVitorio/F1-StratLab/commit/5679490e6332d036cda3555766b3980c4c3816b4))
* **audits:** add P5 data-engineering audit plan ([dcda30c](https://github.com/VforVitorio/F1-StratLab/commit/dcda30cc6384b408716e0ca949a7ac0bac96aebd))
* **audits:** add security & prompt-injection audit plan ([0be2ce9](https://github.com/VforVitorio/F1-StratLab/commit/0be2ce9cfb86bb994882caa34a11b7ad31cabd31))
* **audits:** add security & prompt-injection audit plan ([4fe8094](https://github.com/VforVitorio/F1-StratLab/commit/4fe8094e8b33e234bbfc7310d1f4597a66b0ff39))
* **audits:** promote P4 CLI audit plan to main ([9882b45](https://github.com/VforVitorio/F1-StratLab/commit/9882b45be37cb800f0639e82780312e3126a13e2))
* **audits:** promote P5 data-engineering audit plan to main ([b774014](https://github.com/VforVitorio/F1-StratLab/commit/b774014ac04d8111ecb8c4fd33f8850e0dce79f2))
* **audits:** promote security audit plan to main ([af3b2ef](https://github.com/VforVitorio/F1-StratLab/commit/af3b2eff1adf2db5fb0881fdc48862b5a77bdde0))
* **readme:** lead with product value and fix agent count ([05212ef](https://github.com/VforVitorio/F1-StratLab/commit/05212ef533176b8c860ac7222ceb9bf87a269cd7))

## [1.6.4](https://github.com/VforVitorio/F1-StratLab/compare/v1.6.3...v1.6.4) (2026-07-05)


### Documentation

* **audits:** add arcade, ML-eval and docs-accuracy audit plans ([dd390bd](https://github.com/VforVitorio/F1-StratLab/commit/dd390bd67d9ec68abd1d1e6540e146614a6b2a73))
* **audits:** add arcade, ML-eval and docs-accuracy audit plans ([caf4347](https://github.com/VforVitorio/F1-StratLab/commit/caf4347365528f1d879e00d4be4eea77e2f95f2e))
* **audits:** promote arcade, ML-eval and docs-accuracy audit plans ([574acca](https://github.com/VforVitorio/F1-StratLab/commit/574accaf25fc48c2032ca56a5ea2517d08f888f0))

## [1.6.3](https://github.com/VforVitorio/F1-StratLab/compare/v1.6.2...v1.6.3) (2026-07-05)


### Documentation

* add 2026-regulation concept-drift readiness audit ([36148d6](https://github.com/VforVitorio/F1-StratLab/commit/36148d645504a9a492e54c8b23c731b72d3d302a))
* add 2026-regulation concept-drift readiness audit ([c708b8e](https://github.com/VforVitorio/F1-StratLab/commit/c708b8ec58e270c35c35fbb2d36ab751ca77be71))

## [1.6.2](https://github.com/VforVitorio/F1-StratLab/compare/v1.6.1...v1.6.2) (2026-07-04)


### Documentation

* **audits:** add backend, loading and core-compute audit plans ([0b628f2](https://github.com/VforVitorio/F1-StratLab/commit/0b628f2904de4aa69fc010f3bcbfd44e2a140be0))
* **audits:** add testing and QA strategy audit plan ([ff4d30e](https://github.com/VforVitorio/F1-StratLab/commit/ff4d30ebea85bf7bb05862a291f0c11cf2cce91d))
* **audits:** P1, P2 and P2b audit plans ([dc90de9](https://github.com/VforVitorio/F1-StratLab/commit/dc90de966b9b3f88142dd30818671f54b63a8f8f))
* **audits:** testing and QA strategy audit + tests/fixtures carve-out ([bfb9d0f](https://github.com/VforVitorio/F1-StratLab/commit/bfb9d0fb42bbb10a8c9a74071ccc44f0eca89a9e))

## [1.6.1](https://github.com/VforVitorio/F1-StratLab/compare/v1.6.0...v1.6.1) (2026-06-29)


### Bug Fixes

* **docs:** accessibility (WCAG AA) pass on the docs site ([1577d9a](https://github.com/VforVitorio/F1-StratLab/commit/1577d9a3f9ea6b6320a760e738a0626dc72e83cd))
* **docs:** correct the docs-maintenance framework label ([3645987](https://github.com/VforVitorio/F1-StratLab/commit/36459876c26013d30be1e530f0b17855cbd3a2cf))
* **docs:** hide prerendered placeholder until React mounts ([9dec895](https://github.com/VforVitorio/F1-StratLab/commit/9dec8956580b8b5b19cabff401575617e2829fb3))
* **docs:** hide prerendered placeholder until React mounts (no FOUC) ([cda4da0](https://github.com/VforVitorio/F1-StratLab/commit/cda4da0e5df055a800f829068516a6ebd313a622))
* **docs:** Sprint 2 - accessibility (WCAG AA) ([b5c53da](https://github.com/VforVitorio/F1-StratLab/commit/b5c53da795d796e4aff3ba55b74ad397bf18aaa1))
* **docs:** Sprint 3 - correct docs-maintenance framework label ([15c55ff](https://github.com/VforVitorio/F1-StratLab/commit/15c55ffb255b9f94b8e193a08150601bcbacc5e3))


### Performance

* **docs:** drop @babel/standalone, load app as plain scripts ([75b4a10](https://github.com/VforVitorio/F1-StratLab/commit/75b4a108cf64b56a21ec7067cb818b15ada917b5))
* **docs:** drop @babel/standalone, load app as plain scripts ([e448bae](https://github.com/VforVitorio/F1-StratLab/commit/e448bae8be0f6ada4ffce8c774f892eae86239c8)), closes [#136](https://github.com/VforVitorio/F1-StratLab/issues/136)
* **docs:** optimize demo media and page load ([6590db2](https://github.com/VforVitorio/F1-StratLab/commit/6590db246559c52678db44dc495eb724fca7e528))
* **docs:** Sprint 1 - performance & load ([5687206](https://github.com/VforVitorio/F1-StratLab/commit/568720685552b17470161a4bb55b6a4f15ff06fd))


### Documentation

* describe the real React stack, drop MkDocs references ([c944640](https://github.com/VforVitorio/F1-StratLab/commit/c944640d56204b993a238d8db9abfdba4397ab80))
* describe the real React stack, drop MkDocs references ([9b781d4](https://github.com/VforVitorio/F1-StratLab/commit/9b781d48b64c5e5bbbf0a6e752ce85016180407a)), closes [#156](https://github.com/VforVitorio/F1-StratLab/issues/156)

## [1.6.0](https://github.com/VforVitorio/F1-StratLab/compare/v1.5.5...v1.6.0) (2026-06-28)


### Features

* **docs:** add GEO quick wins (llms.txt, JSON-LD, AI robots, prod React) ([133c66f](https://github.com/VforVitorio/F1-StratLab/commit/133c66fb51d9989a82a6a602f5b11ff1b46ecc48)), closes [#117](https://github.com/VforVitorio/F1-StratLab/issues/117)
* **docs:** GEO quick wins — llms.txt, JSON-LD, AI robots, prod React ([ae5a9fd](https://github.com/VforVitorio/F1-StratLab/commit/ae5a9fd7e5ff36132bd223ac9ec0637b20c6d8d6))
* **docs:** prerender pages to crawlable HTML + real-URL sitemap ([e6d2bfc](https://github.com/VforVitorio/F1-StratLab/commit/e6d2bfcc6293b89fe5740d87e876cb58ea8218bf))
* **docs:** prerender pages to crawlable HTML with real URLs ([b06ea02](https://github.com/VforVitorio/F1-StratLab/commit/b06ea02ecd6cca8484caa4bd36a3378e75a49cd7)), closes [#118](https://github.com/VforVitorio/F1-StratLab/issues/118)


### Bug Fixes

* **ci:** use GITHUB_TOKEN for release-please instead of expired PAT ([e6e6cea](https://github.com/VforVitorio/F1-StratLab/commit/e6e6cea7b42c489c7ef4ab6ba5a10298e77faa92))
* **ci:** use GITHUB_TOKEN for release-please instead of expired PAT ([605eb97](https://github.com/VforVitorio/F1-StratLab/commit/605eb9716d4c30dfca7b6cbdd555a33a99f20984))


### Documentation

* add arcade demo, TFG distinction note, and next-core-releases roadmap ([78e8386](https://github.com/VforVitorio/F1-StratLab/commit/78e8386b53a7be8ed9c30a59487b210734370e8d))
* add full project timeline roadmap page ([cc39dd3](https://github.com/VforVitorio/F1-StratLab/commit/cc39dd3e1b5a01198b632f9cd1a31a1b1979d671))
* **content:** add definitional leads, FAQ and routing-rule prose for AI citability ([d01b7c9](https://github.com/VforVitorio/F1-StratLab/commit/d01b7c96ef46f18a3895948d033c4ab6a072803f)), closes [#120](https://github.com/VforVitorio/F1-StratLab/issues/120)
* **content:** definitional leads + FAQ + routing-rule prose (citability) ([cf92cd7](https://github.com/VforVitorio/F1-StratLab/commit/cf92cd7d06c9fb68af1d5c4465357fc37109934e))
* embed CLI, Arcade and Streamlit demo gifs and add the demo videos to docs/assets/demo ([7678b57](https://github.com/VforVitorio/F1-StratLab/commit/7678b577220d1e451c8fdd48abf4cf101383d1e8))
* link the landing demo carousel from the README hero ([f2d14a9](https://github.com/VforVitorio/F1-StratLab/commit/f2d14a934190ebbd23a852f362a9e2c0044178fe))
* remove em-dashes from README and ROADMAP prose ([5086e58](https://github.com/VforVitorio/F1-StratLab/commit/5086e58d19ba8f1ab8afcc891b3e4d959723083e))
* update README, CONTRIBUTING, INDEX and ROADMAP; add TFG thesis and IEEE report PDFs ([5a9371c](https://github.com/VforVitorio/F1-StratLab/commit/5a9371cc27f2c766a2f6c54ba6b013bd5e17abce))

## [1.5.5](https://github.com/VforVitorio/F1-StratLab/compare/v1.5.4...v1.5.5) (2026-05-23)


### Documentation

* **telemetry:** bump submodule to refreshed README (601fd23) ([195d4aa](https://github.com/VforVitorio/F1-StratLab/commit/195d4aa541de8317bdb98c2a13dbab3b16ae6096))

## [1.5.4](https://github.com/VforVitorio/F1-StratLab/compare/v1.5.3...v1.5.4) (2026-05-22)


### Documentation

* añade memoria del TFG y paper bajo documents/thesis/ ([16497cc](https://github.com/VforVitorio/F1-StratLab/commit/16497cc1c4d6379eeb755be28fd4389f10454412))

## [1.5.3](https://github.com/VforVitorio/F1-StratLab/compare/v1.5.2...v1.5.3) (2026-05-21)


### Bug Fixes

* **ci:** replace autoupdate action with direct gh api call ([810b9b0](https://github.com/VforVitorio/F1-StratLab/commit/810b9b0f250daab415b9f75d22f35cd1f73d74f3))
* **ci:** replace autoupdate action with direct gh api call ([5a89b6e](https://github.com/VforVitorio/F1-StratLab/commit/5a89b6efc2d5fd09a210e7a098a6960f887524c5))
* **ci:** replace autoupdate action with direct gh api call ([a175266](https://github.com/VforVitorio/F1-StratLab/commit/a1752661d82730f7e3befea8097a81ac68205b47))

## [1.5.2](https://github.com/VforVitorio/F1-StratLab/compare/v1.5.1...v1.5.2) (2026-05-20)


### Bug Fixes

* **tests:** suppress catboost_info dir creation ([5a5b87d](https://github.com/VforVitorio/F1-StratLab/commit/5a5b87db1392df8e329ea2f37097d334f56ce095))
* **tests:** suppress catboost_info dir creation on dep-imports test ([d49fa94](https://github.com/VforVitorio/F1-StratLab/commit/d49fa9446940fb3f988b2cc019371011b4ac5679))

## [1.5.1](https://github.com/VforVitorio/F1-StratLab/compare/v1.5.0...v1.5.1) (2026-05-17)


### Bug Fixes

* **ci:** key uv cache off pyproject.toml since uv.lock is gitignored ([7fafc9b](https://github.com/VforVitorio/F1-StratLab/commit/7fafc9b04345cb43df062195a9ac12358cb22153))
* **ci:** set cache-dependency-glob on lint job too ([df90899](https://github.com/VforVitorio/F1-StratLab/commit/df908992d9ec2c515016a00fef73b6fd8daf4594))

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
