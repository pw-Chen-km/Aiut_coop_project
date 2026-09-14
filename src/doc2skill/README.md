# Doc2Skill package

For the current adaptive v2 implementation, installation and deployment, see the
[project README](../../README.md) and [algorithm specification](../../docs/navigation-v2.md).

`adaptive_pipeline.build_adaptive` consumes complete parsed corpus registries,
corrects structure, chunks within chapters, generates four-field source-grounded
metadata, compiles tokenizer-budgeted navigation MD and writes SQLite/NumPy indexes.
No questions or reference answers are read by this offline construction path.

The package also retains legacy `build`, `prepare-structure`, `refine`, `query`
and `evaluate` interfaces for existing bundles. Legacy `query navigation-expand`
is retrieval-only and is not the v2 QA agent's multi-round answering loop.
Do not mix its results or defaults with v2 without an explicit comparison protocol.

PDF input requires a source manifest of exact file identities. A structure review
is tied to the parsed corpus fingerprint and must assign all original nodes.
`refine` accepts reviewed metadata and publishes a new version without editing
the original bundle. An incomplete/no-metadata build is not a working skill.

Source units, sections and retrievable chunks are separate concepts. Rechunking
can preserve source coordinates within one parse; rerunning PDF extraction or OCR
does not guarantee identical coordinates. Figures remain source references, not
evidence of visual understanding. All generated corpus artifacts remain private.
