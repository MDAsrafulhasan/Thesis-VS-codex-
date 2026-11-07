# Paper figures reused in the presentation

These standalone files reproduce the exact TikZ/PGFPlots visuals used in
`Huntrix_Technical_Report.tex`.

| Report visual | Presentation slide | Standalone source | Exported asset |
|---|---:|---|---|
| Frozen v91 architecture | 3 | `architecture_standalone.tex` | `architecture.{pdf,svg,png}` |
| Claim-level context verification | 7 | `context_verification_standalone.tex` | `context_verification.{pdf,svg,png}` |
| Typed evidence admission | 9 | `typed_evidence_standalone.tex` | `typed_evidence.{pdf,svg,png}` |
| Experiment progression | 10 | `progress_standalone.tex` | `progress.{pdf,svg,png}` |

Compile a source with Tectonic, then export SVG and PNG with Poppler:

```bash
tectonic architecture_standalone.tex
pdftocairo -svg architecture_standalone.pdf architecture.svg
pdftocairo -png -singlefile -r 400 -transp architecture_standalone.pdf architecture
```

The PPTX embeds the high-resolution transparent PNG exports for broad
PowerPoint compatibility. The PDF and SVG masters remain available for
publication-quality reuse.
