# Build the technical report

The paper uses the official ACL LaTeX class in final, non-anonymous mode. The
style files are vendored unmodified at the exact upstream commit recorded in
`ACL_STYLE_SOURCE.md`. TikZ and PGFPlots generate the figures, so there is no
external image dependency.

From the public repository root:

```bash
tectonic report/Huntrix_Technical_Report.tex --outdir .
pdfinfo ../Huntrix_report.pdf | grep -E 'Pages|Page size'
pdffonts ../Huntrix_report.pdf
```

The expected result is one five-page A4 PDF: pages 1--4 contain the paper body,
and page 5 begins the references. Every listed font should be embedded. Tectonic
may download LaTeX packages on the first local build; report compilation is
separate from the offline Kaggle inference notebook.
