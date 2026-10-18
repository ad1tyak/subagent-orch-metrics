# Project proposal (LaTeX)

Build the PDF:

    latexmk -pdf main.tex

Or without latexmk: `pdflatex main && bibtex main && pdflatex main && pdflatex main`.
Overleaf: upload this folder as a zip; it compiles with the default pdfLaTeX settings.

Files
- main.tex: the proposal
- references.bib: BibTeX references (cited in order of appearance, unsrtnat style)
- figures/fig_system.tex, figures/fig_timeline.tex: TikZ sources for Figures 1 and 2
- figures/colors.tex: shared figure colors
- figures/standalone_*.tex: build each figure on its own (run from figures/)
- figures/fig_*.pdf, figures/fig_*.png: prebuilt figures for slides

Before submitting
- references.bib: fill in the authors of the Librarian paper (arXiv:2605.27787), marked TODO.
