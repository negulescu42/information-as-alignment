PAPER = main
ENGINE = pdflatex
LATEXMK_FLAGS = -$(ENGINE) -interaction=nonstopmode -halt-on-error -file-line-error

.PHONY: all clean watch

all: $(PAPER).pdf

$(PAPER).pdf: $(PAPER).tex
	latexmk $(LATEXMK_FLAGS) $(PAPER).tex

watch:
	latexmk $(LATEXMK_FLAGS) -pvc $(PAPER).tex

clean:
	latexmk -C
	rm -f *.aux *.log *.out *.toc *.bbl *.blg *.fls *.fdb_latexmk *.synctex.gz
