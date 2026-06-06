MAIN ?= final
LATEXMK ?= latexmk

.PHONY: all pdf clean distclean

all: pdf

pdf: $(MAIN).pdf

$(MAIN).pdf: $(MAIN).tex references.bib neurips_2019.sty
	$(LATEXMK) -pdf -interaction=nonstopmode -halt-on-error $(MAIN).tex

clean:
	$(LATEXMK) -C $(MAIN).tex

distclean: clean
	rm -f $(MAIN).pdf
