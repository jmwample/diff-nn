

all:
	pdflatex differential_nets_prop.tex && bibtex differential_nets_prop.aux && pdflatex differential_nets_prop.tex

paper:
	pdflatex differential_nets_prop.tex && bibtex differential_nets_prop.aux && pdflatex differential_nets_prop.tex

clean:
	$(RM) differential_nets_prop.pdf differential_nets_prop.aux differential_nets_prop.log differential_nets_prop.blg differential_nets_prop.bbl

