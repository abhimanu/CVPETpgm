
midterm: midterm.tex reference.bib intro.tex related_work.tex method.tex experiments.tex conclusion.tex
	pdflatex midterm.tex
	bibtex midterm
	pdflatex midterm.tex
	pdflatex midterm.tex
	rm -f *.log *.aux *.bbl *.blg *.out

proposal: proposal.tex reference.bib	
	latex proposal.tex
	bibtex proposal
	latex proposal.tex
	pdflatex proposal.tex
	rm *.log *.aux *.bbl *.blg *.dvi *.out

clean:
	rm *.pdf
