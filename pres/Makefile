all:
	lualatex pres
#	bibtex pres
	lualatex pres
	lualatex pres
clean:
	-@rm -rf *.log
	-@rm -rf *.aux
	-@rm -rf *.t1
	-@rm -rf *.mp
	-@rm -rf pres.bbl
	-@rm -rf pres.toc
	-@rm -rf pres.out
	-@rm -rf pres.blg
	-@rm -rf pres.nav
	-@rm -rf pres.snm

open:
	gnome-open pres.pdf </dev/null &>/dev/null &
# TODO get the correct viewer for the laptop installed, check
