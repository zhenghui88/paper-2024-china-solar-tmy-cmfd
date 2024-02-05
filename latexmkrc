$pdf_mode = 1;
$postscript_mode = 0;
$dvi_mode = 0;
$pdflatex = 'pdflatex -synctex=1 -interaction=nonstopmode';
$lualatex = "lualatex -synctex=1 -interaction=nonstopmode %O %S";
@generated_exts = (@generated_exts, 'synctex.gz');
@default_files = ('main.tex');
