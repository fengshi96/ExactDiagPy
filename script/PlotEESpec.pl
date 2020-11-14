#!/usr/bin/perl

use strict;
use warnings;
use utf8;

# get Entanglement Entropy
my @entropylist;
opendir (DIR, ".") or die $!;
while (my $dir = readdir(DIR)) {
	if ($dir =~ m/Bxx/) {
		$dir =~ /Bxx_n(.+)/;
		my $field = $1;
		open(FILEIN, "<", "$dir"."/logfile.log") or die $!;
		while(<FILEIN>) {
			if (/Entanglement Entropy= (.+)/) {
				$_ =~ m/.+= (.+)/;
				my $EE = $1;
				push @entropylist, "$field $EE";
			}
		}

		close(FILEIN);
	}
}
closedir(DIR);

open(FILEOUT, ">", "0.ESCollector/EE.dat") or die $!;
foreach (sort(@entropylist)) {
	print FILEOUT "$_\n";
}
close(FILEOUT);




# Plot Entanglement spectrum and entropy
my @filelist;
my $directory = "0.ESCollector";
opendir (DIR, $directory) or die $!;
while (my $file = readdir(DIR)) {
	if ($file ne ".." and $file ne "." and $file ne "EE.dat") {
	print "$file\n";
        push @filelist, $file;
	}
}
closedir(DIR);

my $pgfplotsTemp = "EE.tex";
my $pgfplots = "PLOTEX.tex";
open(FILEOUT, ">", "$pgfplotsTemp") or die "$0: Cannot open: $pgfplotsTemp: $!\n";
open(FILEIN, "<","$pgfplots") or die "$0: Cannot open: $pgfplots: $!\n";
while(<FILEIN>) {
	print FILEOUT $_;
	if (/%%%%%% marker for perl, DO NOT DELET %%%%%%/){
			print FILEOUT "\\addplot [draw = red, mark = o] table [x index = 0, y index = 1, col sep=space] {0.ESCollector/EE.dat};\n";
		foreach (@filelist) {
			print FILEOUT "\\addplot [draw = black, only marks, mark = -] table [x index = 0, y index = 1, col sep=space] {0.ESCollector/$_};\n";
		}	
	}
}
close(FILEIN);
close(FILEOUT);

my $plotcommand = "pdflatex $pgfplotsTemp";
system($plotcommand);
system("rm *.log *.aux $pgfplotsTemp");










